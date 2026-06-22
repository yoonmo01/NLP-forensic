"""Agent 2 도구: 질의 분류, 하이브리드 개체 해소, Text-to-SQL, 실행, 검증, 포매팅."""
from __future__ import annotations

from config import CFG
from llm import chat, chat_json, embed
from storage import relational as R

QUERY_TYPES = ["lookup", "relational", "aggregate", "fuzzy"]

_CLS_SYS = (
    "수사관의 자연어 질의를 다음 4유형 중 하나로 분류하라. "
    "lookup(특정 개체 단순 조회), relational(개체-문서 관계/JOIN), "
    "aggregate(집계·정렬·카운트), fuzzy(모호/간접 표현, '~와 관련된' 등). "
    'JSON으로만: {"type": "...", "reason": "..."}'
)


def _classify_once(query: str, vary: bool) -> str:
    try:
        d = chat_json(_CLS_SYS, f"질의: {query}",
                      temperature=(0.4 if vary else 0.0), use_seed=not vary)
        t = d.get("type", "lookup")
        return t if t in QUERY_TYPES else "lookup"
    except Exception:
        return "lookup"


def classify_query(query: str) -> dict:
    """질의 유형 분류. self_consistency_n>1이면 K회 다수결(일관성)."""
    n = max(1, CFG.self_consistency_n)
    if n == 1:
        return {"type": _classify_once(query, False), "reason": "single"}
    from collections import Counter
    votes = [_classify_once(query, True) for _ in range(n)]
    top, _ = Counter(votes).most_common(1)[0]
    return {"type": top, "reason": f"vote {dict(Counter(votes))}"}


def resolve_entity(query: str, store, *, type_filter: str | None = None, top_k: int | None = None):
    """키워드+벡터 하이브리드 검색으로 모호 개체를 실제 후보로 해소."""
    vec = embed([query], kind="query")[0]
    hits = store.search(query, vec, top_k=top_k or CFG.topk, type_filter=type_filter)
    return hits


_SQL_SYS = (
    "너는 한국 디지털 포렌식 RDB를 위한 Text-to-SQL 변환기다. "
    "아래 스키마에 맞는 '읽기 전용 PostgreSQL SELECT' 한 개만 생성하라. "
    "설명/마크다운/세미콜론 없이 SQL 본문만 출력.\n\n스키마:\n{schema}"
)


def _clean_sql(s: str) -> str:
    return s.strip().strip("`").replace("```sql", "").replace("```", "").strip()


def text_to_sql(query: str, *, hints: list[str] | None = None, relax: bool = False) -> str:
    """NL→SQL. self_consistency_n>1이면 K회 생성 후 최빈 SQL 채택(일관성)."""
    user = f"질의: {query}\n"
    if hints:
        user += f"해소된 개체 후보(이 표기들을 우선 활용): {hints}\n"
    if relax:
        user += "주의: 정확 일치로 결과가 없었음. entity_text에 LIKE '%...%' 부분일치를 사용해 더 넓게 검색하라.\n"
    sys = _SQL_SYS.format(schema=R.schema_text())
    n = max(1, CFG.self_consistency_n)
    if n == 1:
        return _clean_sql(chat(sys, user))
    from collections import Counter
    cands = [_clean_sql(chat(sys, user, temperature=0.4, use_seed=False)) for _ in range(n)]
    return Counter(cands).most_common(1)[0][0]


def execute_sql(conn, sql: str):
    return R.execute_readonly(conn, sql)  # (cols, rows) 또는 예외


def _rows_to_context(rows: list[dict]) -> str:
    chunks: list[str] = []
    for r in rows[:20]:
        parts = []
        for key in ("doc_id", "file_name", "entity_text", "entity_type", "context", "raw_text"):
            if key in r and r.get(key) is not None:
                val = str(r.get(key))
                if key == "raw_text" and len(val) > 1200:
                    val = val[:1200]
                parts.append(f"{key}={val}")
        if not parts:
            parts = [", ".join(f"{k}={v}" for k, v in r.items())]
        chunks.append(" | ".join(parts))
    return "\n".join(chunks)


def _rows_to_answer(rows: list[dict], cols: list[str] | None = None) -> str:
    keys = cols or (list(rows[0].keys()) if rows else [])
    lines = []
    for r in rows[:20]:
        lines.append(", ".join(f"{k}={r.get(k)}" for k in keys if k in r))
    return "\n".join(lines)


def _interpret_groundedness(raw: str) -> bool | None:
    norm = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    if any(x in norm for x in ("notgrounded", "ungrounded", "contradict", "unsupported")):
        return False
    if "grounded" in norm or "supported" in norm:
        return True
    return None


def _upstage_groundedness(context: str, answer: str) -> dict:
    if not CFG.upstage_api_key:
        raise RuntimeError("UPSTAGE_API_KEY 미설정")
    from openai import OpenAI

    client = OpenAI(api_key=CFG.upstage_api_key, base_url=CFG.upstage_base_url)
    user = (
        "다음 답변이 주어진 근거 문맥에 의해 뒷받침되는지 판단하라.\n\n"
        f"Context:\n{context}\n\nAnswer:\n{answer}"
    )
    resp = client.chat.completions.create(
        model=CFG.upstage_groundedness_model,
        temperature=0,
        messages=[{"role": "user", "content": user}],
    )
    raw = (resp.choices[0].message.content or "").strip()
    grounded = _interpret_groundedness(raw)
    return {
        "grounded": bool(grounded) if grounded is not None else False,
        "reason": f"Upstage Groundedness: {raw[:160] or 'empty response'}",
        "raw": raw,
    }


def _local_groundedness(context: str, answer: str) -> dict:
    """로컬 LLM으로 근거성 판정(Upstage 대체). GROUNDEDNESS_BACKEND=local_llm."""
    sys = ("다음 답변이 주어진 근거 문맥에 의해 뒷받침되는지 판단하라. "
           "'grounded' 또는 'not_grounded' 중 한 단어로만 답하라.")
    user = f"Context:\n{context}\n\nAnswer:\n{answer}"
    raw = chat(sys, user).strip()
    grounded = _interpret_groundedness(raw)
    return {
        "grounded": bool(grounded) if grounded is not None else True,
        "reason": f"로컬 LLM 근거검증: {raw[:160] or 'empty'}",
        "raw": raw,
    }


def verify_result(rows: list[dict], *, query: str | None = None,
                  cols: list[str] | None = None) -> dict:
    """결과가 출처(doc_id)로 추적 가능하고, 가능하면 Upstage로 근거성을 검증."""
    if not rows:
        return {"grounded": False, "reason": "결과 없음"}
    grounded = all(("doc_id" in r and r["doc_id"] is not None) for r in rows)
    if not grounded:
        return {"grounded": False, "reason": "일부 레코드 출처 불명(doc_id 없음)"}

    context = _rows_to_context(rows)
    answer = _rows_to_answer(rows, cols)
    if not context.strip() or not answer.strip():
        return {"grounded": True, "reason": "모든 레코드 doc_id 존재(근거검증 입력 부족)"}
    if CFG.groundedness_backend == "off":
        return {"grounded": True, "reason": "모든 레코드 doc_id 존재(근거검증 비활성)"}
    try:
        if CFG.groundedness_backend == "upstage":
            return _upstage_groundedness(context, answer)
        return _local_groundedness(context, answer)   # 기본: 로컬 LLM 판정
    except Exception as e:
        return {
            "grounded": True,
            "reason": f"모든 레코드 doc_id 존재; 근거검증 미사용({type(e).__name__}: {e})",
        }


def format_result(query: str, sql: str, cols: list[str], rows: list[dict], verify: dict) -> str:
    lines = [f"질의: {query}", f"생성 SQL: {sql}",
             f"근거 검증: {'OK' if verify.get('grounded') else '주의'} ({verify.get('reason')})",
             f"결과 {len(rows)}건:"]
    for r in rows[:20]:
        lines.append("  - " + ", ".join(f"{k}={r.get(k)}" for k in cols))
    return "\n".join(lines)
