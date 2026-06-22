"""오프라인 스모크 테스트: API 키 없이 가짜 LLM으로 전체 파이프라인 배선·로직 검증.

실제 LLM 호출부(임베딩/NER/분류/Text-to-SQL)만 결정론적 가짜로 교체하고,
LangGraph 라우팅·fallback·검색정책·SQL 실행·검증은 실제 코드로 돌린다.
"""
from __future__ import annotations
import hashlib
import os
import re
import tempfile

# --- config 로드 전에 환경 설정 (임시 DB, 로컬 벡터) ---
_tmp = tempfile.mkdtemp()
os.environ["DB_PATH"] = os.path.join(_tmp, "smoke.db")
os.environ["VECTOR_BACKEND"] = "local"
os.environ.setdefault("OPENAI_API_KEY", "test-dummy")

import agent1.graph as a1g          # noqa: E402
import agent1.tools as a1t          # noqa: E402
import agent2.tools as a2t          # noqa: E402
from agent1.graph import ingest_dir  # noqa: E402
from agent2.graph import answer      # noqa: E402

DIM = 32
PERSONS = ["홍길동", "김철수", "이영희", "박민수", "정대만"]
PLACES = ["강남역", "부산", "해운대", "서울", "영등포", "여의도"]


def fake_embed(texts, kind="query"):
    vecs = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(DIM)]
        vecs.append(v)
    return vecs


def fake_ner(text):
    ents = []
    def add(val, typ):
        ents.append({"entity_text": val, "entity_type": typ, "context": text[:60]})
    for d in re.findall(r"\d{4}-\d{2}-\d{2}", text):
        add(d, "날짜")
    for a in re.findall(r"[\d,]+원", text):
        add(a, "금액")
    for ac in re.findall(r"\d{2,4}-\d{2,4}-\d{3,6}", text):
        add(ac, "계좌")
    for p in PERSONS:
        if p in text:
            add(p, "인물")
    for pl in PLACES:
        if pl in text:
            add(pl, "장소")
    # 중복 제거
    seen, out = set(), []
    for e in ents:
        k = (e["entity_text"], e["entity_type"])
        if k not in seen:
            seen.add(k); out.append(e)
    return out


def fake_classify(query):
    if "관련" in query:
        return {"type": "fuzzy", "reason": "fake"}
    if any(k in query for k in ["가장", "몇", "순", "카운트"]):
        return {"type": "aggregate", "reason": "fake"}
    if "문서" in query:
        return {"type": "relational", "reason": "fake"}
    return {"type": "lookup", "reason": "fake"}


def _token(query):
    for p in PERSONS + PLACES:
        if p in query:
            return p
    m = re.search(r"[가-힣]{2,4}", query)
    return m.group(0) if m else ""


def fake_sql(query, hints=None, relax=False):
    base = ("SELECT e.entity_text, e.entity_type, e.doc_id, d.file_name "
            "FROM entities e JOIN documents d ON e.doc_id = d.doc_id")
    if hints:
        vals = ",".join("'" + h.replace("'", "") + "'" for h in hints)
        return f"{base} WHERE e.entity_text IN ({vals})"
    tok = _token(query)
    if not tok:
        return base
    if relax:
        return f"{base} WHERE e.entity_text LIKE '%{tok}%' OR e.context LIKE '%{tok}%'"
    return f"{base} WHERE e.entity_text = '{tok}'"


def patch():
    a1g.embed = fake_embed
    a1t.run_ner = fake_ner
    a2t.embed = fake_embed
    a2t.classify_query = fake_classify
    a2t.text_to_sql = fake_sql


def main():
    patch()
    print("### 1) Agent 1 수집 (라우팅·품질·NER·적재) ###")
    stats = ingest_dir("data/sample", verbose=True)
    print("\n수집 요약:", stats)
    assert stats["files"] >= 4, "샘플 파일 수집 실패"
    assert stats["ok"] >= 3, "정상 적재 부족"
    assert stats["duplicate"] >= 1, "중복 제거 미작동(dup 파일)"
    assert stats["entities"] >= 8, "개체 추출 부족"

    print("\n### 2) Agent 2 질의 (분류·검색정책·SQL·검증) ###")
    for q in [
        "홍길동 계좌 보여줘",                 # lookup
        "홍길동과 관련된 계좌",               # fuzzy → 하이브리드 해소
        "김철수가 등장한 문서",               # relational
    ]:
        out = answer(q)
        print(f"\n[Q] {q}")
        for line in out.get("log", []):
            print("   ·", line)
        print("   결과행수:", len(out.get("rows", [])))
        assert out.get("sql"), "SQL 생성 실패"

    print("\n=== 스모크 테스트 통과 ===")


if __name__ == "__main__":
    main()
