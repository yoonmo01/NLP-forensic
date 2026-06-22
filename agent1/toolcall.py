"""Agent 1 (tool-calling 버전): LLM이 직접 도구를 호출해 수집·구조화.

LangGraph 결정론적 그래프(graph.py)와 달리, **LLM이 function calling으로 도구를 선택**한다.
- 의사결정(어느 파서·fallback 여부·종료)을 LLM이 수행 → 진짜 'tool-using agent'.
- 무거운 텍스트는 서버측 컨텍스트에 보관하고, 도구 결과는 메타데이터(글자수·품질)만 반환.
- 모든 도구 호출은 trace 로그로 남겨 포렌식 재현성/감사성 확보.
- 온도 0 + seed(config)로 일관성.

키 필요(LLM tool calling). 디렉터리 단위는 ingest_dir_toolcall 사용.
"""
from __future__ import annotations
import json
import os
import re

# 마크다운 이미지 플레이스홀더 — Upstage가 글자 없는 사진에 반환하는 잡음
_PLACEHOLDER_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _meaningful_text(t: str) -> str:
    """플레이스홀더 제거 후 남는 실질 텍스트."""
    return _PLACEHOLDER_RE.sub("", t or "").strip()

from config import CFG
from llm import client, embed
from storage import relational as R
from . import tools as T

MAX_STEPS = 10

SYSTEM = (
    "너는 디지털 포렌식 수집 에이전트다. 주어진 파일 하나를 처리한다. 반드시 제공된 도구만 사용하라.\n"
    "절차: (1) detect_type로 유형과 가능한 parse 전략을 확인한다. "
    "(2) parse로 첫 전략을 시도한다. (3) 반환된 quality가 0.6 미만이면 남은 다른 전략으로 parse를 재시도한다"
    "(가능한 전략을 소진할 때까지). (4) 충분한 품질의 텍스트가 확보되면 extract_entities를 호출한다. "
    "(5) 마지막으로 finalize를 호출해 적재한다. 도구 없이 내용을 지어내지 말 것. finalize 후 종료."
)

TOOLSPECS = [
    {"type": "function", "function": {
        "name": "detect_type", "description": "파일 유형 감지 및 가능한 parse 전략 목록 반환",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "parse", "description": "지정한 전략으로 파일을 파싱하고 추출 품질을 반환",
        "parameters": {"type": "object", "properties": {
            "strategy": {"type": "string", "description": "detect_type가 알려준 전략 중 하나"}},
            "required": ["strategy"]}}},
    {"type": "function", "function": {
        "name": "extract_entities", "description": "현재 보관된 텍스트에서 NER로 증거 개체 추출",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "finalize", "description": "문서·개체를 DB에 적재하고 벡터 색인 후 종료",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


def ingest_file_toolcall(path, conn, store, seen_fps, *, verbose=False):
    ctx = {"text": "", "quality": 0.0, "file_type": None, "entities": [],
           "strategies": [], "tried": []}
    trace = []

    def _finalize():
        """문서를 반드시 색인. 텍스트가 비면 이미지는 VLM 캡션, 그 외는 마커로 기록(전체 파일 색인)."""
        text = ctx["text"]
        extra = None
        meaningful = _meaningful_text(text)
        is_img = ctx["file_type"] == "image"
        # 이미지: 실질 텍스트가 거의 없으면(플레이스홀더/짧은 잡음) VLM 캡션으로 대체.
        # 그 외 파일: 완전히 비었을 때만 마커.
        if (is_img and len(meaningful) < 20) or (not is_img and not meaningful):
            if is_img:
                try:
                    cap = T.vlm_caption_image(path)
                except Exception:
                    cap = ""
                text = ("[이미지 설명] " + cap) if cap.strip() else "[이미지: 설명 생성 실패]"
                extra = "captioned"
            else:
                text = "[추출 텍스트 없음]"
                extra = "no_text"
        fp = T.text_fingerprint(text)
        if fp in seen_fps:
            return {"status": "duplicate"}
        seen_fps.add(fp)
        doc_id = R.insert_document(conn, os.path.basename(path), ctx["file_type"] or "unknown", text)
        ids = R.insert_entities(conn, doc_id, ctx["entities"])
        if ids:
            recs = [R.fetch_entity(conn, i) for i in ids]
            vecs = embed([f"{r['entity_text']} {r['context']}" for r in recs], kind="passage")
            for r, v in zip(recs, vecs):
                r["vector"] = v
            store.add(recs)
        st = extra or ("ok" if ctx["quality"] >= CFG.parse_quality_threshold else "low_quality")
        return {"status": st, "doc_id": doc_id, "entities": len(ids)}

    def dispatch(name, args):
        if name == "detect_type":
            ft = T.detect_file_type(path)
            ctx["file_type"] = ft
            ctx["strategies"] = list(T.STRATEGIES.get(ft, ["txt"]))
            return {"file_type": ft, "strategies": ctx["strategies"]}
        if name == "parse":
            strat = args.get("strategy", "")
            if strat not in T.PARSERS:
                return {"error": f"알 수 없는 전략: {strat}", "available": ctx["strategies"]}
            ctx["tried"].append(strat)
            try:
                text = T.PARSERS[strat](path)
                q = T.assess_extraction_quality(text)
                if q > ctx["quality"]:
                    ctx["text"], ctx["quality"] = text, q
                return {"strategy": strat, "ok": True, "chars": len(text), "quality": q}
            except Exception as e:
                return {"strategy": strat, "ok": False, "error": type(e).__name__,
                        "remaining": [s for s in ctx["strategies"] if s not in ctx["tried"]]}
        if name == "extract_entities":
            ents = T.run_ner(ctx["text"])
            ctx["entities"] = ents
            return {"count": len(ents)}
        if name == "finalize":
            return _finalize()
        return {"error": f"알 수 없는 도구: {name}"}

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"다음 파일을 처리하라. 경로: {path}"}]
    params = {"model": CFG.llm_model, "temperature": 0, "tools": TOOLSPECS, "tool_choice": "auto"}
    if CFG.seed is not None:
        params["seed"] = CFG.seed

    result = {"status": "failed", "trace": trace}
    for _ in range(MAX_STEPS):
        resp = client().chat.completions.create(messages=messages, **params)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            break
        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            out = dispatch(tc.function.name, args)
            trace.append(f"{tc.function.name}({args})→{out}")
            if verbose:
                print(f"     · {tc.function.name}({args}) → {out}")
            if tc.function.name == "finalize":
                result = {**out, "trace": trace, "entities_list": ctx["entities"]}
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(out, ensure_ascii=False)})
        if result.get("status") in ("ok", "low_quality", "duplicate", "captioned", "no_text") and \
           any("finalize" in t for t in trace):
            break
    # 사후 폴백: 에이전트가 finalize를 호출하지 않았어도 반드시 색인(전체 파일 보장)
    if result.get("status") not in ("ok", "low_quality", "duplicate", "captioned", "no_text"):
        try:
            out = _finalize()
            result = {**out, "trace": trace, "entities_list": ctx["entities"]}
            trace.append(f"_finalize(fallback)→{out}")
        except Exception as e:
            result = {"status": "failed", "reason": f"fallback: {e}",
                      "trace": trace, "entities_list": ctx["entities"]}
    return result


def ingest_dir_toolcall(input_dir, *, verbose=True):
    conn = R.connect(); R.init_schema(conn)
    from storage.vectorstore import get_store
    store = get_store()
    seen = set()
    for txt in R.all_raw_texts(conn):
        seen.add(T.text_fingerprint(txt))

    stats = {"ok": 0, "low_quality": 0, "duplicate": 0, "failed": 0,
             "files": 0, "skipped": 0, "entities": 0}
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if d.lower() not in T.EXCLUDE_DIRS]
        if T.is_excluded(root):
            continue
        for fn in files:
            if fn.lower() == "manifest.csv" or os.path.splitext(fn)[1].lower() not in T.INGEST_EXTS:
                stats["skipped"] += 1
                continue
            path = os.path.join(root, fn)
            stats["files"] += 1
            if verbose:
                print(f"[file] {fn}")
            try:
                out = ingest_file_toolcall(path, conn, store, seen, verbose=verbose)
            except Exception as e:
                stats["failed"] += 1
                if verbose:
                    print(f"   예외: {e}")
                continue
            st = out.get("status", "failed")
            stats[st] = stats.get(st, 0) + 1
            stats["entities"] += len(out.get("entities_list", []))
    conn.close()
    return stats
