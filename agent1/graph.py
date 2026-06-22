"""Agent 1: 라우팅·실패복구 수집 에이전트 (LangGraph).

State/Node/Conditional Edge로 '의사결정 지점'과 '실패 처리(fallback)'를 명시.
흐름: detect → parse → assess →(품질 OK?)→ ner → persist → END
                         └(저품질 & 대체전략 잔여)→ parse (fallback)
"""
from __future__ import annotations
from typing import TypedDict

from langgraph.graph import StateGraph, END

from config import CFG
from llm import embed
from storage import relational as R
from . import tools as T


class IngestState(TypedDict, total=False):
    path: str
    file_type: str
    strategies: list[str]   # 남은 전략 큐
    tried: list[str]
    text: str
    quality: float
    status: str             # ok | duplicate | low_quality | failed
    error: str
    entities: list[dict]
    doc_id: int
    log: list[str]          # 의사결정 감사 로그


def build_ingest_graph(conn, store, seen_fps: set[str]):
    def n_detect(s: IngestState) -> IngestState:
        ft = T.detect_file_type(s["path"])
        return {"file_type": ft, "strategies": list(T.STRATEGIES.get(ft, ["txt"])),
                "tried": [], "log": [f"detect: {ft}"]}

    def n_parse(s: IngestState) -> IngestState:
        strat = s["strategies"][0]
        rest = s["strategies"][1:]
        log = s.get("log", []) + [f"parse 시도: {strat}"]
        try:
            text = T.PARSERS[strat](s["path"])
            return {"text": text, "tried": s["tried"] + [strat],
                    "strategies": rest, "log": log + [f"parse 성공: {strat} ({len(text)}자)"]}
        except Exception as e:
            return {"text": "", "tried": s["tried"] + [strat], "strategies": rest,
                    "error": f"{strat}:{e}", "log": log + [f"parse 실패: {strat} ({e})"]}

    def n_assess(s: IngestState) -> IngestState:
        q = T.assess_extraction_quality(s.get("text", ""))
        return {"quality": q, "log": s["log"] + [f"품질 평가: {q}"]}

    def route_after_assess(s: IngestState) -> str:
        if s["quality"] >= CFG.parse_quality_threshold:
            return "ner"
        if s["strategies"]:                      # 대체 전략 남음 → fallback
            return "fallback"
        return "ner_lowq"                        # 더 없음 → 저품질이라도 진행

    def n_ner(s: IngestState) -> IngestState:
        ents = T.run_ner(s.get("text", ""))
        return {"entities": ents, "log": s["log"] + [f"NER: {len(ents)}개 개체"]}

    def n_persist(s: IngestState) -> IngestState:
        text = s.get("text", "")
        fp = T.text_fingerprint(text)
        if fp in seen_fps:
            return {"status": "duplicate", "log": s["log"] + ["중복 문서 → 적재 생략"]}
        if not text.strip():
            return {"status": "failed", "log": s["log"] + ["빈 텍스트 → 적재 생략"]}
        seen_fps.add(fp)
        import os
        doc_id = R.insert_document(conn, os.path.basename(s["path"]), s["file_type"], text)
        ents = s.get("entities", [])
        ids = R.insert_entities(conn, doc_id, ents)
        # 임베딩 + 벡터 색인
        if ids:
            recs = [R.fetch_entity(conn, i) for i in ids]
            vecs = embed([f"{r['entity_text']} {r['context']}" for r in recs], kind="passage")
            for r, v in zip(recs, vecs):
                r["vector"] = v
            store.add(recs)
        status = "ok" if s["quality"] >= CFG.parse_quality_threshold else "low_quality"
        return {"doc_id": doc_id, "status": status,
                "log": s["log"] + [f"적재 완료 doc_id={doc_id}, 개체 {len(ids)}건"]}

    g = StateGraph(IngestState)
    g.add_node("detect", n_detect)
    g.add_node("parse", n_parse)
    g.add_node("assess", n_assess)
    g.add_node("ner", n_ner)
    g.add_node("persist", n_persist)
    g.set_entry_point("detect")
    g.add_edge("detect", "parse")
    g.add_edge("parse", "assess")
    g.add_conditional_edges("assess", route_after_assess,
                            {"ner": "ner", "ner_lowq": "ner", "fallback": "parse"})
    g.add_edge("ner", "persist")
    g.add_edge("persist", END)
    return g.compile()


def ingest_dir(input_dir: str, *, verbose: bool = True) -> dict:
    """디렉터리 내 파일을 일괄 수집. 요약 통계 반환."""
    import os
    from storage.vectorstore import get_store

    conn = R.connect()
    R.init_schema(conn)
    store = get_store()
    # 기존 문서 지문 로드(교차-실행 중복 방지)
    seen: set[str] = set()
    for txt in R.all_raw_texts(conn):
        seen.add(T.text_fingerprint(txt))

    app = build_ingest_graph(conn, store, seen)
    stats = {"ok": 0, "low_quality": 0, "duplicate": 0, "failed": 0,
             "files": 0, "skipped": 0, "entities": 0}
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if d.lower() not in T.EXCLUDE_DIRS]  # 시스템 폴더 가지치기
        if T.is_excluded(root):
            continue
        for fn in files:
            if fn.lower() == "manifest.csv" or os.path.splitext(fn)[1].lower() not in T.INGEST_EXTS:
                stats["skipped"] += 1
                continue
            path = os.path.join(root, fn)
            stats["files"] += 1
            try:
                out = app.invoke({"path": path})
            except Exception as e:
                stats["failed"] += 1
                if verbose:
                    print(f"[X] {fn}: 예외 {e}")
                continue
            st = out.get("status", "failed")
            stats[st] = stats.get(st, 0) + 1
            stats["entities"] += len(out.get("entities", [])) if st in ("ok", "low_quality") else 0
            if verbose:
                print(f"[{st}] {fn}")
                for line in out.get("log", []):
                    print(f"     - {line}")
    conn.close()
    return stats
