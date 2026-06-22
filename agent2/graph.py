"""Agent 2: ReAct 수사 질의 에이전트 (LangGraph) + 검색정책(narrow→broaden).

흐름: classify →(fuzzy?)→ resolve → gen_sql → exec → decide
  decide: 결과 있음→verify→format / 결과 없음 & 반복 여유→ (개체해소 or 완화) 재시도
검색 깊이 제어: 고정밀(정확 일치) 시작 → 0건이면 하이브리드 해소 → LIKE 완화.
"""
from __future__ import annotations
from typing import TypedDict

from langgraph.graph import StateGraph, END

from config import CFG
from storage import relational as R
from . import tools as T


class QueryState(TypedDict, total=False):
    query: str
    qtype: str
    hints: list[str]
    relax: bool
    sql: str
    cols: list[str]
    rows: list[dict]
    error: str
    iters: int
    verify: dict
    result_text: str
    log: list[str]
    _next: str


def build_query_graph(conn, store):
    def n_classify(s: QueryState) -> QueryState:
        c = T.classify_query(s["query"])
        return {"qtype": c["type"], "iters": 0, "hints": [], "relax": False,
                "log": [f"분류: {c['type']} ({c['reason']})"]}

    def route_after_classify(s: QueryState) -> str:
        return "resolve" if s["qtype"] == "fuzzy" else "gen_sql"

    def n_resolve(s: QueryState) -> QueryState:
        hits = T.resolve_entity(s["query"], store)
        hints = list(dict.fromkeys(h["entity_text"] for h in hits))  # 중복 제거, 순서 유지
        return {"hints": hints,
                "log": s["log"] + [f"하이브리드 개체 해소: {hints[:CFG.topk]}"]}

    def n_gen_sql(s: QueryState) -> QueryState:
        sql = T.text_to_sql(s["query"], hints=s.get("hints"), relax=s.get("relax", False))
        return {"sql": sql, "log": s["log"] + [f"SQL 생성{' (완화)' if s.get('relax') else ''}: {sql}"]}

    def n_exec(s: QueryState) -> QueryState:
        try:
            cols, rows = T.execute_sql(conn, s["sql"])
            return {"cols": cols, "rows": rows, "error": "",
                    "log": s["log"] + [f"실행: {len(rows)}건"]}
        except Exception as e:
            return {"cols": [], "rows": [], "error": str(e),
                    "log": s["log"] + [f"실행 오류: {e}"]}

    def n_decide(s: QueryState) -> QueryState:
        rows = s.get("rows", [])
        iters = s.get("iters", 0)
        if rows:
            return {"_next": "verify"}
        if iters >= CFG.search_max_iters:
            return {"_next": "verify", "log": s["log"] + ["최대 반복 도달 → 종료"]}
        iters += 1
        # broaden 단계: 1) 아직 개체 해소 안 했으면 하이브리드 해소  2) 했으면 LIKE 완화
        if not s.get("hints"):
            return {"_next": "resolve", "iters": iters,
                    "log": s["log"] + ["결과 0건 → broaden: 하이브리드 개체 해소"]}
        if not s.get("relax"):
            return {"_next": "gen_sql", "iters": iters, "relax": True,
                    "log": s["log"] + ["결과 0건 → broaden: LIKE 부분일치 완화"]}
        return {"_next": "verify", "iters": iters,
                "log": s["log"] + ["추가 완화 불가 → 종료"]}

    def n_verify(s: QueryState) -> QueryState:
        v = T.verify_result(s.get("rows", []), query=s.get("query"), cols=s.get("cols", []))
        return {"verify": v, "log": s["log"] + [f"검증: {v['reason']}"]}

    def n_format(s: QueryState) -> QueryState:
        txt = T.format_result(s["query"], s.get("sql", ""), s.get("cols", []),
                              s.get("rows", []), s.get("verify", {}))
        return {"result_text": txt}

    g = StateGraph(QueryState)
    for name, fn in [("classify", n_classify), ("resolve", n_resolve), ("gen_sql", n_gen_sql),
                     ("exec", n_exec), ("decide", n_decide), ("verify", n_verify),
                     ("format", n_format)]:
        g.add_node(name, fn)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route_after_classify,
                            {"resolve": "resolve", "gen_sql": "gen_sql"})
    g.add_edge("resolve", "gen_sql")
    g.add_edge("gen_sql", "exec")
    g.add_edge("exec", "decide")
    g.add_conditional_edges("decide", lambda s: s["_next"],
                            {"verify": "verify", "resolve": "resolve", "gen_sql": "gen_sql"})
    g.add_edge("verify", "format")
    g.add_edge("format", END)
    return g.compile()


def answer(query: str) -> dict:
    """단일 질의 처리(비대화). 결과 dict 반환."""
    from storage.vectorstore import get_store
    conn = R.connect()
    R.init_schema(conn)
    store = get_store()
    app = build_query_graph(conn, store)
    out = app.invoke({"query": query})
    conn.close()
    return out
