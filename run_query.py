"""Agent 2 실행: HITL 수사 질의 루프.

사용: python run_query.py
- 질의 입력 → 생성 SQL·근거·결과 제시 → 수사관 승인/보완(HITL)
"""
from __future__ import annotations

from agent2.graph import build_query_graph
from storage import relational as R
from storage.vectorstore import get_store


def main():
    conn = R.connect()
    R.init_schema(conn)
    store = get_store()
    app = build_query_graph(conn, store)

    print("=== 수사 질의 시스템 (Agent 2, HITL) ===")
    print("질의를 입력하세요. 종료: 빈 줄 또는 'quit'\n")
    while True:
        try:
            query = input("질의> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit"):
            break

        out = app.invoke({"query": query})
        print("\n--- ReAct 추적 로그 ---")
        for line in out.get("log", []):
            print("  ·", line)
        print("\n--- 결과 ---")
        print(out.get("result_text", "(결과 없음)"))

        # HITL: 수사관 검토
        verdict = input("\n[검토] 승인(엔터) / 보완 요청(내용 입력)> ").strip()
        if verdict:
            # 보완 질의로 재실행 (HITL 루프)
            out2 = app.invoke({"query": f"{query} (보완: {verdict})"})
            print("\n--- 보완 결과 ---")
            print(out2.get("result_text", "(결과 없음)"))
        print("\n" + "=" * 50 + "\n")

    conn.close()
    print("종료.")


if __name__ == "__main__":
    main()
