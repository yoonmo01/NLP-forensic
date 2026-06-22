"""Agent 1 실행: 디렉터리의 압수 파일을 파싱·구조화하여 DB+벡터 색인에 적재.

사용: python run_ingest.py --input-dir data/sample
"""
from __future__ import annotations
import argparse

from agent1.graph import ingest_dir
from agent1.toolcall import ingest_dir_toolcall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/sample", help="압수 파일 디렉터리")
    ap.add_argument("--mode", choices=["toolcall", "graph"], default="toolcall",
                    help="toolcall=LLM이 직접 도구 호출(기본) / graph=결정론적 라우터(베이스라인)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print(f"[Agent 1 / {args.mode}] 수집 시작: {args.input_dir}")
    runner = ingest_dir_toolcall if args.mode == "toolcall" else ingest_dir
    stats = runner(args.input_dir, verbose=not args.quiet)
    print("\n=== 수집 요약 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
