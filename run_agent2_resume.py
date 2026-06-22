"""Agent2 재개 스크립트.

기존 run_experiment 결과 디렉터리(Agent1 완료 상태)에
Agent2 질의 평가(H3)만 추가 실행한 뒤 metrics.json / meta.json / summary.md 를 완성한다.

사용:
  python run_agent2_resume.py --run-dir eval/results/run_20260610_111131 \
      --db forensic_run_20260610_111131 \
      --milvus entities_run_20260610_111131 \
      --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


def log(msg: str) -> None:
    print(f"[a2] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="기존 run 결과 디렉터리")
    ap.add_argument("--db", required=True, help="Postgres DB 이름(forensic_run_...)")
    ap.add_argument("--milvus", required=True, help="Milvus 컬렉션(entities_run_...)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--queries", default="eval/queries.jsonl")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    run = Path(args.run_dir)
    if not run.exists():
        raise SystemExit(f"run-dir not found: {run}")

    # 환경 설정
    os.environ["RELATIONAL_BACKEND"] = "postgres"
    os.environ["VECTOR_BACKEND"] = "milvus"

    if args.fake:
        import mock_llm
        mock_llm.apply_fakes()

    # DB URL을 지정 DB로 교체
    from urllib.parse import urlparse, urlunparse, unquote
    from config import CFG
    CFG.vector_backend = "milvus"
    CFG.milvus_collection = args.milvus
    u = urlparse(CFG.database_url)
    CFG.database_url = urlunparse(u._replace(path="/" + args.db))
    log(f"DB: {args.db} | Milvus: {args.milvus}")

    import run_eval as RE
    from storage.vectorstore import get_store

    queries = RE.load_queries(Path(args.queries))
    log(f"queries={len(queries)} repeats={args.repeats} topk={args.topk}")

    # Agent2 K회 반복
    q_runs = []
    all_rows = []
    for i in range(args.repeats):
        store = get_store()
        qm, qrows = RE.evaluate_queries(queries, store, args.topk)
        q_runs.append(qm)
        for r in qrows:
            r["_repeat"] = i + 1
        all_rows += qrows
        log(f"  [r{i+1}] Recall@{args.topk}={qm['retrieval_recall_at_k']:.3f} "
            f"SQL={qm['sql_exec_success']:.3f} Answer={qm['answer_hit']:.3f}")

    (run / "queries_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows),
        encoding="utf-8")

    def agg(key):
        vals = [r[key] for r in q_runs if key in r]
        if not vals:
            return None
        return {"mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "values": [round(v, 4) for v in vals]}

    q_keys = sorted({k for r in q_runs for k in r})
    q_summary = {k: agg(k) for k in q_keys}

    # 기존 metrics.json에서 agent1/H1/H2 읽기(없으면 빈 dict)
    existing_metrics: dict = {}
    existing_meta: dict = {}
    if (run / "metrics.json").exists():
        existing_metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if (run / "meta.json").exists():
        existing_meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))

    # cer/ner 지표: cer_per_file.jsonl + ner_metrics_canonical.json에서 재로드
    cer_metrics: dict = existing_metrics.get("h1_cer", {})
    ner_metrics: dict = existing_metrics.get("h2_ner_recall", {})
    canonical_path = run / "ner_metrics_canonical.json"
    if canonical_path.exists():
        ner_metrics = json.loads(canonical_path.read_text(encoding="utf-8"))
        log("NER: canonical metrics loaded")

    metrics = {
        "agent1_parse": existing_metrics.get("agent1_parse", {}),
        "h1_cer": cer_metrics,
        "h2_ner_recall": ner_metrics,
        "h3_query": q_summary,
    }
    (run / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"metrics.json written")

    # meta.json 갱신
    existing_meta.update({
        "agent2_repeats": args.repeats,
        "agent2_topk": args.topk,
        "agent2_fake": args.fake,
        "agent2_db": args.db,
        "agent2_milvus": args.milvus,
    })
    (run / "meta.json").write_text(
        json.dumps(existing_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary.md 갱신
    (run / "tables").mkdir(exist_ok=True)
    lines = [f"# {run.name} 요약", ""]
    if cer_metrics:
        lines += ["## H1 CER", "| 유형 | N | CER |", "|---|---|---|"]
        for t in ("pdf", "hwp", "audio", "image", "txt"):
            if f"cer_{t}" in cer_metrics:
                lines.append(f"| {t} | {int(cer_metrics[f'cer_n_{t}'])} | {cer_metrics[f'cer_{t}']:.4f} |")
        lines += [f"| **전체** | {int(cer_metrics.get('cer_n_total',0))} | **{cer_metrics.get('cer_overall',0):.4f}** |", ""]
    if ner_metrics:
        lines += ["## H2 NER Recall", "| 유형 | Recall |", "|---|---|"]
        for t in RE.NER_TYPES:
            lines.append(f"| {t} | {ner_metrics.get(f'ner_recall_{t}', 0):.3f} |")
        lines += [f"| **micro** | **{ner_metrics.get('ner_recall_micro',0):.3f}** |",
                  f"| macro | {ner_metrics.get('ner_recall_macro',0):.3f} |", ""]
    lines += ["## H3 질의 (K회 평균±표준편차)", "| 지표 | mean | std |", "|---|---|---|"]
    for k in ("classification_accuracy", "retrieval_recall_at_k", "retrieval_mrr",
              "sql_exec_success", "answer_hit", "answer_hit_T1", "answer_hit_T2", "answer_hit_T3"):
        if q_summary.get(k):
            lines.append(f"| {k} | {q_summary[k]['mean']:.3f} | {q_summary[k]['std']:.3f} |")
    (run / "tables" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"summary.md written")

    # 주요 지표 출력
    ans = q_summary.get("answer_hit", {})
    log(f"=== 완료 ===")
    log(f"  answer_hit  mean={ans.get('mean','-'):.3f} std={ans.get('std','-'):.3f}")
    log(f"  sql_exec    mean={q_summary.get('sql_exec_success',{}).get('mean',0):.3f}")
    log(f"  recall@{args.topk}  mean={q_summary.get('retrieval_recall_at_k',{}).get('mean',0):.3f}")
    log(f"결과: {run}/metrics.json")


if __name__ == "__main__":
    main()
