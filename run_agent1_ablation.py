"""Agent1 ablation 전용 러너 — Agent2 없이 Agent1 CER/NER만 실행.

C1 비교 실험: 동일 파일에 graph(고정 파이프라인) vs toolcall(LLM 라우팅) 적용.
각 모드를 별도 DB/Milvus 컬렉션에 실행 → 결과 비교.

사용:
  # graph 모드 (기준선)
  python run_agent1_ablation.py --mode graph --out-tag graph

  # toolcall 모드 (제안) — 이미 run_experiment로 실행한 결과가 있다면 skip
  python run_agent1_ablation.py --mode toolcall --out-tag toolcall
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse, unquote


def log(msg: str) -> None:
    print(f"[a1abl] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["toolcall", "graph"], required=True)
    ap.add_argument("--out-tag", default=None, help="결과 디렉터리 접미사 (기본: mode명)")
    ap.add_argument("--docs-dir", default="data/eval_set")
    ap.add_argument("--ner-gold", default="eval/ner_gold.jsonl")
    ap.add_argument("--cer-ref", default="eval/cer_ref")
    ap.add_argument("--results-dir", default="eval/results")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    tag = args.out_tag or args.mode
    os.environ.setdefault("RELATIONAL_BACKEND", "postgres")
    os.environ["VECTOR_BACKEND"] = "milvus"

    if args.fake:
        import mock_llm
        mock_llm.apply_fakes()

    from config import CFG
    CFG.vector_backend = "milvus"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = Path(args.results_dir) / f"ablation_{tag}_{ts}"
    for sub in ("", "parsed", "tables"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    log(f"mode={args.mode}  run dir: {run}")

    # 새 DB 생성
    import psycopg2
    u = urlparse(CFG.database_url)
    admin = psycopg2.connect(
        host=u.hostname or "localhost", port=u.port or 5432,
        user=unquote(u.username or ""), password=unquote(u.password or ""),
        dbname=(u.path or "/").lstrip("/"), client_encoding="UTF8")
    admin.autocommit = True
    newdb = f"forensic_abl_{tag}_{ts}"
    with admin.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{newdb}"')
    admin.close()
    CFG.database_url = urlunparse(u._replace(path="/" + newdb))
    CFG.milvus_collection = f"abl_{tag}_{ts}"
    log(f"DB={newdb}  Milvus={CFG.milvus_collection}")

    # Agent1 실행
    from agent1.toolcall import ingest_dir_toolcall
    from agent1.graph import ingest_dir
    runner = ingest_dir_toolcall if args.mode == "toolcall" else ingest_dir
    log("Agent1 실행 중 ...")
    stats = runner(args.docs_dir, verbose=True)
    log(f"ingest stats: {stats}")

    # 파싱 텍스트 수집
    from storage import relational as R
    conn = R.connect()
    _, docs = R.execute_readonly(conn, "SELECT file_name, file_type, raw_text FROM documents")
    sys_text_by_stem, type_by_stem = {}, {}
    for d in docs:
        stem = os.path.splitext(d["file_name"])[0]
        txt = d.get("raw_text") or ""
        (run / "parsed" / f"{stem}.txt").write_text(txt, encoding="utf-8")
        sys_text_by_stem[stem] = txt
        type_by_stem[stem] = d.get("file_type", "?")
    _, ents = R.execute_readonly(conn, "SELECT COUNT(*) AS n FROM entities")
    n_ents = ents[0]["n"] if ents else 0
    try:
        conn.close()
    except Exception:
        pass
    log(f"docs={len(docs)} entities={n_ents}")

    # H1 CER
    import run_eval as RE
    cer_dir = Path(args.cer_ref)
    cer_metrics, cer_rows = {}, []
    ref_by_stem = {}
    if cer_dir.exists():
        for p in cer_dir.glob("*.txt"):
            t = p.read_text(encoding="utf-8").strip()
            if t:
                ref_by_stem[p.stem] = t
        if ref_by_stem:
            cer_metrics, cer_rows = RE.evaluate_cer(ref_by_stem, sys_text_by_stem, type_by_stem)
            with (run / "cer_per_file.jsonl").open("w", encoding="utf-8") as f:
                for r in cer_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"CER overall={cer_metrics.get('cer_overall'):.4f} (N={cer_metrics.get('cer_n_total')})")

    # H2 NER
    ner_metrics = {}
    ner_gold_path = Path(args.ner_gold)
    if ner_gold_path.exists():
        ner_gold = RE.load_ner_gold(ner_gold_path)
        if ner_gold:
            ner_metrics, ner_rows = RE.evaluate_ner(ner_gold)
            with (run / "ner_eval.jsonl").open("w", encoding="utf-8") as f:
                for r in ner_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"NER micro={ner_metrics.get('ner_recall_micro'):.3f}")

    # 결과 저장
    results = {
        "mode": args.mode, "ts": ts,
        "ingest_stats": stats,
        "h1_cer": cer_metrics,
        "h2_ner_recall": ner_metrics,
    }
    (run / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약 표
    lines = [f"# ablation_{tag}_{ts}", f"mode={args.mode}", ""]
    if cer_metrics:
        lines += ["## H1 CER (낮을수록 좋음)", "| 유형 | N | CER |", "|---|---|---|"]
        for t in ("pdf", "hwp", "audio", "image", "txt"):
            if f"cer_{t}" in cer_metrics:
                lines.append(f"| {t} | {int(cer_metrics[f'cer_n_{t}'])} | {cer_metrics[f'cer_{t}']:.4f} |")
        lines.append(f"| **전체** | {int(cer_metrics['cer_n_total'])} | **{cer_metrics['cer_overall']:.4f}** |")
        lines.append("")
    lines += ["## Agent1 처리 통계", f"ok={stats.get('ok')} low_quality={stats.get('low_quality')} "
              f"failed={stats.get('failed')} captioned={stats.get('captioned', 0)} "
              f"no_text={stats.get('no_text', 0)} duplicate={stats.get('duplicate')}"]
    (run / "tables" / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    log(f"완료 → {run}/results.json")
    log(f"  CER: pdf={cer_metrics.get('cer_pdf', '-'):.4f}  hwp={cer_metrics.get('cer_hwp', '-'):.4f}  "
        f"audio={cer_metrics.get('cer_audio', '-'):.4f}  image={cer_metrics.get('cer_image', '-'):.4f}  "
        f"txt={cer_metrics.get('cer_txt', '-'):.4f}")
    log(f"  전체={cer_metrics.get('cer_overall', '-'):.4f}")
    log(f"  색인: ok={stats.get('ok')}  captioned={stats.get('captioned', 0)}  "
        f"no_text={stats.get('no_text', 0)}  failed={stats.get('failed')}")


if __name__ == "__main__":
    main()
