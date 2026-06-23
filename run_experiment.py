"""실험 오케스트레이터 (확정 설계).

설계:
  - DB(파싱+NER, Agent1 영역)는 1회 구축. Agent2 질의만 K회 반복.
  - 백엔드는 .env(CFG)를 따름. 기본 = SQLite + 로컬 하이브리드(무Docker, DGX Spark 로컬 실행).
    옵션 = PostgreSQL + Milvus. 어느 쪽이든 'DB는 실행당 1개'(반복은 Agent2 질의만 재실행).
  - 결과물은 eval/results/run_<UTC타임스탬프>/ 에 전부 보존.
    sqlite는 run/db/*.db 파일 그대로, postgres는 라이브 유지 + pg_dump 덤프.

지표:
  H1 CER(유형별)  ·  H2 NER Recall(유형별)  ·  H3 질의(Recall@k·MRR·SQL·AnswerHit, K회 평균±std)
  + Agent1 파싱 지표(단일)

사용:
  python run_experiment.py --repeats 3            # 실제 실행(API 비용)
  python run_experiment.py --fake --repeats 1     # 플러밍 점검(API 0, mock LLM)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote, urlunparse


def log(msg: str) -> None:
    print(f"[exp] {msg}", flush=True)


def create_run_db(ts: str):
    """기존 Postgres 서버에 실행 전용 DB를 새로 만들고 CFG.database_url 을 그쪽으로 전환."""
    import psycopg2
    from config import CFG
    if CFG.relational_backend != "postgres":
        raise SystemExit("RELATIONAL_BACKEND=postgres 필요(.env 확인). 현재: " + CFG.relational_backend)
    u = urlparse(CFG.database_url)
    admin = psycopg2.connect(
        host=u.hostname or "localhost", port=u.port or 5432,
        user=unquote(u.username or ""), password=unquote(u.password or ""),
        dbname=(u.path or "/").lstrip("/"), client_encoding="UTF8")
    admin.autocommit = True
    newdb = f"forensic_run_{ts}"
    with admin.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{newdb}"')
    admin.close()
    CFG.database_url = urlunparse(u._replace(path="/" + newdb))
    log(f"created database {newdb} (port {u.port})")
    return newdb, (u.hostname or "localhost"), (u.port or 5432), unquote(u.username or "")


def dump_db(newdb: str, host: str, port: int, user: str, out_path: Path) -> bool:
    """pg_dump 덤프(best-effort). 컨테이너 안에서 실행. 실패해도 라이브 DB는 보존됨."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 1) 호스트 pg_dump  2) docker exec
    attempts = [
        ["pg_dump", "-h", host, "-p", str(port), "-U", user, "-d", newdb],
        ["docker", "exec", "forensic-postgres", "pg_dump", "-U", user, "-d", newdb],
    ]
    for cmd in attempts:
        try:
            with out_path.open("w", encoding="utf-8") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, check=True, timeout=600)
            if out_path.stat().st_size > 0:
                log(f"db dump -> {out_path} ({cmd[0]})")
                return True
        except Exception:
            continue
    log("db dump 실패(라이브 DB는 보존됨). pg_dump/docker 미사용 환경.")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3, help="Agent2 질의 반복 횟수(H3 변동성)")
    ap.add_argument("--docs-dir", default="data/eval_set")
    ap.add_argument("--queries", default="eval/queries.jsonl")
    ap.add_argument("--ner-gold", default="eval/ner_gold.jsonl")
    ap.add_argument("--cer-ref", default="eval/cer_ref")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mode", choices=["toolcall", "graph"], default="toolcall")
    ap.add_argument("--fake", action="store_true", help="mock LLM(오프라인, API 0) — 플러밍 점검용")
    ap.add_argument("--skip-dump", action="store_true")
    ap.add_argument("--results-dir", default="eval/results")
    args = ap.parse_args()

    # 백엔드: .env(CFG) 값을 존중. 기본 sqlite + local(무Docker). postgres+milvus도 옵션 지원.
    if args.fake:
        import mock_llm
        mock_llm.apply_fakes()

    from config import CFG
    use_pg = CFG.relational_backend == "postgres"
    use_milvus = CFG.vector_backend == "milvus"
    log(f"backend: relational={CFG.relational_backend} vector={CFG.vector_backend}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = Path(args.results_dir) / f"run_{ts}"
    for sub in ("", "parsed", "stt", "db", "tables"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    log(f"run dir: {run}")

    # 입력 로드 + 스냅샷(재현용 정답 고정)
    import shutil
    import run_eval as RE
    (run / "inputs").mkdir(exist_ok=True)
    for src in (args.queries, args.ner_gold):
        if Path(src).exists():
            shutil.copyfile(src, run / "inputs" / Path(src).name)
    queries = RE.load_queries(Path(args.queries))
    ner_gold = RE.load_ner_gold(Path(args.ner_gold))
    log(f"queries={len(queries)} ner_gold_files={len(ner_gold)} repeats={args.repeats} fake={args.fake}")

    # 1) 실행 전용 DB(+벡터 네임스페이스) — 백엔드별
    if use_pg:
        newdb, host, port, user = create_run_db(ts)
    else:
        # sqlite: 실행 전용 DB 파일. 로컬 벡터는 같은 경로 + ".vec.json" 자동 사용.
        run_db = (run / "db" / f"forensic_run_{ts}.db").resolve()
        CFG.db_path = str(run_db)
        newdb, host, port, user = run_db.name, "localhost", 0, ""
        log(f"sqlite db: {run_db}")
    if use_milvus:
        CFG.milvus_collection = f"entities_run_{ts}"
        log(f"milvus collection: {CFG.milvus_collection}")

    # 2) 적재(Agent1) — 1회: 파싱·NER·DB·벡터
    from agent1.toolcall import ingest_dir_toolcall
    from agent1.graph import ingest_dir
    runner = ingest_dir_toolcall if args.mode == "toolcall" else ingest_dir
    log("ingesting (Agent1) ... 파싱/STT/NER 1회")
    stats = runner(args.docs_dir, verbose=True)
    log(f"ingest stats: {stats}")

    # 3) DB 스냅샷 → parsed/, stt/, entities.jsonl
    from storage import relational as R
    conn = R.connect()
    _, docs = R.execute_readonly(conn, "SELECT file_name, file_type, raw_text FROM documents")
    parsed_by_stem, sys_text_by_stem, type_by_stem = {}, {}, {}
    for d in docs:
        stem = os.path.splitext(d["file_name"])[0]
        txt = d.get("raw_text") or ""
        (run / "parsed" / f"{stem}.txt").write_text(txt, encoding="utf-8")
        if d.get("file_type") == "audio":
            (run / "stt" / f"{stem}.txt").write_text(txt, encoding="utf-8")
        sys_text_by_stem[stem] = txt
        type_by_stem[stem] = d.get("file_type", "?")
    _, ents = R.execute_readonly(
        conn,
        "SELECT d.file_name AS file_name, e.entity_text AS entity_text, "
        "e.entity_type AS entity_type, e.context AS context "
        "FROM entities e JOIN documents d ON e.doc_id = d.doc_id")
    with (run / "entities.jsonl").open("w", encoding="utf-8") as f:
        for e in ents:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    try:
        conn.close()
    except Exception:
        pass
    log(f"saved parsed={len(docs)} entities={len(ents)}")

    # 4) Agent1 파싱 지표(단일)
    a1 = RE.agent1_metrics(stats)

    # 5) H2 NER Recall(단일)
    ner_metrics, ner_rows = ({}, [])
    if ner_gold:
        ner_metrics, ner_rows = RE.evaluate_ner(ner_gold)
        with (run / "ner_eval.jsonl").open("w", encoding="utf-8") as f:
            for r in ner_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"NER Recall micro={ner_metrics.get('ner_recall_micro')}")

    # 6) H1 CER(단일) — cer_ref(비어있지 않은) vs 시스템 파싱
    cer_metrics, cer_rows = ({}, [])
    cer_dir = Path(args.cer_ref)
    if cer_dir.exists():
        ref_by_stem = {}
        for p in cer_dir.glob("*.txt"):
            t = p.read_text(encoding="utf-8").strip()
            if t:
                ref_by_stem[p.stem] = t
        if ref_by_stem:
            cer_metrics, cer_rows = RE.evaluate_cer(ref_by_stem, sys_text_by_stem, type_by_stem)
            with (run / "cer_per_file.jsonl").open("w", encoding="utf-8") as f:
                for r in cer_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"CER overall={cer_metrics.get('cer_overall')} (N={cer_metrics.get('cer_n_total')})")

    # 7) H3 Agent2 질의 — K회 반복(질의만)
    from storage.vectorstore import get_store
    q_runs = []
    all_rows = []
    for i in range(args.repeats):
        store = get_store()
        qm, qrows = RE.evaluate_queries(queries, store, args.topk)
        q_runs.append(qm)
        for r in qrows:
            r["_repeat"] = i + 1
        all_rows += qrows
        log(f"  [Agent2 r{i+1}] Recall@{args.topk}={qm['retrieval_recall_at_k']:.2f} "
            f"SQL={qm['sql_exec_success']:.2f} Answer={qm['answer_hit']:.2f}")
    with (run / "queries_rows.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Agent2 지표 평균±표준편차
    def agg(key):
        vals = [r[key] for r in q_runs if key in r]
        if not vals:
            return None
        return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "values": [round(v, 4) for v in vals]}
    q_keys = sorted({k for r in q_runs for k in r})
    q_summary = {k: agg(k) for k in q_keys}

    # 8) metrics.json + meta.json
    metrics = {
        "agent1_parse": a1,           # 단일
        "h1_cer": cer_metrics,        # 단일
        "h2_ner_recall": ner_metrics, # 단일
        "h3_query": q_summary,        # 평균±표준편차
    }
    (run / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def pkg_ver(name):
        try:
            import importlib.metadata as m
            return m.version(name)
        except Exception:
            return None
    meta = {
        "timestamp_utc": ts, "fake": args.fake, "mode": args.mode,
        "llm_model": CFG.llm_model, "embed_backend": CFG.embed_backend,
        "embed_model": CFG.embed_model, "upstage_document_model": CFG.upstage_document_model,
        "clova_language": CFG.clova_language,
        "relational_backend": CFG.relational_backend, "database": newdb,
        "vector_backend": CFG.vector_backend, "milvus_collection": CFG.milvus_collection,
        "topk": args.topk, "repeats": args.repeats, "docs_dir": args.docs_dir,
        "n_docs": stats.get("files"), "n_entities": len(ents),
        "queries": len(queries), "ner_gold_files": len(ner_gold),
        "cer_ref_files": len(ref_by_stem) if cer_dir.exists() else 0,
        "packages": {p: pkg_ver(p) for p in ("pymilvus", "psycopg2-binary", "rapidfuzz", "openai", "boto3")},
        "note": "비밀키 미포함. DB는 라이브 유지 + db/ 덤프.",
    }
    (run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 9) 요약 표(markdown)
    lines = [f"# run_{ts} 요약", "",
             f"- model: {CFG.llm_model} / docs: {stats.get('files')} / repeats: {args.repeats} / fake: {args.fake}",
             f"- database: {newdb} / milvus: {CFG.milvus_collection}", ""]
    if cer_metrics:
        lines += ["## H1 CER (낮을수록 좋음)", "| 유형 | N | CER |", "|---|---|---|"]
        for t in ("pdf", "hwp", "audio", "image", "txt"):
            if f"cer_{t}" in cer_metrics:
                lines.append(f"| {t} | {int(cer_metrics[f'cer_n_{t}'])} | {cer_metrics[f'cer_{t}']:.4f} |")
        lines.append(f"| **전체** | {int(cer_metrics['cer_n_total'])} | **{cer_metrics['cer_overall']:.4f}** |")
        lines.append("")
    if ner_metrics:
        lines += ["## H2 NER Recall (유형별)", "| 유형 | Recall |", "|---|---|"]
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

    # 10) DB 덤프(best-effort) — postgres만. sqlite는 run/db/ 에 .db 파일 그대로 보존됨.
    if not args.skip_dump and use_pg:
        dump_db(newdb, host, port, user, run / "db" / f"{newdb}.sql")

    # 11) INDEX.md 갱신(append)
    idx = Path(args.results_dir) / "INDEX.md"
    head = "" if idx.exists() else "# 실험 실행 이력\n\n| run | model | docs | repeats | fake | database | CER | NER micro | Answer |\n|---|---|---|---|---|---|---|---|---|\n"
    row = (f"| run_{ts} | {CFG.llm_model} | {stats.get('files')} | {args.repeats} | {args.fake} | "
           f"{newdb} | {cer_metrics.get('cer_overall','-')} | {ner_metrics.get('ner_recall_micro','-')} | "
           f"{q_summary.get('answer_hit',{}).get('mean','-') if q_summary.get('answer_hit') else '-'} |\n")
    with idx.open("a", encoding="utf-8") as f:
        if head:
            f.write(head)
        f.write(row)

    log(f"DONE → {run}")
    log(f"  metrics.json / tables/summary.md / db live={newdb}")


if __name__ == "__main__":
    main()
