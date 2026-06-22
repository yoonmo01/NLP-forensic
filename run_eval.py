"""평가 하네스 (C 보완판).

두 단계로 분리하여 평가한다.
  Phase A — Agent1 폴드 평가: 평가 문서 집합 5개(set_01~05)를 각각 적재하여
            문서 단위 지표(파싱 성공률, 고품질률, 실패율, 파일당 개체수)를 폴드별 평균±표준편차로 보고.
            (파싱/추출은 문서별 독립이므로 분할이 안전하다.)
  Phase B — Agent2 전체 DB 평가: 전체 평가 문서(기본 data/eval_set, 50건)를 '하나의 DB'에 적재하고
            질의셋을 실행한다. 관계형 질의가 깨지지 않도록 절대 분할하지 않는다.
            변동성은 전체 파이프라인을 --repeats 회 재실행하여 평균±표준편차로 보고.

사용:
  python make_eval_sets.py
  python run_eval.py --fake
  python run_eval.py --mode toolcall --repeats 3
  python run_eval.py --input-dir data/sample --fake   # 단일 디렉터리 1회 평가
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

AGENT1_KEYS = [
    "agent1_parse_success", "agent1_high_quality",
    "agent1_failed_rate", "agent1_entities_per_file",
]
AGENT2_KEYS = [
    "classification_accuracy", "retrieval_recall_at_k",
    "retrieval_mrr", "sql_exec_success", "answer_hit",
]
# 가설2: NER 유형별 Recall (인물·날짜·금액·계좌·장소)
NER_TYPES = ["인물", "날짜", "금액", "계좌", "장소"]
NER_KEYS = [f"ner_recall_{t}" for t in NER_TYPES] + ["ner_recall_micro", "ner_recall_macro"]
LABELS = {
    "agent1_parse_success": "Agent1 Parse Success",
    "agent1_high_quality": "Agent1 High Quality",
    "agent1_failed_rate": "Agent1 Failed Rate",
    "agent1_entities_per_file": "Agent1 Entities/File",
    "classification_accuracy": "Classification Accuracy",
    "retrieval_recall_at_k": "Retrieval Recall@k",
    "retrieval_mrr": "Retrieval MRR",
    "sql_exec_success": "SQL Exec Success",
    "answer_hit": "Answer Hit",
    "ner_recall_micro": "NER Recall (micro)",
    "ner_recall_macro": "NER Recall (macro)",
    **{f"ner_recall_{t}": f"NER Recall [{t}]" for t in NER_TYPES},
}


def load_queries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def configure_isolated_run(run_name: str, backend: str, topk: int) -> str:
    """반복 평가가 운영 DB를 오염시키지 않도록 임시 SQLite + 격리 벡터스토어를 사용."""
    tmp = tempfile.mkdtemp(prefix=f"{run_name}_")
    db_path = os.path.join(tmp, "eval.db")
    os.environ["DB_PATH"] = db_path
    os.environ["VECTOR_BACKEND"] = backend
    os.environ["TOPK"] = str(topk)
    from config import CFG
    CFG.db_path = db_path
    CFG.vector_backend = backend
    CFG.topk = topk
    return db_path


def agent1_metrics(stats: dict) -> dict[str, float]:
    files = stats.get("files", 0)
    indexed = (stats.get("ok", 0) + stats.get("low_quality", 0)
               + stats.get("captioned", 0) + stats.get("no_text", 0))
    return {
        "agent1_parse_success": ratio(stats.get("ok", 0) + stats.get("low_quality", 0), files),
        "agent1_high_quality": ratio(stats.get("ok", 0), files),
        "agent1_failed_rate": ratio(stats.get("failed", 0), files),
        "agent1_entities_per_file": ratio(stats.get("entities", 0), files),
        "agent1_indexed_rate": ratio(indexed, files),     # 전체 파일 색인율(캡션·마커 포함)
        "agent1_captioned": float(stats.get("captioned", 0)),
        "agent1_no_text": float(stats.get("no_text", 0)),
        "agent1_duplicate": float(stats.get("duplicate", 0)),
    }


def _ingest(input_dir: Path, args, verbose: bool) -> dict:
    from agent1.graph import ingest_dir
    from agent1.toolcall import ingest_dir_toolcall
    runner = ingest_dir_toolcall if args.mode == "toolcall" else ingest_dir
    return runner(str(input_dir), verbose=verbose)


import re as _re
# 한자/한글 날짜 구분자 통일 (年월日/년월일 → '-')
_DATE_TRANS = str.maketrans({"年": "-", "月": "-", "日": "", "년": "-", "월": "-", "일": ""})


def _norm(s: object) -> str:
    """엔티티 매칭용 정규화: 공백·콤마·통화기호 제거, 날짜 표기 통일.
    gold·시스템 출력 양쪽에 동일 적용하므로 표면형 차이로 인한 false negative를 막는다."""
    if not s:
        return ""
    t = str(s).translate(_DATE_TRANS)
    t = _re.sub(r"\s+", "", t)
    t = t.replace(",", "").replace("₩", "")
    t = _re.sub(r"원$", "", t)          # 말미 통화단위만 제거(원단 등 어휘 보존)
    t = _re.sub(r"[./]", "-", t)
    t = _re.sub(r"-+", "-", t).strip("-")
    return t.lower()


_NER_NOISE = _re.compile(r"[\[\](){}<>|#*`_~\-–—=]+")
_NER_ROLE_WORDS = (
    "대표이사", "부사장", "팀장님", "과장님", "대리님", "사장님",
    "대표", "팀장", "과장", "대리", "사원", "차장", "부장", "이사", "님",
)
_NER_DEPT_WORDS = (
    "구매팀", "생산팀", "디자인팀", "인사팀", "총무팀", "영업팀", "IT팀",
    "운송관리부", "경영기획부", "부서장",
)
_WEEKDAYS = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")


def _compact_entity_text(s: object) -> str:
    if not s:
        return ""
    t = str(s)
    t = _NER_NOISE.sub(" ", t)
    for w in _WEEKDAYS:
        t = t.replace(w, " ")
    for w in _NER_DEPT_WORDS + _NER_ROLE_WORDS:
        t = t.replace(w, " ")
    t = t.replace(",", " ").replace("·", " ")
    t = _re.sub(r"\s+", "", t)
    return t.lower()


def _date_key(s: object) -> str:
    if not s:
        return ""
    t = str(s)
    m = _re.search(r"(\d{4})\s*[년./-]\s*(\d{1,2})\s*[월./-]\s*(\d{1,2})\s*일?", t)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = _re.search(r"(\d{4})(\d{2})(\d{2})", t)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return ""


def _amount_key(s: object) -> str:
    if not s:
        return ""
    t = str(s).lower()
    t = t.replace(",", "").replace(" ", "")
    t = t.replace("원", "").replace("₩", "")
    t = t.replace("불", "usd").replace("달러", "usd")
    return t


def _account_key(s: object) -> str:
    return _re.sub(r"\D+", "", str(s or ""))


def _ner_match(entity_type: str, gold: object, system: object) -> bool:
    """Recall-oriented matching for canonical gold vs extracted surface forms."""
    if not gold or not system:
        return False
    if entity_type == "날짜":
        g = _date_key(gold)
        s = _date_key(system)
        return bool(g and s and g == s)
    if entity_type == "금액":
        g = _amount_key(gold)
        s = _amount_key(system)
        return bool(g and (g in s or s in g))
    if entity_type == "계좌":
        g = _account_key(gold)
        s = _account_key(system)
        return bool(g and (g in s or s in g))
    g = _compact_entity_text(gold)
    s = _compact_entity_text(system)
    return bool(g and (g in s or s in g))


def evaluate_queries(queries: list[dict], store, topk: int) -> tuple[dict[str, float], list[dict]]:
    from agent2.graph import answer
    from agent2 import tools as T

    n = len(queries)
    cls_ok = sql_ok = ans_ok = 0
    recall_sum = mrr_sum = 0.0
    rows_out: list[dict[str, Any]] = []
    tier_tot: dict[str, int] = {}
    tier_hit: dict[str, int] = {}
    for q in queries:
        gold_e = set(q.get("gold_entities", []))
        gold_f = set(q.get("gold_files", []))
        gold_en = {_norm(x) for x in gold_e} - {""}   # 정규화된 정답 엔티티

        ctype = T.classify_query(q["query"]).get("type")
        cls = ctype == q.get("qtype")
        cls_ok += int(cls)

        hits = T.resolve_entity(q["query"], store, top_k=topk)
        hit_texts = [h["entity_text"] for h in hits]
        found = [i for i, text in enumerate(hit_texts) if _norm(text) in gold_en]
        recall = ratio(len({_norm(t) for t in hit_texts} & gold_en), len(gold_en)) if gold_en else 0.0
        mrr = (1.0 / (found[0] + 1)) if found else 0.0
        recall_sum += recall
        mrr_sum += mrr

        out = answer(q["query"])
        rows = out.get("rows", [])
        sqlok = bool(rows) and not out.get("error")
        sql_ok += int(sqlok)
        got_e = {_norm(r.get("entity_text")) for r in rows}
        got_f = {r.get("file_name") for r in rows}
        if q.get("expect_empty"):
            ans = (len(rows) == 0)          # 부정 질의: 빈 결과가 정답
        else:
            ans = bool((got_e & gold_en) or (got_f & gold_f))
        ans_ok += int(ans)
        tier = q.get("tier", "T?")
        tier_tot[tier] = tier_tot.get(tier, 0) + 1
        tier_hit[tier] = tier_hit.get(tier, 0) + int(ans)
        rows_out.append({
            "query": q["query"], "tier": tier, "expected_type": q.get("qtype"),
            "predicted_type": ctype, "classification_ok": cls,
            "retrieval_recall_at_k": round(recall, 4), "retrieval_mrr": round(mrr, 4),
            "sql_exec_success": sqlok, "answer_hit": ans,
        })
    metrics = {
        "classification_accuracy": ratio(cls_ok, n),
        "retrieval_recall_at_k": ratio(recall_sum, n),
        "retrieval_mrr": ratio(mrr_sum, n),
        "sql_exec_success": ratio(sql_ok, n),
        "answer_hit": ratio(ans_ok, n),
    }
    for t in sorted(tier_tot):
        metrics[f"answer_hit_{t}"] = ratio(tier_hit[t], tier_tot[t])
    return metrics, rows_out


# ---------------- 가설2: NER 유형별 Recall ----------------
def load_ner_gold(path: Path) -> dict[str, dict[str, set]]:
    """파일별 수동 NER 정답. 줄 형식:
    {"file_name": "...", "entities": {"인물": [...], "날짜": [...], ...}}"""
    gold: dict[str, dict[str, set]] = {}
    if not path.exists():
        return gold
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            ents = o.get("entities", {})
            gold[o["file_name"]] = {t: set(ents.get(t, [])) for t in NER_TYPES}
    return gold


def evaluate_ner(gold_by_file: dict[str, dict[str, set]]) -> tuple[dict[str, float], list[dict]]:
    """현재 적재된 관계형 DB의 추출 개체를 정답과 비교해 유형별 Recall 산출.
    Recall = |정답 ∩ 시스템추출| / |정답| (정규화 후, 정답 보유 파일에 한해 집계)."""
    from storage import relational as R
    conn = R.connect()
    try:
        _, rows = R.execute_readonly(
            conn,
            "SELECT d.file_name AS file_name, e.entity_text AS entity_text, "
            "e.entity_type AS entity_type FROM entities e "
            "JOIN documents d ON e.doc_id = d.doc_id",
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass

    sys_by_file: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        txt = str(r.get("entity_text") or "").strip()
        if not txt:
            continue
        sys_by_file.setdefault(r["file_name"], {}).setdefault(r.get("entity_type"), []).append(txt)

    hit = {t: 0 for t in NER_TYPES}
    tot = {t: 0 for t in NER_TYPES}
    per_file: list[dict] = []
    for fn, gtypes in gold_by_file.items():
        sysf = sys_by_file.get(fn, {})
        row = {"file_name": fn}
        for t in NER_TYPES:
            gold_vals = [x for x in gtypes.get(t, set()) if str(x).strip()]
            sys_vals = sysf.get(t, [])
            matched = sum(1 for g in gold_vals if any(_ner_match(t, g, s) for s in sys_vals))
            tot[t] += len(gold_vals)
            hit[t] += matched
            if gold_vals:
                row[t] = f"{matched}/{len(gold_vals)}"
        per_file.append(row)

    metrics: dict[str, float] = {}
    for t in NER_TYPES:
        metrics[f"ner_recall_{t}"] = ratio(hit[t], tot[t])
    metrics["ner_recall_micro"] = ratio(sum(hit.values()), sum(tot.values()))
    present = [t for t in NER_TYPES if tot[t] > 0]
    metrics["ner_recall_macro"] = (
        ratio(sum(metrics[f"ner_recall_{t}"] for t in present), len(present)) if present else 0.0
    )
    return metrics, per_file


# ---------------- 가설1: CER (문자 오류율) ----------------
# 정규화: 표기호/마크다운·문장부호 제거, 간투어(어/음/으) 제거, 공백 전부 제거 후 char 비교.
_CER_SYMBOL = _re.compile(r"[|#*`_~\->\[\](){}<>/\\]+")
_CER_PUNCT = _re.compile(r"[.,!?;:'\"·…“”‘’，。！？、　〈〉《》「」『』\-—–]+")
_CER_FILLER = _re.compile(r"^[어음으]+$")


def _cer_norm(s: str) -> str:
    if not s:
        return ""
    t = _CER_SYMBOL.sub(" ", str(s))
    t = _CER_PUNCT.sub(" ", t)
    toks = [w for w in t.split() if not _CER_FILLER.match(w)]   # 간투어 토큰 제거
    return "".join(toks).lower()


def _edit_distance(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except Exception:
        # 순수 파이썬 폴백(짧은 문자열용). 긴 문서는 rapidfuzz 권장.
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def evaluate_cer(ref_by_stem: dict[str, str], sys_by_stem: dict[str, str],
                 type_of: dict[str, str]) -> tuple[dict[str, float], list[dict]]:
    """파일별 CER 계산 후 유형별 평균. ref가 빈 파일은 제외.
    ref_by_stem/sys_by_stem: {파일stem: 텍스트}. type_of: {stem: 파일유형}."""
    per_file: list[dict] = []
    by_type: dict[str, list[float]] = {}
    for stem, ref in ref_by_stem.items():
        rn = _cer_norm(ref)
        if not rn:
            continue                       # 빈 정답(사진 등) 제외
        sysn = _cer_norm(sys_by_stem.get(stem, ""))
        dist = _edit_distance(rn, sysn)
        cer = dist / len(rn)
        ftype = type_of.get(stem, "?")
        by_type.setdefault(ftype, []).append(cer)
        per_file.append({"file": stem, "type": ftype, "ref_len": len(rn),
                         "sys_len": len(sysn), "cer": round(cer, 4)})
    metrics: dict[str, float] = {}
    all_cer: list[float] = []
    for t, vals in by_type.items():
        metrics[f"cer_{t}"] = ratio(sum(vals), len(vals))
        metrics[f"cer_n_{t}"] = float(len(vals))
        all_cer += vals
    metrics["cer_overall"] = ratio(sum(all_cer), len(all_cer))
    metrics["cer_n_total"] = float(len(all_cer))
    return metrics, per_file


# ---------------- Phase A: Agent1 폴드 평가 ----------------
def run_agent1_folds(set_dirs: list[tuple[str, Path]], args) -> list[dict]:
    runs = []
    for name, d in set_dirs:
        configure_isolated_run(name, args.backend, args.topk)
        stats = _ingest(d, args, verbose=not args.quiet)
        runs.append({"run": name, "input_dir": str(d), "ingest": stats,
                     "metrics": agent1_metrics(stats)})
        print(f"  [A:{name}] files={stats.get('files')} "
              f"parse_success={agent1_metrics(stats)['agent1_parse_success']:.2f} "
              f"entities/file={agent1_metrics(stats)['agent1_entities_per_file']:.2f}")
    return runs


# ---------------- Phase B: Agent2 전체 DB 평가 (K회 재실행) ----------------
def run_agent2_full(full_dir: Path, queries: list[dict], args,
                    ner_gold: dict[str, dict[str, set]] | None = None) -> list[dict]:
    from storage.vectorstore import get_store
    runs = []
    for i in range(args.repeats):
        name = f"full_run{i + 1}"
        configure_isolated_run(name, args.backend, args.topk)
        stats = _ingest(full_dir, args, verbose=False)
        store = get_store()
        qm, qrows = evaluate_queries(queries, store, args.topk)
        nrows = None
        if ner_gold:
            nm, nrows = evaluate_ner(ner_gold)
            qm = {**qm, **nm}
        runs.append({"run": name, "input_dir": str(full_dir), "ingest": stats,
                     "metrics": qm, "queries": qrows, "ner": nrows})
        ner_msg = f" NER(micro)={qm['ner_recall_micro']:.2f}" if ner_gold else ""
        print(f"  [B:{name}] docs={stats.get('files')} "
              f"Recall@{args.topk}={qm['retrieval_recall_at_k']:.2f} "
              f"SQL={qm['sql_exec_success']:.2f} Answer={qm['answer_hit']:.2f}{ner_msg}")
    return runs


def summarize(runs: list[dict], keys: list[str]) -> dict:
    out = {}
    for key in keys:
        vals = [r["metrics"][key] for r in runs if key in r.get("metrics", {})]
        if not vals:
            continue
        out[key] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "values": [round(v, 4) for v in vals],
        }
    return out


def print_summary(title: str, summary: dict, keys: list[str]) -> None:
    print(f"\n=== {title} (평균±표준편차) ===")
    print(f"{'metric':<26}{'mean':>8}{'std':>8}  values")
    for key in keys:
        if key not in summary:
            continue
        s = summary[key]
        vals = ", ".join(f"{v:.2f}" for v in s["values"])
        print(f"{LABELS[key]:<26}{s['mean']:>8.3f}{s['std']:>8.3f}  [{vals}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="가짜 LLM(오프라인)")
    ap.add_argument("--backend", default="local", choices=["local", "milvus"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mode", choices=["graph", "toolcall"], default="toolcall")
    ap.add_argument("--sets-dir", default="eval/sets", help="Phase A: Agent1 폴드 디렉터리")
    ap.add_argument("--full-dir", default="data/eval_set", help="Phase B: 전체 DB로 적재할 디렉터리")
    ap.add_argument("--repeats", type=int, default=3, help="Phase B 전체 파이프라인 재실행 횟수")
    ap.add_argument("--input-dir", default=None, help="단일 디렉터리만 1회 평가")
    ap.add_argument("--queries", default="eval/queries.jsonl")
    ap.add_argument("--ner-gold", default="eval/ner_gold.jsonl", help="가설2: 파일별 수동 NER 정답")
    ap.add_argument("--out", default="eval/results/repeated_eval.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.environ["RELATIONAL_BACKEND"] = "sqlite"
    os.environ["VECTOR_BACKEND"] = args.backend
    os.environ.setdefault("OPENAI_API_KEY", "test-dummy")
    if args.fake:
        import mock_llm
        mock_llm.apply_fakes()

    queries = load_queries(Path(args.queries))
    ner_gold = load_ner_gold(Path(args.ner_gold))
    print(f"[setup] mode={args.mode} fake={args.fake} backend={args.backend} "
          f"queries={len(queries)} ner_gold_files={len(ner_gold)} repeats={args.repeats}")

    payload: dict[str, Any] = {"mode": args.mode, "fake": args.fake, "backend": args.backend,
                               "topk": args.topk, "queries": args.queries}

    # 단일 디렉터리 모드: Phase A/B 합쳐 1회만
    if args.input_dir:
        d = Path(args.input_dir)
        configure_isolated_run("single", args.backend, args.topk)
        stats = _ingest(d, args, verbose=not args.quiet)
        from storage.vectorstore import get_store
        qm, qrows = evaluate_queries(queries, get_store(), args.topk)
        nrows = None
        if ner_gold:
            nm, nrows = evaluate_ner(ner_gold)
            qm = {**qm, **nm}
        run = {"run": "single", "input_dir": str(d), "ingest": stats,
               "metrics": {**agent1_metrics(stats), **qm}, "queries": qrows, "ner": nrows}
        payload["single"] = run
        keys = AGENT1_KEYS + AGENT2_KEYS + (NER_KEYS if ner_gold else [])
        print_summary("단일 평가", {k: {"mean": run["metrics"][k], "std": 0.0,
                                       "values": [run["metrics"][k]]} for k in keys if k in run["metrics"]},
                      keys)
    else:
        # Phase A — Agent1 폴드
        set_dirs = [(p.name, p) for p in sorted(Path(args.sets_dir).glob("set_*")) if p.is_dir()]
        if not set_dirs:
            raise SystemExit(f"평가 문서 집합 없음: {args.sets_dir} (make_eval_sets.py 먼저 실행)")
        print(f"\n[Phase A] Agent1 폴드 평가 — {len(set_dirs)}개 세트")
        a_runs = run_agent1_folds(set_dirs, args)
        a_summary = summarize(a_runs, AGENT1_KEYS)
        print_summary("Phase A: Agent1 (폴드 간)", a_summary, AGENT1_KEYS)

        # Phase B — Agent2 전체 DB, K회 재실행 (+ 가설2 NER Recall)
        print(f"\n[Phase B] Agent2 전체 DB 평가 — {args.full_dir}, {args.repeats}회 재실행")
        b_runs = run_agent2_full(Path(args.full_dir), queries, args, ner_gold=ner_gold)
        b_keys = AGENT2_KEYS + (NER_KEYS if ner_gold else [])
        b_summary = summarize(b_runs, b_keys)
        print_summary("Phase B: Agent2 (재실행 간)", b_summary, b_keys)

        payload.update({
            "phase_a_agent1": {"runs": a_runs, "summary": a_summary},
            "phase_b_agent2": {"runs": b_runs, "summary": b_summary},
            "agent2_oracle_eval": {"status": "not_configured",
                                   "reason": "정답 문맥 기반 Agent2 단독(상한) 평가는 oracle JSONL 작성 후 추가."},
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
