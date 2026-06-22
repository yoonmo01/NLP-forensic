"""저장된 실행 결과물만으로 metrics 를 다시 계산해 보고 수치와 일치하는지 검증한다.
API 호출 0회. '나중에 확인 가능'의 실질 보장.

사용:
  python recompute.py eval/results/run_<ts>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("사용: python recompute.py eval/results/run_<ts>")
    run = Path(sys.argv[1])
    import run_eval as RE

    reported = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    print(f"== recompute {run.name} ==")

    ok = True

    # --- H1 CER: parsed/ vs cer_ref(스냅샷 우선, 없으면 라이브) ---
    parsed = {p.stem: p.read_text(encoding="utf-8") for p in (run / "parsed").glob("*.txt")}
    # 파일유형: 파일명 둘째 토큰(0013_pdf_... → pdf)
    type_by_stem = {s: (s.split("_")[1] if len(s.split("_")) > 1 else "?") for s in parsed}
    cer_dir = run / "inputs" / "cer_ref"
    if not cer_dir.exists():
        cer_dir = Path("eval/cer_ref")
    ref = {}
    if cer_dir.exists():
        for p in cer_dir.glob("*.txt"):
            t = p.read_text(encoding="utf-8").strip()
            if t:
                ref[p.stem] = t
    if ref:
        cer_m, _ = RE.evaluate_cer(ref, parsed, type_by_stem)
        rep = reported.get("h1_cer", {}).get("cer_overall")
        rec = cer_m.get("cer_overall")
        match = (rep is None) or (abs((rep or 0) - (rec or 0)) < 1e-6)
        ok &= match
        print(f"H1 CER overall: reported={rep} recomputed={rec}  {'OK' if match else 'MISMATCH'}")

    # --- H2 NER Recall: entities.jsonl + ner_gold(스냅샷) ---
    ents = load_jsonl(run / "entities.jsonl")
    # file_name 이 entities.jsonl 에 없으면 doc_id→file_name 매핑 필요. 여기선 ner_gold 의 file_name 키로 비교 위해
    # entities 에 file_name 이 있다고 가정(없으면 스킵).
    ner_gold_path = run / "inputs" / "ner_gold.jsonl"
    if not ner_gold_path.exists():
        ner_gold_path = Path("eval/ner_gold.jsonl")
    gold = RE.load_ner_gold(ner_gold_path)
    if ents and gold and all("file_name" in e for e in ents[:1]):
        # 시스템 추출(정규화) by file,type
        sysf = {}
        for e in ents:
            txt = RE._norm(e.get("entity_text"))
            if txt:
                sysf.setdefault(e["file_name"], {}).setdefault(e.get("entity_type"), set()).add(txt)
        hit = {t: 0 for t in RE.NER_TYPES}
        tot = {t: 0 for t in RE.NER_TYPES}
        for fn, gtypes in gold.items():
            sf = sysf.get(fn, {})
            for t in RE.NER_TYPES:
                gold_norm = {RE._norm(x) for x in gtypes.get(t, set())} - {""}
                tot[t] += len(gold_norm)
                hit[t] += len(gold_norm & sf.get(t, set()))
        micro = sum(hit.values()) / sum(tot.values()) if sum(tot.values()) else 0.0
        rep = reported.get("h2_ner_recall", {}).get("ner_recall_micro")
        match = (rep is None) or (abs((rep or 0) - micro) < 1e-6)
        ok &= match
        print(f"H2 NER micro:   reported={rep} recomputed={round(micro,4)}  {'OK' if match else 'MISMATCH'}")
    else:
        print("H2 NER: entities.jsonl 에 file_name 없음 → 오프라인 재계산 스킵(라이브 DB로 검증).")

    # --- H3 질의: queries_rows.jsonl 에서 집계 ---
    rows = load_jsonl(run / "queries_rows.jsonl")
    if rows:
        reps = sorted({r.get("_repeat", 1) for r in rows})
        per = []
        for rp in reps:
            rr = [r for r in rows if r.get("_repeat", 1) == rp]
            ans = sum(1 for r in rr if r.get("answer_hit")) / len(rr) if rr else 0.0
            per.append(ans)
        mean = sum(per) / len(per)
        rep = reported.get("h3_query", {}).get("answer_hit", {})
        rep_mean = rep.get("mean") if isinstance(rep, dict) else None
        match = (rep_mean is None) or (abs(rep_mean - mean) < 1e-6)
        ok &= match
        print(f"H3 answer_hit:  reported={rep_mean} recomputed={round(mean,4)} (repeats={len(per)})  {'OK' if match else 'MISMATCH'}")

    print("== 결과:", "전부 일치 [OK]" if ok else "불일치 항목 있음 [FAIL]", "==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
