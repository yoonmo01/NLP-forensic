"""단순 추출 기준선 CER — pdfplumber/txt 전용(HWP·audio·image 불가=1.0)."""
import os, sys, re
from pathlib import Path

sys.path.insert(0, ".")

_SYM  = re.compile(r"[|#*`_~\->\[\](){}<>/\\]+")
_PUNC = re.compile(r"[.,!?;:'\".·…“”‘’\-—–]+")
_FILL = re.compile(r"^[어음으]+$")


def _norm(s):
    if not s:
        return ""
    t = _SYM.sub(" ", str(s))
    t = _PUNC.sub(" ", t)
    toks = [w for w in t.split() if not _FILL.match(w)]
    return "".join(toks).lower()


def _cer(ref, sys_):
    rn = _norm(ref)
    sn = _norm(sys_)
    if not rn:
        return None
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(rn, sn) / len(rn)


cer_ref = Path("eval/cer_ref")
data_dir = Path("data/eval_set")

results = []
for ref_path in sorted(cer_ref.glob("*.txt")):
    ref_text = ref_path.read_text(encoding="utf-8").strip()
    if not ref_text:
        continue
    stem = ref_path.stem

    orig = None
    for ext in [".pdf", ".hwp", ".txt", ".mp3", ".wav", ".m4a", ".png", ".jpg"]:
        for f in data_dir.rglob(stem + ext):
            orig = f
            break
        if orig:
            break

    if not orig:
        print(f"NOT FOUND: {stem}")
        continue

    ftype = orig.suffix.lower().lstrip(".")
    baseline_text = ""
    try:
        if ftype == "pdf":
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(str(orig))
            parts = []
            for pg in doc:
                parts.append(pg.get_textpage().get_text_bounded() or "")
            baseline_text = "\n".join(parts)
        elif ftype == "txt":
            baseline_text = orig.read_text(encoding="utf-8", errors="replace")
        # hwp / audio / image: cannot extract without special tools → empty → CER=1.0
    except Exception as e:
        baseline_text = ""

    c = _cer(ref_text, baseline_text)
    if c is None:
        print(f"{ftype:6} {stem[:45]:45} ref_empty → skip")
        continue
    results.append({"stem": stem, "type": ftype, "baseline_cer": round(c, 4)})
    print(f"{ftype:6} {stem[:45]:45} baseline_cer={c:.4f}")

print()
by_type = {}
for r in results:
    by_type.setdefault(r["type"], []).append(r["baseline_cer"])
for t, vals in sorted(by_type.items()):
    print(f"  {t}: n={len(vals)}  mean={sum(vals)/len(vals):.4f}  vals={vals}")
