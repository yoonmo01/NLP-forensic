"""실제 표본 파일의 '파싱 단계'만 오프라인 검증 (NER/임베딩 없음 → API 키 불필요).

각 파일에 대해 detect → 전략별 파싱 시도(fallback 포함) → 품질 평가 결과를 표로 출력.
실제 압수 데이터에서 파싱 라우팅·실패복구가 어떻게 동작하는지 키 없이 확인할 수 있다.

사용: python test_parse.py --dir data/work_sample
"""
from __future__ import annotations
import argparse
import os

from config import CFG
from agent1 import tools as T


def parse_file(path: str):
    ftype = T.detect_file_type(path)
    strategies = T.STRATEGIES.get(ftype, ["txt"])
    log = []
    best = ("", 0.0)
    for strat in strategies:
        try:
            text = T.PARSERS[strat](path)
            q = T.assess_extraction_quality(text)
            log.append(f"{strat}=OK(len={len(text)},q={q})")
            if q > best[1]:
                best = (text, q)
            if q >= CFG.parse_quality_threshold:
                break  # 품질 충족 → fallback 중단
        except Exception as e:
            log.append(f"{strat}=실패({type(e).__name__})")
    return ftype, best[1], len(best[0]), " → ".join(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("data", "work_sample"))
    args = ap.parse_args()

    files = [os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir))
             if not f.lower().endswith(".csv")]
    print(f"[파싱 검증] {len(files)}개 파일  (임계값 q>={CFG.parse_quality_threshold})\n")
    print(f"{'file':<34}{'type':<7}{'q':<7}{'chars':<8}경로/로그")
    ok = low = fail = 0
    by_type = {}
    for path in files:
        ftype, q, n, log = parse_file(path)
        status = "OK" if q >= CFG.parse_quality_threshold else ("LOW" if n > 0 else "FAIL")
        ok += status == "OK"; low += status == "LOW"; fail += status == "FAIL"
        by_type.setdefault(ftype, {"ok": 0, "low": 0, "fail": 0})
        by_type[ftype][status.lower()] += 1
        name = os.path.basename(path)
        print(f"{name[:32]:<34}{ftype:<7}{q:<7}{n:<8}{log}")

    print(f"\n=== 요약 === OK={ok}  LOW={low}  FAIL={fail}  (총 {len(files)})")
    for t, c in sorted(by_type.items()):
        print(f"   {t}: OK {c['ok']} / LOW {c['low']} / FAIL {c['fail']}")
    print("\n참고: soffice(LibreOffice) 미설치 시 .doc 및 HWP libreoffice fallback은 실패로 표기됨.")


if __name__ == "__main__":
    main()
