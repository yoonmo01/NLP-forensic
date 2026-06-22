"""압수 C드라이브에서 '사용자 영역의 처리 대상 파일'만 골라 표본을 작업 폴더로 복사.

- 시스템/캐시 디렉터리(Windows, Program Files, AppData 등) 자동 제외
- 확장자 필터(기본 pdf/txt/hwp, --include-doc 시 doc/docx 추가)
- 타입별 균형 표본 N건, 매니페스트(csv) 생성
- 키 불필요(순수 파일시스템). 이후 run_ingest.py --input-dir <dst> 로 적재.

예) python make_sample.py --n 60 --include-doc
    python make_sample.py --dry-run            # 복사 없이 후보 통계만
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import re
import shutil

from agent1.tools import EXCLUDE_DIRS, detect_file_type

DEFAULT_SRC = os.path.join("data", "구매팀_강수민(대리)")


def safe_name(idx: int, ftype: str, orig: str) -> str:
    base = re.sub(r"[^0-9a-zA-Z가-힣._-]", "_", os.path.basename(orig))
    return f"{idx:04d}_{ftype}_{base}"


def collect(src: str, exts: set[str], max_bytes: int, user_only: bool) -> dict[str, list[str]]:
    by_type: dict[str, list[str]] = {}
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        low = root.replace("\\", "/").lower()
        if any(p in EXCLUDE_DIRS for p in low.split("/")):
            continue
        if user_only and "/users/" not in low:
            continue
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            path = os.path.join(root, fn)
            low_fn = fn.lower()
            # 비증거 잡파일 제외 (라이선스/리드미/체인지로그 등)
            if any(j in low_fn for j in ("license", "licence", "readme", "copying",
                                         "changelog", "notice", "라이선스", "사용권")):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > max_bytes:
                continue
            if ext in (".txt", ".md") and size < 300:   # 빈/잡 메모(예: 0.1KB) 제외
                continue
            by_type.setdefault(detect_file_type(path), []).append(path)
    return by_type


def balanced_sample(by_type: dict[str, list[str]], n: int, seed: int) -> list[str]:
    random.seed(seed)
    types = [t for t in by_type if by_type[t]]
    if not types:
        return []
    selected: list[str] = []
    quota = max(1, n // len(types))
    leftover: list[str] = []
    for t in types:
        pool = by_type[t][:]
        random.shuffle(pool)
        selected += pool[:quota]
        leftover += pool[quota:]
    random.shuffle(leftover)
    selected += leftover[: max(0, n - len(selected))]
    return selected[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=os.path.join("data", "work_sample"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--types", default="pdf,txt,hwp")
    ap.add_argument("--include-doc", action="store_true", help="doc/docx 포함")
    ap.add_argument("--max-mb", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-areas", action="store_true", help="Users 외 영역도 포함(비권장)")
    ap.add_argument("--dry-run", action="store_true", help="복사 없이 통계만")
    args = ap.parse_args()

    exts = {"." + e.strip().lstrip(".") for e in args.types.split(",") if e.strip()}
    if args.include_doc:
        exts |= {".doc", ".docx"}
    max_bytes = int(args.max_mb * 1024 * 1024)

    if not os.path.isdir(args.src):
        print(f"[오류] 소스 폴더 없음: {args.src}")
        return

    print(f"[수집] src={args.src} exts={sorted(exts)} user_only={not args.all_areas}")
    by_type = collect(args.src, exts, max_bytes, user_only=not args.all_areas)
    total = sum(len(v) for v in by_type.values())
    print(f"[후보] 총 {total}건")
    for t, v in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"   {t}: {len(v)}")

    picks = balanced_sample(by_type, args.n, args.seed)
    print(f"\n[표본] {len(picks)}건 선택 (요청 {args.n})")
    picked_types: dict[str, int] = {}
    for p in picks:
        picked_types[detect_file_type(p)] = picked_types.get(detect_file_type(p), 0) + 1
    print("   유형별:", picked_types)

    if args.dry_run:
        print("\n[dry-run] 복사 생략.")
        return

    os.makedirs(args.dst, exist_ok=True)
    manifest = os.path.join(args.dst, "manifest.csv")
    rows = []
    for i, src_path in enumerate(picks, 1):
        ftype = detect_file_type(src_path)
        newname = safe_name(i, ftype, src_path)
        dst_path = os.path.join(args.dst, newname)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"   복사 실패: {src_path} ({e})")
            continue
        rel = os.path.relpath(src_path, args.src)
        rows.append({
            "new_name": newname, "type": ftype,
            "ext": os.path.splitext(src_path)[1].lower(),
            "size_kb": round(os.path.getsize(dst_path) / 1024, 1),
            "orig_relpath": rel,
        })
    with open(manifest, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["new_name", "type", "ext", "size_kb", "orig_relpath"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[완료] {len(rows)}건 → {args.dst}")
    print(f"[매니페스트] {manifest}")
    print(f"\n다음: python run_ingest.py --input-dir {args.dst}   (.env 키 필요)")


if __name__ == "__main__":
    main()
