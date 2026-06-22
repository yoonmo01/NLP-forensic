"""파일 유형 균형을 유지한 평가 문서 집합을 생성한다.

기본 입력은 data/eval_set/manifest.csv 이며, 유형별 문서를 섞은 뒤 5개 Set에
라운드로빈으로 배정한다. 각 Set 디렉터리에는 원본 파일의 하드링크를 만들고,
하드링크가 실패하면 복사로 대체한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({str(k).strip(): v for k, v in row.items()})
        return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def materialize(src: Path, dst: Path) -> str:
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "exists"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build_sets(rows: list[dict], runs: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    sets: list[list[dict]] = [[] for _ in range(runs)]
    for _typ, items in sorted(by_type.items()):
        items = sorted(items, key=lambda r: r["new_name"])
        rng.shuffle(items)
        for idx, row in enumerate(items):
            sets[idx % runs].append(row)

    for rows_for_set in sets:
        rows_for_set.sort(key=lambda r: (r["type"], r["new_name"]))
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="data/eval_set")
    ap.add_argument("--manifest", default="data/eval_set/manifest.csv")
    ap.add_argument("--out-dir", default="eval/sets")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    rows = read_manifest(manifest_path)
    sets = build_sets(rows, args.runs, args.seed)

    summary = {
        "source_dir": str(source_dir),
        "manifest": str(manifest_path),
        "runs": args.runs,
        "seed": args.seed,
        "sets": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, set_rows in enumerate(sets, start=1):
        set_name = f"set_{i:02d}"
        set_dir = out_dir / set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = defaultdict(int)
        link_modes: dict[str, int] = defaultdict(int)
        materialized_rows = []
        for row in set_rows:
            src = source_dir / row["new_name"]
            dst = set_dir / row["new_name"]
            if not src.exists():
                raise FileNotFoundError(src)
            mode = materialize(src, dst)
            link_modes[mode] += 1
            counts[row["type"]] += 1
            out_row = dict(row)
            out_row["set"] = set_name
            out_row["set_relpath"] = str(Path(set_name) / row["new_name"])
            materialized_rows.append(out_row)
        write_manifest(set_dir / "manifest.csv", materialized_rows)
        summary["sets"].append({
            "set": set_name,
            "dir": str(set_dir),
            "files": len(materialized_rows),
            "type_counts": dict(sorted(counts.items())),
            "materialized": dict(sorted(link_modes.items())),
        })

    with (out_dir / "manifest_runs.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
