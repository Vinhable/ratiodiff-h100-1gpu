#!/usr/bin/env python3
"""Build an exact binary-preference streaming manifest from local parquet shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import pathlib
import re

import pyarrow.parquet as pq


SHARD_PATTERN = re.compile(r"^train-(\d+)-of-\d+-.*\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--target-rows", type=int, default=851_293)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--first-training-shard", type=int, default=1)
    parser.add_argument("--repo-id", default="liuhuohuo2/pick-a-pic-v2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_rows <= 0 or args.workers <= 0:
        raise ValueError("target rows and workers must be positive")

    shards: list[tuple[int, pathlib.Path]] = []
    held_out: list[str] = []
    for path in sorted(args.data_dir.glob("train-*.parquet")):
        match = SHARD_PATTERN.fullmatch(path.name)
        if not match:
            continue
        shard_index = int(match.group(1))
        filename = f"data/{path.name}"
        if shard_index < args.first_training_shard:
            held_out.append(filename)
        else:
            shards.append((shard_index, path))
    if not shards:
        raise RuntimeError(f"no training parquet shards found in {args.data_dir}")

    def scan(item: tuple[int, pathlib.Path]) -> tuple[int, str, int, list[int]]:
        shard_index, path = item
        table = pq.read_table(path, columns=["label_0"])
        labels = table.column("label_0").to_pylist()
        indices = [index for index, label in enumerate(labels) if label in (0, 1)]
        return shard_index, f"data/{path.name}", len(labels), indices

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        scanned = list(pool.map(scan, shards))
    scanned.sort(key=lambda row: row[0])

    selected: list[str] = []
    raw_rows: dict[str, int] = {}
    valid_indices: dict[str, list[int]] = {}
    total_binary = 0
    total_raw = 0
    for _, filename, row_count, indices in scanned:
        if total_binary >= args.target_rows:
            break
        selected.append(filename)
        raw_rows[filename] = row_count
        valid_indices[filename] = indices
        total_raw += row_count
        total_binary += len(indices)
        print(
            f"label_scan={len(selected):03d} file={pathlib.Path(filename).name} "
            f"raw={row_count} binary={len(indices)} cumulative={total_binary}",
            flush=True,
        )
    if total_binary < args.target_rows:
        raise RuntimeError(
            f"dataset exhausted at {total_binary} binary rows; need {args.target_rows}"
        )

    file_rows = {filename: len(valid_indices[filename]) for filename in selected}
    payload = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "target_rows": args.target_rows,
        "selection": (
            "local materialized train shards after held-out shard 0; exact human "
            "labels label_0 in {0,1}; deterministic shard/row shuffle"
        ),
        "pair_label_source": "human_label_0",
        "pair_label_policy": "prefiltered_binary_error",
        "held_out_files": held_out,
        "files": selected,
        "file_rows": file_rows,
        "raw_file_rows": raw_rows,
        "valid_row_indices": valid_indices,
        "available_binary_rows": total_binary,
        "scanned_raw_rows": total_raw,
        "num_shards": len(selected),
        "local_data_dir": str(args.data_dir.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "target_rows": args.target_rows,
        "available_binary_rows": total_binary,
        "scanned_raw_rows": total_raw,
        "shards": len(selected),
        "held_out_files": held_out,
    }, indent=2))


if __name__ == "__main__":
    main()
