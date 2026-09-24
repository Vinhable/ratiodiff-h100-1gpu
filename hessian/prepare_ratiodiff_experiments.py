#!/usr/bin/env python3
"""Bind portable templates to a GPU host's existing data/model paths.

Optionally recover exact held-out prompts from a previous full_scores.csv.
Never downloads, trains, deletes old runs, or modifies the original templates.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hessian.train_ratiodiff_sdxl import atomic_json, digest


def extract_prompts(scores_file):
    records = {name: {} for name in ("pickapic_v2", "partiprompt", "hpdv2")}
    with Path(scores_file).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            if dataset not in records:
                continue
            key, prompt = int(row["prompt_id"]), row["prompt"]
            if key in records[dataset] and records[dataset][key] != prompt:
                raise ValueError(f"Conflicting prompt for {dataset}/{key}")
            records[dataset][key] = prompt
    if any(not rows for rows in records.values()):
        raise ValueError("Source CSV must contain all three evaluation benchmarks")
    return {name: [dict(id=i, prompt=rows[i]) for i in sorted(rows)] for name, rows in records.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sdxl-model", required=True)
    p.add_argument("--sdxl-vae", required=True)
    p.add_argument("--sd15-model", required=True)
    p.add_argument("--overwrite", action="store_true",
                   help="Replace only generated sdxl.json/sd15.json files")
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt-dir")
    prompts.add_argument("--prompt-scores-csv", help="Reuse exact prompts, never training calibration scores")
    args = p.parse_args()
    dest = Path(args.output_dir).resolve()
    generated = [dest/"sdxl.json", dest/"sd15.json"]
    if not args.overwrite and any(path.exists() for path in generated):
        raise FileExistsError("Configuration already exists; pass --overwrite to regenerate it")
    dest.mkdir(parents=True, exist_ok=True)
    prompt_dir = Path(args.prompt_dir).resolve() if args.prompt_dir else dest/"prompts"
    if args.prompt_scores_csv:
        for name, rows in extract_prompts(args.prompt_scores_csv).items():
            atomic_json(prompt_dir/f"{name}.json", rows)
        atomic_json(prompt_dir/"source.json", dict(source=str(Path(args.prompt_scores_csv).resolve()),
                                                  sha256=digest(args.prompt_scores_csv)))
    for family, filename in (("sdxl", "ratiodiff_xl_h100_1gpu.json"),
                             ("sd15", "ratiodiff_sd15_h100_1gpu.json")):
        c = json.loads(Path(__file__).with_name(filename).read_text())
        c.update(data_dir=str(Path(args.data_dir).resolve()), manifest=str(Path(args.manifest).resolve()),
                 model=str(Path(args.sdxl_model if family == "sdxl" else args.sd15_model).resolve()),
                 vae=str(Path(args.sdxl_vae).resolve()) if family == "sdxl" else None,
                 prompt_dir=str(prompt_dir), work_root=str(dest/f"runtime_{family}"))
        atomic_json(dest/f"{family}.json", c)
        print(f"Prepared {family}: {dest/f'{family}.json'}")


if __name__ == "__main__":
    main()
