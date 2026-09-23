#!/usr/bin/env python3
"""Run the Modal paper evaluator directly on HessianShell."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import pathlib
import sys
import types
from typing import Any


class _DummyImage:
    @classmethod
    def debian_slim(cls, **_: Any) -> "_DummyImage":
        return cls()

    def pip_install(self, *_: Any, **__: Any) -> "_DummyImage":
        return self

    def apt_install(self, *_: Any, **__: Any) -> "_DummyImage":
        return self

    def run_commands(self, *_: Any, **__: Any) -> "_DummyImage":
        return self

    def env(self, *_: Any, **__: Any) -> "_DummyImage":
        return self


class _DummyVolume:
    @classmethod
    def from_name(cls, *_: Any, **__: Any) -> "_DummyVolume":
        return cls()

    def commit(self) -> None:
        return None


class _DummyApp:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def function(self, *_: Any, **__: Any):
        return lambda function: function

    def local_entrypoint(self, *_: Any, **__: Any):
        return lambda function: function


def _load_evaluator(project_root: pathlib.Path, work_root: pathlib.Path):
    fake_modal = types.ModuleType("modal")
    fake_modal.App = _DummyApp
    fake_modal.Image = _DummyImage
    fake_modal.Volume = _DummyVolume
    sys.modules["modal"] = fake_modal
    sys.path.insert(0, str(project_root))
    module = importlib.import_module("modal_evaluate")
    module.CACHE_ROOT = pathlib.Path(
        os.environ.get("RATIO_EVAL_CACHE_DIR", str(work_root / "cache"))
    )
    module.RUNS_ROOT = work_root / "runs"
    module.EVAL_ROOT = work_root / "eval"
    return module


def _set_unet(module, pipe, train_run_name: str, model_label: str) -> None:
    import torch
    from diffusers import UNet2DConditionModel

    if model_label == "base":
        path = module.MODEL_ID
        kwargs = {"subfolder": "unet", "cache_dir": str(module.CACHE_ROOT / "huggingface")}
    else:
        step = int(model_label.split("-")[-1])
        path = module.RUNS_ROOT / train_run_name / f"checkpoint-{step}" / "unet"
        if not (path / "diffusion_pytorch_model.safetensors").is_file():
            path = module.RUNS_ROOT / train_run_name / "unet"
        kwargs = {}
    if not isinstance(path, str) and not (path / "diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing UNet for {model_label}: {path}")
    unet = UNet2DConditionModel.from_pretrained(path, torch_dtype=torch.float16, **kwargs)
    pipe.unet = unet.to("cuda", memory_format=torch.channels_last)


def _records_for_scope(module, eval_name: str, scope: str):
    if scope == "pilot":
        datasets = ["pickapic_v2"]
        return datasets, {
            "pickapic_v2": module._read_json(module._prompt_path(eval_name, "pickapic_v2", pilot=True))
        }
    datasets = ["pickapic_v2", "partiprompt", "hpdv2"]
    return datasets, {
        dataset: module._read_json(module._prompt_path(eval_name, dataset)) for dataset in datasets
    }


def _models(scope: str, checkpoints: list[int], selected: int) -> list[str]:
    if scope == "pilot":
        return ["base", *[f"checkpoint-{step}" for step in checkpoints]]
    return ["base", f"checkpoint-{selected}"]


def generate_worker(module, args: argparse.Namespace, checkpoints: list[int]) -> None:
    datasets, records_by_dataset = _records_for_scope(module, args.eval_name, args.scope)
    models = _models(args.scope, checkpoints, args.selected_checkpoint)
    pipe = module._load_pipeline()
    results = []
    for model in models:
        _set_unet(module, pipe, args.train_run_name, model)
        for dataset in datasets:
            records = [
                row for position, row in enumerate(records_by_dataset[dataset])
                if position % args.world_size == args.rank
            ]
            results.append(
                module._generate_records(
                    pipe,
                    args.eval_name,
                    dataset,
                    model,
                    records,
                    42,
                    8,
                    50,
                    7.5,
                )
            )
    output = module.EVAL_ROOT / args.eval_name / "manifests" / f"{args.scope}_generate_rank{args.rank}.json"
    module._write_json(output, {"rank": args.rank, "world_size": args.world_size, "results": results})
    print(json.dumps({"output": str(output), "results": results}, indent=2))


def _items(module, eval_name: str, scope: str, checkpoints: list[int], selected: int):
    datasets = ["pickapic_v2"] if scope == "pilot" else ["pickapic_v2", "partiprompt", "hpdv2"]
    models = _models(scope, checkpoints, selected)
    return module._score_items(eval_name, datasets, models, pilot=scope == "pilot")


def score_metric(module, args: argparse.Namespace, checkpoints: list[int]) -> None:
    items = _items(module, args.eval_name, args.scope, checkpoints, args.selected_checkpoint)
    rows = [
        {"dataset": row["dataset"], "model": row["model"], "prompt_id": row["prompt_id"]}
        for row in items
    ]
    if args.metric == "pickscore":
        values = module._score_pickscore(items)
        for row, value in zip(rows, values):
            row["pickscore"] = value
    elif args.metric == "hpsv2":
        values = module._score_hpsv2(items)
        for row, value in zip(rows, values):
            row["hpsv2"] = value
    elif args.metric == "aesthetics_clip":
        aesthetics, clip = module._score_aesthetic_and_clip(items)
        for row, aesthetic, clip_value in zip(rows, aesthetics, clip):
            row["aesthetics"] = aesthetic
            row["clip"] = clip_value
    elif args.metric == "imagereward":
        values = module._score_imagereward(items)
        for row, value in zip(rows, values):
            row["imagereward"] = value
    else:
        raise ValueError(args.metric)
    output = module.EVAL_ROOT / args.eval_name / "scores" / f"{args.scope}_{args.metric}.json"
    module._write_json(output, rows)
    print(json.dumps({"output": str(output), "rows": len(rows)}))


def merge_scores(module, args: argparse.Namespace, checkpoints: list[int]) -> None:
    items = _items(module, args.eval_name, args.scope, checkpoints, args.selected_checkpoint)
    lookup = {
        (row["dataset"], row["model"], int(row["prompt_id"])): row for row in items
    }
    for metric in ("pickscore", "hpsv2", "aesthetics_clip", "imagereward"):
        path = module.EVAL_ROOT / args.eval_name / "scores" / f"{args.scope}_{metric}.json"
        for row in module._read_json(path):
            key = (row["dataset"], row["model"], int(row["prompt_id"]))
            for name, value in row.items():
                if name not in {"dataset", "model", "prompt_id"}:
                    lookup[key][name] = float(value)
    output_name = "pilot_scores.csv" if args.scope == "pilot" else "full_scores.csv"
    output = module.EVAL_ROOT / args.eval_name / "scores" / output_name
    module._write_scores(output, items)
    print(json.dumps({"output": str(output), "rows": len(items)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "generate-worker", "score-metric", "merge", "select", "report"])
    parser.add_argument("--project-root", type=pathlib.Path, required=True)
    parser.add_argument("--work-root", type=pathlib.Path, required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--train-run-name", default="ratio_sd15_85k_4xa100_full")
    parser.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--checkpoints", default="100,200,300,400,500,600,665")
    parser.add_argument("--selected-checkpoint", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--metric", choices=["pickscore", "hpsv2", "aesthetics_clip", "imagereward"])
    parser.add_argument("--selection-policy", choices=["pilot", "final"], default="pilot")
    args = parser.parse_args()
    checkpoints = [int(value) for value in args.checkpoints.split(",") if value]
    module = _load_evaluator(args.project_root, args.work_root)
    if args.stage == "prepare":
        print(json.dumps(module.prepare_prompts(args.eval_name, pilot_count=100, seed=42), indent=2))
    elif args.stage == "generate-worker":
        generate_worker(module, args, checkpoints)
    elif args.stage == "score-metric":
        score_metric(module, args, checkpoints)
    elif args.stage == "merge":
        merge_scores(module, args, checkpoints)
    elif args.stage == "select":
        print(json.dumps(module.select_pilot_checkpoint(args.eval_name), indent=2))
    elif args.stage == "report":
        result = module.build_report(args.eval_name, args.selected_checkpoint)
        if args.selection_policy == "final":
            report_dir = module.EVAL_ROOT / args.eval_name / "report"
            report_path = report_dir / "REPORT.md"
            lines = report_path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if line.startswith("Selected checkpoint:"):
                    lines[index] = (
                        f"Selected checkpoint: `checkpoint-{args.selected_checkpoint}` "
                        "(final training checkpoint, fixed before evaluation)."
                    )
                    break
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            manifest_path = report_dir / "report_manifest.json"
            manifest = module._read_json(manifest_path)
            manifest["selection_policy"] = "final_training_checkpoint"
            module._write_json(manifest_path, manifest)
            result["selection_policy"] = "final_training_checkpoint"
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
