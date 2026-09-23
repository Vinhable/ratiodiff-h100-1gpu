#!/usr/bin/env python3
"""SDXL/SD1.5 generation with the existing five-metric paired evaluation."""
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hessian.train_ratiodiff_sdxl import atomic_json, digest, load_config
from hessian.evaluate_standalone import _load_evaluator, score_metric, merge_scores

DATASETS = ("pickapic_v2", "partiprompt", "hpdv2")
METRICS = ("pickscore", "hpsv2", "aesthetics_clip", "imagereward")


def prepare_prompts(module, c, name, pilot):
    provenance = {}
    for dataset in DATASETS:
        source = Path(c["prompt_dir"]) / f"{dataset}.json"
        records = json.loads(source.read_text(encoding="utf-8"))
        ids = [int(row["id"]) for row in records]
        if not records or len(ids) != len(set(ids)) or any(not str(row["prompt"]).strip() for row in records):
            raise ValueError(f"Empty, duplicated or invalid prompt list: {source}")
        provenance[dataset] = dict(sha256=digest(source), source=str(source), count=len(records))
        if pilot:
            records = records[:2]
        dest = module._prompt_path(name, dataset)
        if dest.exists() and module._read_json(dest) != records:
            raise ValueError("Cannot reuse evaluation images with changed prompts")
        atomic_json(dest, records)
    atomic_json(module.EVAL_ROOT / name / "prompts/manifest.json", provenance)
    return provenance


def generate(module, c, name, run, step, rank, world):
    import torch
    from diffusers import (StableDiffusionXLPipeline, StableDiffusionPipeline,
                          AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler)
    e = c["eval"]
    if e["scheduler"] != "EulerDiscreteScheduler":
        raise ValueError("Only the declared Euler protocol is implemented")
    local = c["local_files_only"]
    is_xl = c["model_family"] == "sdxl"
    pipeline_cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
    extras = {} if is_xl else dict(safety_checker=None, requires_safety_checker=False)
    pipe = pipeline_cls.from_pretrained(c["model"], local_files_only=local, torch_dtype=torch.bfloat16, **extras)
    vae_kwargs = {} if c["vae"] else {"subfolder": "vae"}
    # BF16 VAE keeps pipeline latent/decoder dtypes consistent on both families.
    pipe.vae = AutoencoderKL.from_pretrained(c["vae"] or c["model"], **vae_kwargs,
                                           local_files_only=local, torch_dtype=torch.bfloat16)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for label in ("base", f"checkpoint-{step}"):
        if label != "base":
            # Release base UNet before loading the tuned one; sharded safetensors supported.
            del pipe.unet
            gc.collect(); torch.cuda.empty_cache()
            pipe.unet = UNet2DConditionModel.from_pretrained(run / "unet", local_files_only=True,
                                                            torch_dtype=torch.bfloat16).to("cuda")
        for dataset in DATASETS:
            rows = module._read_json(module._prompt_path(name, dataset))[rank::world]
            for begin in range(0, len(rows), e["batch_size"]):
                batch = [r for r in rows[begin:begin+e["batch_size"]]
                         if not module._valid_image(module._image_path(name, dataset, label, int(r["id"])))]
                if not batch:
                    continue
                gens = [torch.Generator("cuda").manual_seed(e["seed"]+int(r["id"])) for r in batch]
                with torch.inference_mode():
                    images = pipe([r["prompt"] for r in batch], generator=gens,
                        height=e["resolution"], width=e["resolution"],
                        num_inference_steps=e["steps"], guidance_scale=e["guidance_scale"]).images
                for row, img in zip(batch, images):
                    path = module._image_path(name, dataset, label, int(row["id"]))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix(".tmp.jpg")
                    img.save(temp, format="JPEG", quality=95, subsampling=0)
                    temp.replace(path)
                print(f"generate model={label} dataset={dataset} rank={rank} {min(begin+e['batch_size'],len(rows))}/{len(rows)}", flush=True)
    atomic_json(module.EVAL_ROOT / name / f"manifests/generation_rank{rank}.json", dict(rank=rank, world_size=world))


def validate_scores(module, args):
    items = module._score_items(args.eval_name, list(DATASETS), ["base", f"checkpoint-{args.selected_checkpoint}"], False)
    expected = {(r["dataset"], r["model"], int(r["prompt_id"])) for r in items}
    fields = dict(pickscore=["pickscore"], hpsv2=["hpsv2"], aesthetics_clip=["aesthetics", "clip"], imagereward=["imagereward"])
    for metric in METRICS:
        rows = module._read_json(module.EVAL_ROOT / args.eval_name / f"scores/full_{metric}.json")
        actual = [(r["dataset"], r["model"], int(r["prompt_id"])) for r in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f"Missing/duplicated scores: {metric}")
        if any(not math.isfinite(float(r[k])) for r in rows for k in fields[metric]):
            raise FloatingPointError(f"Nonfinite scores: {metric}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "generate", "score", "report"))
    p.add_argument("--config", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--metric", choices=METRICS)
    args = p.parse_args()
    c = load_config(args.config, args.pilot)
    if args.variant not in c["variants"]:
        raise ValueError("Variant is not configured for this backbone")
    root = Path(c["work_root"]).resolve() / ("pilot" if args.pilot else "full")
    run = root / "runs" / args.variant
    module = _load_evaluator(Path(__file__).resolve().parents[1], root)
    module.MODEL_ID = c["model"]
    name = args.variant
    provenance = prepare_prompts(module, c, name, args.pilot)
    if args.stage == "prepare":
        print(json.dumps(provenance))
        return
    done = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    step = done["step"]
    protocol = dict(recipe_id=done["recipe_id"], variant=args.variant, generation=c["eval"],
                    model=c["model"], vae=c["vae"], prompts=provenance, selected_checkpoint=step,
                    selection="final", generation_dtype="bf16", vae_dtype="bf16",
                    pilot=args.pilot, image_encoding="JPEG95_subsampling0")
    path = module.EVAL_ROOT / name / "protocol.json"
    if path.exists() and module._read_json(path) != protocol:
        raise ValueError("Evaluation protocol changed; use a new output directory")
    atomic_json(path, protocol)
    if args.stage == "generate":
        generate(module, c, name, run, step, args.rank, args.world_size)
        return
    eval_args = SimpleNamespace(eval_name=name, scope="full", selected_checkpoint=step, metric=args.metric)
    if args.stage == "score":
        # Check every image before scoring; do not accept a partial generation.
        for dataset in DATASETS:
            for row in module._read_json(module._prompt_path(name, dataset)):
                for label in ("base", f"checkpoint-{step}"):
                    if not module._valid_image(module._image_path(name, dataset, label, int(row["id"]))):
                        raise FileNotFoundError(f"Missing/corrupt image: {dataset}/{label}/{row['id']}")
        score_metric(module, eval_args, [step])
        return
    validate_scores(module, eval_args)
    merge_scores(module, eval_args, [step])
    module.build_report(name, step)
    report = module.EVAL_ROOT / name / "report/REPORT.md"
    lines = report.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# SD1.5"):
            lines[i] = f"# RatioDiff {c['model_family']} — {args.variant}" + (" (PILOT ONLY)" if args.pilot else "")
        elif line.startswith("Selected checkpoint:"):
            lines[i] = f"Selected checkpoint: final step {step}; fixed before evaluation."
        elif line.startswith("Generation:"):
            lines[i] = "Generation protocol: " + json.dumps(c["eval"])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(module.EVAL_ROOT / name / "complete.json", protocol)


if __name__ == "__main__":
    main()
