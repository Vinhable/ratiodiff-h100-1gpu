"""Paper-style evaluation for the SD1.5 Ratio Diffusion-DPO Modal run.

Stages are intentionally resumable. Images, raw per-sample scores, manifests,
and reports are persisted in the ``dpo-ratio-eval`` Modal Volume.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import pathlib
import random
import statistics
import time
from collections import defaultdict
from typing import Any, Iterable

import modal


APP_NAME = "dpo-ratio-sd15-eval"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CACHE_ROOT = pathlib.Path("/cache")
RUNS_ROOT = pathlib.Path("/runs")
EVAL_ROOT = pathlib.Path("/eval")

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("dpo-ratio-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("dpo-ratio-runs", create_if_missing=True)
eval_volume = modal.Volume.from_name("dpo-ratio-eval", create_if_missing=True)

COMMON_ENV = {
    "HF_HOME": "/cache/huggingface",
    "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
    "TRANSFORMERS_CACHE": "/cache/huggingface",
    "TORCH_HOME": "/cache/torch",
    "HPS_ROOT": "/cache/hpsv2",
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

prompt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==0.24.7",
        "pyarrow==17.0.0",
        "pandas==2.2.2",
        "requests==2.32.3",
    )
    .env(COMMON_ENV)
)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        (
            "python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 "
            "torch==2.3.1+cu121 torchvision==0.18.1+cu121"
        ),
        (
            "python -m pip install diffusers==0.30.3 transformers==4.44.2 "
            "accelerate==0.33.0 huggingface_hub==0.24.7 safetensors==0.4.4 "
            "pillow==10.4.0 numpy==1.26.4 tqdm==4.66.5"
        ),
    )
    .env(COMMON_ENV)
)

score_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "tk")
    .run_commands(
        "python -m pip install --upgrade pip 'setuptools<81' wheel",
        (
            "python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 "
            "torch==2.3.1+cu121 torchvision==0.18.1+cu121"
        ),
        (
            "python -m pip install transformers==4.44.2 huggingface_hub==0.24.7 "
            "pillow==10.4.0 numpy==1.26.4 pandas==2.2.2 tqdm==4.66.5 "
            "open_clip_torch==2.26.1 hpsv2==1.2.0 image-reward==1.5 "
            "git+https://github.com/openai/CLIP.git"
        ),
        (
            "python -c \"import pathlib,site,urllib.request; "
            "p=pathlib.Path(site.getsitepackages()[0])/'hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz'; "
            "urllib.request.urlretrieve('https://raw.githubusercontent.com/tgxs002/HPSv2/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz', p)\""
        ),
    )
    .env(COMMON_ENV)
)

report_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pillow==10.4.0", "numpy==1.26.4")
    .env(COMMON_ENV)
)


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_path(eval_name: str, dataset: str, pilot: bool = False) -> pathlib.Path:
    suffix = "_pilot" if pilot else ""
    return EVAL_ROOT / eval_name / "prompts" / f"{dataset}{suffix}.json"


def _image_path(
    eval_name: str, dataset: str, model_label: str, prompt_id: int
) -> pathlib.Path:
    return (
        EVAL_ROOT
        / eval_name
        / "images"
        / dataset
        / model_label
        / f"{prompt_id:05d}.jpg"
    )


def _valid_image(path: pathlib.Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


@app.function(
    image=prompt_image,
    cpu=4,
    memory=8192,
    timeout=1800,
    volumes={"/cache": cache_volume, "/eval": eval_volume},
)
def prepare_prompts(eval_name: str, pilot_count: int = 100, seed: int = 42) -> dict[str, Any]:
    """Download and normalize the exact prompt sources used by the paper."""
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out_dir = EVAL_ROOT / eval_name / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    pick_file = hf_hub_download(
        "liuhuohuo2/pick-a-pic-v2",
        "data/test_unique-00000-of-00001-ecef20a3469ae6e6.parquet",
        repo_type="dataset",
        cache_dir=str(CACHE_ROOT / "huggingface"),
        resume_download=True,
    )
    pick_table = pq.read_table(pick_file, columns=["caption"])
    pick_prompts = [str(value.as_py()).strip() for value in pick_table["caption"]]
    pick_prompts = list(dict.fromkeys(prompt for prompt in pick_prompts if prompt))

    parti_file = hf_hub_download(
        "nateraw/parti-prompts",
        "PartiPrompts.tsv",
        repo_type="dataset",
        cache_dir=str(CACHE_ROOT / "huggingface"),
    )
    parti_frame = pd.read_csv(parti_file, sep="\t")
    parti_prompts = [str(value).strip() for value in parti_frame["Prompt"] if str(value).strip()]

    hpd_file = hf_hub_download(
        "ymhao/HPDv2",
        "test.json",
        repo_type="dataset",
        cache_dir=str(CACHE_ROOT / "huggingface"),
    )
    hpd_rows = json.loads(pathlib.Path(hpd_file).read_text(encoding="utf-8"))
    hpd_prompts = [str(row["prompt"]).strip() for row in hpd_rows if str(row["prompt"]).strip()]

    all_sets = {
        "pickapic_v2": pick_prompts,
        "partiprompt": parti_prompts,
        "hpdv2": hpd_prompts,
    }
    manifest: dict[str, Any] = {"seed": seed, "pilot_count": pilot_count, "datasets": {}}
    for dataset, prompts in all_sets.items():
        records = [{"id": index, "prompt": prompt} for index, prompt in enumerate(prompts)]
        _write_json(_prompt_path(eval_name, dataset), records)
        manifest["datasets"][dataset] = {"count": len(records)}

    rng = random.Random(seed)
    pilot_indices = sorted(rng.sample(range(len(pick_prompts)), min(pilot_count, len(pick_prompts))))
    pilot_records = [{"id": index, "prompt": pick_prompts[index]} for index in pilot_indices]
    _write_json(_prompt_path(eval_name, "pickapic_v2", pilot=True), pilot_records)
    manifest["datasets"]["pickapic_v2"]["pilot_count"] = len(pilot_records)
    manifest["sources"] = {
        "pickapic_v2": "liuhuohuo2/pick-a-pic-v2:test_unique",
        "partiprompt": "nateraw/parti-prompts:PartiPrompts.tsv",
        "hpdv2": "ymhao/HPDv2:test.json",
    }
    _write_json(out_dir / "manifest.json", manifest)
    eval_volume.commit()
    return manifest


def _load_pipeline():
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        cache_dir=str(CACHE_ROOT / "huggingface"),
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.to(memory_format=torch.channels_last)
    return pipe


def _set_unet(pipe, train_run_name: str, model_label: str) -> None:
    import torch
    from diffusers import UNet2DConditionModel

    if model_label == "base":
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID,
            subfolder="unet",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            cache_dir=str(CACHE_ROOT / "huggingface"),
        )
    else:
        checkpoint = int(model_label.split("-")[-1])
        unet_path = RUNS_ROOT / train_run_name / f"checkpoint-{checkpoint}" / "unet"
        if not (unet_path / "diffusion_pytorch_model.safetensors").exists():
            raise FileNotFoundError(f"Missing checkpoint UNet: {unet_path}")
        unet = UNet2DConditionModel.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
    pipe.unet = unet.to("cuda", memory_format=torch.channels_last)


def _generate_records(
    pipe,
    eval_name: str,
    dataset: str,
    model_label: str,
    records: list[dict[str, Any]],
    seed: int,
    batch_size: int,
    inference_steps: int,
    guidance_scale: float,
) -> dict[str, Any]:
    import torch

    completed = 0
    generated = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        pending = [row for row in batch if not _valid_image(_image_path(eval_name, dataset, model_label, row["id"]))]
        completed += len(batch) - len(pending)
        if not pending:
            continue
        prompts = [row["prompt"] for row in pending]
        generators = [
            torch.Generator(device="cuda").manual_seed(seed + int(row["id"])) for row in pending
        ]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            images = pipe(
                prompts,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                generator=generators,
                height=512,
                width=512,
            ).images
        for row, image in zip(pending, images):
            path = _image_path(eval_name, dataset, model_label, row["id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="JPEG", quality=95, subsampling=0)
            generated += 1
        completed += len(pending)
        if generated and generated % 50 < len(pending):
            eval_volume.commit()
    eval_volume.commit()
    return {"dataset": dataset, "model": model_label, "completed": completed, "generated": generated}


@app.function(
    image=inference_image,
    gpu="A100-40GB",
    cpu=4,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={"/cache": cache_volume, "/runs": runs_volume, "/eval": eval_volume},
)
def generate_pilot(
    eval_name: str,
    train_run_name: str,
    checkpoints: list[int],
    seed: int = 42,
    batch_size: int = 8,
    inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> dict[str, Any]:
    records = _read_json(_prompt_path(eval_name, "pickapic_v2", pilot=True))
    pipe = _load_pipeline()
    results = []
    for model_label in ["base", *[f"checkpoint-{step}" for step in checkpoints]]:
        _set_unet(pipe, train_run_name, model_label)
        results.append(
            _generate_records(
                pipe,
                eval_name,
                "pickapic_v2",
                model_label,
                records,
                seed,
                batch_size,
                inference_steps,
                guidance_scale,
            )
        )
    manifest = {
        "stage": "pilot_generation",
        "train_run_name": train_run_name,
        "seed": seed,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "results": results,
    }
    _write_json(EVAL_ROOT / eval_name / "pilot_generation.json", manifest)
    eval_volume.commit()
    return manifest


@app.function(
    image=inference_image,
    gpu="A100-40GB",
    cpu=4,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={"/cache": cache_volume, "/runs": runs_volume, "/eval": eval_volume},
)
def generate_full(
    eval_name: str,
    train_run_name: str,
    selected_checkpoint: int,
    seed: int = 42,
    batch_size: int = 8,
    inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> dict[str, Any]:
    pipe = _load_pipeline()
    results = []
    for model_label in ["base", f"checkpoint-{selected_checkpoint}"]:
        _set_unet(pipe, train_run_name, model_label)
        for dataset in ["pickapic_v2", "partiprompt", "hpdv2"]:
            records = _read_json(_prompt_path(eval_name, dataset))
            results.append(
                _generate_records(
                    pipe,
                    eval_name,
                    dataset,
                    model_label,
                    records,
                    seed,
                    batch_size,
                    inference_steps,
                    guidance_scale,
                )
            )
    manifest = {
        "stage": "full_generation",
        "train_run_name": train_run_name,
        "selected_checkpoint": selected_checkpoint,
        "seed": seed,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "results": results,
    }
    _write_json(EVAL_ROOT / eval_name / "full_generation.json", manifest)
    eval_volume.commit()
    return manifest


def _score_items(eval_name: str, datasets: list[str], models: list[str], pilot: bool) -> list[dict[str, Any]]:
    items = []
    for dataset in datasets:
        records = _read_json(_prompt_path(eval_name, dataset, pilot=pilot and dataset == "pickapic_v2"))
        for model in models:
            for row in records:
                image_path = _image_path(eval_name, dataset, model, row["id"])
                if not _valid_image(image_path):
                    raise FileNotFoundError(f"Missing or corrupt image: {image_path}")
                items.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "prompt_id": int(row["id"]),
                        "prompt": row["prompt"],
                        "image_path": str(image_path),
                    }
                )
    return items


def _batched(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _score_pickscore(items: list[dict[str, Any]], batch_size: int = 16) -> list[float]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=str(CACHE_ROOT / "huggingface")
    )
    model = AutoModel.from_pretrained(
        "yuvalkirstain/PickScore_v1",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        cache_dir=str(CACHE_ROOT / "huggingface"),
    ).eval().to("cuda")
    scores: list[float] = []
    for batch in _batched(items, batch_size):
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        prompts = [row["prompt"] for row in batch]
        image_inputs = processor(images=images, return_tensors="pt").to("cuda")
        text_inputs = processor(
            text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            values = (image_features * text_features).sum(dim=-1)
        scores.extend(float(value) for value in values.float().cpu())
        for image in images:
            image.close()
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def _score_hpsv2(items: list[dict[str, Any]], batch_size: int = 16) -> list[float]:
    import torch
    from huggingface_hub import hf_hub_download
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from PIL import Image

    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14",
        "laion2B-s32B-b79K",
        precision="fp16",
        device="cuda",
        light_augmentation=True,
        output_dict=True,
    )
    checkpoint_path = hf_hub_download(
        "xswu/HPSv2",
        "HPS_v2.1_compressed.pt",
        cache_dir=str(CACHE_ROOT / "huggingface"),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to("cuda")
    tokenizer = get_tokenizer("ViT-H-14")
    scores: list[float] = []
    for batch in _batched(items, batch_size):
        images = torch.stack(
            [preprocess(Image.open(row["image_path"]).convert("RGB")) for row in batch]
        ).to("cuda")
        texts = tokenizer([row["prompt"] for row in batch]).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs = model(images, texts)
            values = (outputs["image_features"] * outputs["text_features"]).sum(dim=-1)
        scores.extend(float(value) for value in values.float().cpu())
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def _score_aesthetic_and_clip(items: list[dict[str, Any]], batch_size: int = 32):
    import torch
    import torch.nn as nn
    import open_clip
    from huggingface_hub import hf_hub_download
    from PIL import Image

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )

        def forward(self, value):
            return self.layers(value)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device="cuda", cache_dir=str(CACHE_ROOT / "open_clip")
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    aesthetic_model = MLP().to("cuda").eval()
    weight_path = hf_hub_download(
        "trl-lib/ddpo-aesthetic-predictor",
        "aesthetic-model.pth",
        cache_dir=str(CACHE_ROOT / "huggingface"),
    )
    state = torch.load(weight_path, map_location="cpu")
    aesthetic_model.load_state_dict(state)
    aesthetic_scores: list[float] = []
    clip_scores: list[float] = []
    for batch in _batched(items, batch_size):
        images = torch.stack(
            [preprocess(Image.open(row["image_path"]).convert("RGB")) for row in batch]
        ).to("cuda")
        texts = tokenizer([row["prompt"] for row in batch]).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            clip_values = (image_features * text_features).sum(dim=-1)
            aesthetic_values = aesthetic_model(image_features.float()).squeeze(-1)
        clip_scores.extend(float(value) for value in clip_values.float().cpu())
        aesthetic_scores.extend(float(value) for value in aesthetic_values.float().cpu())
    del clip_model, aesthetic_model
    gc.collect()
    torch.cuda.empty_cache()
    return aesthetic_scores, clip_scores


def _score_imagereward(items: list[dict[str, Any]]) -> list[float]:
    import torch
    import ImageReward as RM

    model = RM.load(
        "ImageReward-v1.0",
        device="cuda",
        download_root=str(CACHE_ROOT / "imagereward"),
    )
    scores = []
    with torch.inference_mode():
        for row in items:
            scores.append(float(model.score(row["prompt"], row["image_path"])))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def _write_scores(path: pathlib.Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "prompt_id",
        "prompt",
        "image_path",
        "pickscore",
        "hpsv2",
        "aesthetics",
        "clip",
        "imagereward",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


def _score_stage(eval_name: str, items: list[dict[str, Any]], output_name: str) -> dict[str, Any]:
    started = time.time()
    output_path = EVAL_ROOT / eval_name / "scores" / output_name
    metrics = ["pickscore", "hpsv2", "aesthetics", "clip", "imagereward"]
    if output_path.exists():
        with output_path.open(encoding="utf-8", newline="") as handle:
            previous = {
                (row["dataset"], row["model"], int(row["prompt_id"])): row
                for row in csv.DictReader(handle)
            }
        for row in items:
            old = previous.get((row["dataset"], row["model"], row["prompt_id"]))
            if old:
                for metric in metrics:
                    if old.get(metric) not in (None, ""):
                        row[metric] = float(old[metric])

    if not all("pickscore" in row for row in items):
        for row, value in zip(items, _score_pickscore(items)):
            row["pickscore"] = value
        _write_scores(output_path, items)
        eval_volume.commit()
    if not all("hpsv2" in row for row in items):
        for row, value in zip(items, _score_hpsv2(items)):
            row["hpsv2"] = value
        _write_scores(output_path, items)
        eval_volume.commit()
    if not all("aesthetics" in row and "clip" in row for row in items):
        aesthetics, clip_scores = _score_aesthetic_and_clip(items)
        for row, aesthetic, clip_score in zip(items, aesthetics, clip_scores):
            row["aesthetics"] = aesthetic
            row["clip"] = clip_score
        _write_scores(output_path, items)
        eval_volume.commit()
    if not all("imagereward" in row for row in items):
        for row, value in zip(items, _score_imagereward(items)):
            row["imagereward"] = value
        _write_scores(output_path, items)
        eval_volume.commit()
    _write_scores(output_path, items)
    summary = {
        "output": str(output_path),
        "rows": len(items),
        "duration_seconds": time.time() - started,
        "metrics": metrics,
    }
    _write_json(output_path.with_suffix(".json"), summary)
    eval_volume.commit()
    return summary


@app.function(
    image=score_image,
    gpu="A100-40GB",
    cpu=4,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={"/cache": cache_volume, "/eval": eval_volume},
)
def score_pilot(eval_name: str, checkpoints: list[int]) -> dict[str, Any]:
    models = ["base", *[f"checkpoint-{step}" for step in checkpoints]]
    items = _score_items(eval_name, ["pickapic_v2"], models, pilot=True)
    return _score_stage(eval_name, items, "pilot_scores.csv")


@app.function(
    image=score_image,
    gpu="A100-40GB",
    cpu=4,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={"/cache": cache_volume, "/eval": eval_volume},
)
def score_full(eval_name: str, selected_checkpoint: int) -> dict[str, Any]:
    models = ["base", f"checkpoint-{selected_checkpoint}"]
    items = _score_items(eval_name, ["pickapic_v2", "partiprompt", "hpdv2"], models, pilot=False)
    return _score_stage(eval_name, items, "full_scores.csv")


def _read_score_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["prompt_id"] = int(row["prompt_id"])
        for metric in ["pickscore", "hpsv2", "aesthetics", "clip", "imagereward"]:
            row[metric] = float(row[metric])
    return rows


def _bootstrap_ci(values: list[float], seed: int = 42, samples: int = 10_000):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]]


@app.function(
    image=report_image,
    cpu=4,
    memory=8192,
    timeout=1800,
    volumes={"/eval": eval_volume},
)
def select_pilot_checkpoint(eval_name: str) -> dict[str, Any]:
    rows = _read_score_csv(EVAL_ROOT / eval_name / "scores" / "pilot_scores.csv")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    metrics = ["pickscore", "hpsv2", "aesthetics", "clip", "imagereward"]
    table = []
    for model, model_rows in sorted(grouped.items()):
        entry: dict[str, Any] = {"model": model, "n": len(model_rows)}
        for metric in metrics:
            entry[metric] = statistics.fmean(row[metric] for row in model_rows)
        table.append(entry)
    candidates = [row for row in table if row["model"].startswith("checkpoint-")]
    selected = max(candidates, key=lambda row: row["pickscore"])
    result = {
        "selection_metric": "pickscore",
        "selected_model": selected["model"],
        "selected_checkpoint": int(selected["model"].split("-")[-1]),
        "pilot_table": table,
        "guardrail_metrics": ["hpsv2", "aesthetics"],
    }
    _write_json(EVAL_ROOT / eval_name / "pilot_selection.json", result)
    eval_volume.commit()
    return result


def _make_grid(eval_name: str, dataset: str, selected_model: str, count: int = 8) -> str:
    from PIL import Image, ImageDraw

    records = _read_json(_prompt_path(eval_name, dataset))[:count]
    cell_w, cell_h = 512, 560
    canvas = Image.new("RGB", (cell_w * 2, cell_h * len(records)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(records):
        for col, model in enumerate(["base", selected_model]):
            image = Image.open(_image_path(eval_name, dataset, model, row["id"])).convert("RGB")
            canvas.paste(image, (col * cell_w, row_index * cell_h))
            draw.text((col * cell_w + 6, row_index * cell_h + 514), model, fill="black")
        text = row["prompt"].replace("\n", " ")[:150]
        draw.text((6, row_index * cell_h + 536), text, fill="black")
    path = EVAL_ROOT / eval_name / "report" / f"qualitative_{dataset}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)
    return str(path)


@app.function(
    image=report_image,
    cpu=4,
    memory=8192,
    timeout=1800,
    volumes={"/eval": eval_volume},
)
def build_report(eval_name: str, selected_checkpoint: int) -> dict[str, Any]:
    rows = _read_score_csv(EVAL_ROOT / eval_name / "scores" / "full_scores.csv")
    selected_model = f"checkpoint-{selected_checkpoint}"
    metrics = ["pickscore", "hpsv2", "aesthetics", "clip", "imagereward"]
    hpsv3_path = EVAL_ROOT / eval_name / "scores" / "hpsv3_scores.csv"
    if hpsv3_path.exists():
        with hpsv3_path.open(encoding="utf-8", newline="") as handle:
            hpsv3_lookup = {
                (row["dataset"], row["model"], int(row["prompt_id"])): float(row["hpsv3"])
                for row in csv.DictReader(handle)
            }
        for row in rows:
            key = (row["dataset"], row["model"], row["prompt_id"])
            if key not in hpsv3_lookup:
                raise ValueError(f"Incomplete HPSv3 scores, missing {key}")
            row["hpsv3"] = hpsv3_lookup[key]
        metrics.append("hpsv3")
    datasets = ["pickapic_v2", "partiprompt", "hpdv2"]
    summary_rows = []
    paired_rows = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        lookup = {(row["model"], row["prompt_id"]): row for row in dataset_rows}
        ids = sorted({row["prompt_id"] for row in dataset_rows})
        for metric in metrics:
            base = [lookup[("base", prompt_id)][metric] for prompt_id in ids]
            tuned = [lookup[(selected_model, prompt_id)][metric] for prompt_id in ids]
            delta = [right - left for left, right in zip(base, tuned)]
            wins = [1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in delta]
            delta_ci = _bootstrap_ci(delta)
            win_rate_ci = _bootstrap_ci(wins)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n": len(ids),
                    "base_mean": statistics.fmean(base),
                    "tuned_mean": statistics.fmean(tuned),
                    "absolute_improvement": statistics.fmean(delta),
                    "relative_improvement_percent": (
                        100.0 * statistics.fmean(delta) / abs(statistics.fmean(base))
                        if statistics.fmean(base) != 0
                        else math.nan
                    ),
                    "paired_win_rate": statistics.fmean(wins),
                    "delta_ci95_low": delta_ci[0],
                    "delta_ci95_high": delta_ci[1],
                    "win_rate_ci95_low": win_rate_ci[0],
                    "win_rate_ci95_high": win_rate_ci[1],
                }
            )
            for prompt_id, base_value, tuned_value in zip(ids, base, tuned):
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "prompt_id": prompt_id,
                        "base": base_value,
                        "tuned": tuned_value,
                        "delta": tuned_value - base_value,
                    }
                )

    report_dir = EVAL_ROOT / eval_name / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in [("summary.csv", summary_rows), ("paired_results.csv", paired_rows)]:
        with (report_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    _write_json(report_dir / "summary.json", summary_rows)

    lines = [
        "# SD1.5 Ratio Diffusion-DPO — paper-style evaluation",
        "",
        f"Selected checkpoint: `{selected_model}` (pilot PickScore on 100 held-out Pick-a-Pic v2 prompts).",
        "Generation: paired deterministic seeds (42 + prompt_id), 50 sampling steps, CFG 7.5, 512×512.",
        "Confidence intervals: paired non-parametric bootstrap, 10,000 resamples.",
        "",
        "| Dataset | Metric | N | Base | Tuned | Δ | Δ 95% CI | Win rate | Win-rate 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {metric} | {n} | {base_mean:.6f} | {tuned_mean:.6f} | "
            "{absolute_improvement:+.6f} | [{delta_ci95_low:+.6f}, {delta_ci95_high:+.6f}] | "
            "{paired_win_rate:.2%} | [{win_rate_ci95_low:.2%}, {win_rate_ci95_high:.2%}] |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "An improvement is treated as statistically supported only when the paired Δ 95% CI excludes zero. "
            "Reward models are imperfect proxies for human judgment, so disagreements across metrics are reported rather than averaged away.",
        ]
    )
    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    grids = [_make_grid(eval_name, dataset, selected_model) for dataset in datasets]
    manifest = {"selected_checkpoint": selected_checkpoint, "summary_rows": len(summary_rows), "grids": grids}
    _write_json(report_dir / "report_manifest.json", manifest)
    eval_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    stage: str,
    eval_name: str = "sd15-ratio-paper-eval-20260806",
    train_run_name: str = "sd15-ratio-pickscore-a100-1000-20260806",
    checkpoints: str = "200,400,600,800,1000",
    selected_checkpoint: int = 0,
) -> None:
    steps = [int(value) for value in checkpoints.split(",") if value.strip()]
    if stage == "prepare":
        result = prepare_prompts.remote(eval_name)
    elif stage == "pilot-generate":
        result = generate_pilot.remote(eval_name, train_run_name, steps)
    elif stage == "pilot-score":
        result = score_pilot.remote(eval_name, steps)
    elif stage == "pilot-select":
        result = select_pilot_checkpoint.remote(eval_name)
    elif stage == "full-generate":
        if selected_checkpoint <= 0:
            raise ValueError("--selected-checkpoint is required for full-generate")
        result = generate_full.remote(eval_name, train_run_name, selected_checkpoint)
    elif stage == "full-score":
        if selected_checkpoint <= 0:
            raise ValueError("--selected-checkpoint is required for full-score")
        result = score_full.remote(eval_name, selected_checkpoint)
    elif stage == "report":
        if selected_checkpoint <= 0:
            raise ValueError("--selected-checkpoint is required for report")
        result = build_report.remote(eval_name, selected_checkpoint)
    else:
        raise ValueError(
            "stage must be prepare, pilot-generate, pilot-score, pilot-select, "
            "full-generate, full-score, or report"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
