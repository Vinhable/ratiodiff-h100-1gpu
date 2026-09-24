#!/usr/bin/env python3
"""Materialize model/prompt assets and generate host-bound H100 configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROMPT_SOURCE = Path(__file__).resolve().parents[1]/"assets/eval_prompts"
PROMPTS = {
    "pickapic_v2.json": "c23a8d3f68090aed1567c27c562304ecd6097d891cb12b9c421607df053e526f",
    "partiprompt.json": "0e83f96f5ccad9bc931bfb680766d6cc5a469df64352754a48f98ab21a7eeac1",
    "hpdv2.json": "032a2a9ca1f89da15d3306b6ca94a8a6f6ee176b33cfc7b91d5d2bceb2c82114",
}
MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl_vae": "madebyollin/sdxl-vae-fp16-fix",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_prompts(directory: Path, offline: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in PROMPTS.items():
        path = directory/name
        if path.is_file() and sha256(path) == expected:
            print(f"prompt_reused={path}", flush=True)
            continue
        source = PROMPT_SOURCE/name
        if not source.is_file() or sha256(source) != expected:
            raise FileNotFoundError(f"Bundled prompt is missing/corrupt: {source}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        actual = sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Prompt checksum mismatch for {name}: {actual}")
        temporary.replace(path)
        print(f"prompt_installed={path}", flush=True)


def model_ready(path: Path, vae_only: bool = False) -> bool:
    if vae_only:
        return (path/"config.json").is_file() and any(path.glob("*.safetensors"))
    return (path/"model_index.json").is_file() and (path/"unet"/"config.json").is_file()


def ensure_model(repo_id: str, path: Path, offline: bool, vae_only: bool = False) -> None:
    if model_ready(path, vae_only):
        print(f"model_reused={path}", flush=True)
        return
    if offline:
        raise FileNotFoundError(f"Missing/incomplete model in offline mode: {path}")
    from huggingface_hub import snapshot_download
    path.mkdir(parents=True, exist_ok=True)
    print(f"model_download_start={repo_id} destination={path}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(path),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.bin", "*.onnx", "*.xml", "*.msgpack", "*.h5"],
    )
    if not model_ready(path, vae_only):
        raise RuntimeError(f"Downloaded model is incomplete: {path}")
    print(f"model_download_completed={path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--sdxl-model", type=Path)
    parser.add_argument("--sdxl-vae", type=Path)
    parser.add_argument("--sd15-model", type=Path)
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_dir = args.setup_dir.resolve()
    experiment = setup_dir.parent
    manifest = (args.manifest or experiment/"assets/binary_manifest.json").resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Binary manifest is missing: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    recorded_data_dir = payload.get("local_data_dir")
    if args.data_dir is None and not recorded_data_dir:
        raise ValueError("Manifest has no local_data_dir; pass --data-dir explicitly")
    data_dir = (args.data_dir or Path(recorded_data_dir)).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Parquet data directory is missing: {data_dir}")

    model_root = (args.model_root or experiment/"assets/models").resolve()
    sdxl_model = (args.sdxl_model or model_root/"stable-diffusion-xl-base-1.0").resolve()
    sdxl_vae = (args.sdxl_vae or model_root/"sdxl-vae-fp16-fix").resolve()
    sd15_model = (args.sd15_model or model_root/"stable-diffusion-v1-5").resolve()
    prompt_dir = (args.prompt_dir or experiment/"assets/eval_prompts").resolve()

    ensure_prompts(prompt_dir, args.offline)
    ensure_model(MODELS["sdxl"], sdxl_model, args.offline)
    ensure_model(MODELS["sdxl_vae"], sdxl_vae, args.offline, vae_only=True)
    ensure_model(MODELS["sd15"], sd15_model, args.offline)

    command = [
        sys.executable, str(Path(__file__).with_name("prepare_ratiodiff_experiments.py")),
        "--output-dir", str(setup_dir), "--data-dir", str(data_dir),
        "--manifest", str(manifest), "--sdxl-model", str(sdxl_model),
        "--sdxl-vae", str(sdxl_vae), "--sd15-model", str(sd15_model),
        "--prompt-dir", str(prompt_dir), "--overwrite",
    ]
    subprocess.run(command, check=True)
    print(f"bootstrap_completed={setup_dir}", flush=True)


if __name__ == "__main__":
    main()
