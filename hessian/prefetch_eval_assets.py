#!/usr/bin/env python3
"""Download every reward-model asset before evaluation workers go offline."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


SCHEMA = "ratiodiff-eval-assets-20260924-v1"


def marker_valid(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("schema") == SCHEMA and all(Path(item).is_file() for item in payload["files"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prefetch(cache_root: Path) -> list[str]:
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    hf_cache = cache_root/"huggingface"
    open_clip_cache = cache_root/"open_clip"
    image_reward_cache = cache_root/"imagereward"
    for directory in (hf_cache, open_clip_cache, image_reward_cache):
        directory.mkdir(parents=True, exist_ok=True)

    files = []
    for repo_id in ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "yuvalkirstain/PickScore_v1"):
        snapshot = Path(snapshot_download(repo_id, cache_dir=str(hf_cache), resume_download=True))
        config = snapshot/"config.json"
        if not config.is_file():
            raise FileNotFoundError(f"Incomplete evaluation model: {repo_id}")
        files.append(str(config.resolve()))

    for repo_id, filename in (
        ("xswu/HPSv2", "HPS_v2.1_compressed.pt"),
        ("trl-lib/ddpo-aesthetic-predictor", "aesthetic-model.pth"),
    ):
        files.append(str(Path(hf_hub_download(repo_id, filename, cache_dir=str(hf_cache))).resolve()))

    import open_clip
    clip_path = open_clip.download_pretrained(
        open_clip.get_pretrained_cfg("ViT-L-14", "openai"), cache_dir=str(open_clip_cache)
    )
    if not clip_path or not Path(clip_path).is_file():
        raise FileNotFoundError("OpenCLIP ViT-L-14 weight was not cached")
    files.append(str(Path(clip_path).resolve()))

    # HPSv2's vendored OpenCLIP loader uses its default/HF cache. Instantiate
    # once on CPU so the exact base checkpoint needed by the offline scorer is present.
    from hpsv2.src.open_clip import create_model_and_transforms
    hps_model, _, _ = create_model_and_transforms(
        "ViT-H-14", "laion2B-s32B-b79K", precision="fp32", device="cpu",
        light_augmentation=True, output_dict=True,
    )
    del hps_model
    gc.collect()

    import ImageReward as RM
    image_reward = RM.load("ImageReward-v1.0", device="cpu", download_root=str(image_reward_cache))
    del image_reward
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    marker = cache_root/"prefetch_complete.json"
    if marker_valid(marker):
        print(f"eval_assets_reused={marker}", flush=True)
        return
    if args.offline:
        raise FileNotFoundError(
            f"Evaluation cache is incomplete: {marker}; run once without RATIODIFF_OFFLINE=1"
        )
    files = prefetch(cache_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema": SCHEMA, "files": files}, indent=2), encoding="utf-8")
    temporary.replace(marker)
    print(f"eval_assets_completed={marker}", flush=True)


if __name__ == "__main__":
    main()
