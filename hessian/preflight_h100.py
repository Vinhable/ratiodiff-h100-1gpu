#!/usr/bin/env python3
"""Fail early unless the release is running on exactly one suitable H100."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hessian.train_ratiodiff_sdxl import check_data, load_config, validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdxl-config", required=True)
    parser.add_argument("--sd15-config", required=True)
    args = parser.parse_args()

    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    if "H100" not in props.name.upper():
        raise RuntimeError(f"Expected an H100, found {props.name}")
    memory_gib = props.total_memory / 2**30
    if memory_gib < 75:
        raise RuntimeError(f"At least 75 GiB VRAM is required, found {memory_gib:.1f} GiB")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required")

    configs = {}
    for family, path in (("sdxl", args.sdxl_config), ("sd15", args.sd15_config)):
        config = load_config(path)
        steps, accumulation, _ = validate_config(config, 1)
        manifest = check_data(config)
        configs[family] = {
            "micro_batch": config["micro_batch"],
            "accumulation": accumulation,
            "effective_batch": config["effective_batch"],
            "optimizer_steps": steps,
            "shards": len(manifest["files"]),
        }
    print(json.dumps({
        "gpu": props.name,
        "vram_gib": round(memory_gib, 2),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "configs": configs,
    }, indent=2))


if __name__ == "__main__":
    main()
