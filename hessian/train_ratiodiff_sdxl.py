#!/usr/bin/env python3
"""Shared full-denoiser trainer: SDXL recipe and SD1.5 ablations. No CPU offload.

Launch through torchrun; Accelerate handles DDP and portable state checkpoints.
The old train.py is intentionally unchanged. See RATIO_DIFF_SDXL_READY.md.
"""
from contextlib import ExitStack
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe_id(config):
    # Physical placement and inference batching do not define the training recipe.
    recipe = {k: v for k, v in config.items() if k not in ("work_root", "prompt_dir", "eval")}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Multiple generation/scoring ranks can write identical metadata concurrently.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name+".", suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for attempt in range(10):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if os.name != "nt" or attempt == 9:
                raise
            time.sleep(.01 * (attempt + 1))


def load_config(path, pilot=False):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if pilot:
        config.update(max_steps=8, warmup_steps=2, calibration_start=3,
                      calibration_end=6, checkpoint_steps=4, log_every=1)
    return config


def validate_config(c, world_size):
    family = c.get("model_family")
    if family not in ("sdxl", "sd15") or world_size < 1:
        raise ValueError("model_family must be sd15 or sdxl; world_size must be positive")
    expected = ["full"] if family == "sdxl" else ["full", "no_ratio_correction", "fixed_regularization"]
    if c.get("variants") != expected:
        raise ValueError("This experiment requires SDXL full only, SD1.5 full + two ablations")
    winner = c["loss"].get("winner_enabled", False)
    if (family == "sdxl" and winner) or (family == "sd15" and not winner):
        raise ValueError("SDXL disables winner auxiliary; SD1.5 ablations retain it")
    for name in ("samples", "epochs", "effective_batch", "micro_batch", "resolution",
                 "vae_batch_size", "log_every", "checkpoint_steps"):
        if not isinstance(c[name], int) or c[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    denominator = world_size * c["micro_batch"]
    if c["effective_batch"] % denominator:
        raise ValueError("effective_batch must divide exactly by GPUs * micro_batch")
    if c["resolution"] % 8:
        raise ValueError("Resolution must be divisible by 8")
    steps_per_epoch = math.ceil(c["samples"] / c["effective_batch"])
    steps = c.get("max_steps", steps_per_epoch * c["epochs"])
    if not 0 < steps <= steps_per_epoch * c["epochs"]:
        raise ValueError("max_steps outside the configured training budget")
    if not 0 <= c["warmup_steps"] < c["calibration_start"] <= c["calibration_end"] <= steps:
        raise ValueError("Predeclare a nonempty calibration window after warmup and within training")
    if c["eval"]["resolution"] != c["resolution"]:
        raise ValueError("Evaluation resolution must match the training protocol")
    for name in ("learning_rate", "head_learning_rate", "max_grad_norm"):
        if not math.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f"Invalid {name}")
    return steps, c["effective_batch"] // denominator, steps_per_epoch


def check_data(c):
    """Only local materialized parquet: preflight all selected shards, never download."""
    import pyarrow.parquet as pq
    manifest = json.loads(Path(c["manifest"]).read_text(encoding="utf-8"))
    if not manifest.get("valid_row_indices"):
        raise ValueError("Binary manifest with valid_row_indices required; ties must be excluded")
    available = 0
    for name in manifest["files"]:
        path = Path(c["data_dir"]) / Path(name).name
        if not path.is_file():
            raise FileNotFoundError(f"Missing local shard (network fallback disabled): {path}")
        pf = pq.ParquetFile(path)
        if not {"jpg_0", "jpg_1", "label_0", "caption"}.issubset(pf.schema.names):
            raise ValueError(f"Missing columns: {path}")
        indices = manifest["valid_row_indices"][name]
        if len(indices) != len(set(indices)) or any(i < 0 or i >= pf.metadata.num_rows for i in indices):
            raise ValueError(f"Bad binary row indices: {name}")
        if "file_rows" in manifest and manifest["file_rows"][name] != len(indices):
            raise ValueError(f"Manifest count mismatch: {name}")
        available += len(indices)
    if available < c["samples"]:
        raise ValueError(f"Only {available} binary pairs available; requested {c['samples']}")
    manifest["target_rows"] = c["samples"]
    return manifest


def image_tensor(payload, resolution):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    if isinstance(payload, dict):
        payload = payload["bytes"]
    with Image.open(io.BytesIO(payload)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        scale = resolution / min(width, height)
        new_w, new_h = max(resolution, round(width * scale)), max(resolution, round(height * scale))
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        left, top = (new_w - resolution) // 2, (new_h - resolution) // 2
        image = image.crop((left, top, left + resolution, top + resolution))
        pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 127.5 - 1
    return pixels, torch.tensor([height, width, top, left, resolution, resolution])


def collate(rows, resolution):
    import torch
    winners, losers, wt, lt, captions = [], [], [], [], []
    for row in rows:
        label = float(row["label_0"])
        if label not in (0, 1):
            raise ValueError("Nonbinary pair in binary manifest; refusing silent filtering")
        winner = 0 if label == 1 else 1
        w, wi = image_tensor(row[f"jpg_{winner}"], resolution)
        l, li = image_tensor(row[f"jpg_{1-winner}"], resolution)
        winners.append(w); losers.append(l); wt.append(wi); lt.append(li)
        captions.append(str(row["caption"]))
    return torch.stack(winners + losers), torch.stack(wt + lt), captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("full", "no_ratio_correction", "fixed_regularization"), default="full")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", help="Own complete checkpoint directory; topology must match")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    c = load_config(args.config, args.pilot)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    steps, accumulation, epoch_steps = validate_config(c, world)
    if args.variant not in c["variants"]:
        raise ValueError("Variant is not part of this backbone's experiment")
    is_xl = c["model_family"] == "sdxl"
    from ratiodiff_sdxl import (LossConfig, ConfidenceHead, Calibration, vp_omega,
                               as_epsilon, confidence_features, normalized_weights,
                               confidence_loss, denoiser_loss)
    loss_config = LossConfig(**c["loss"])
    manifest = check_data(c)
    c["manifest_sha256"] = digest(c["manifest"])
    project = Path(__file__).resolve().parents[1]
    c["source_hashes"] = {name: digest(project/name) for name in
        ("ratiodiff_sdxl.py", "hessian/train_ratiodiff_sdxl.py", "hessian/streaming_pickapic.py")}
    signature = recipe_id(c)
    root = Path(c["work_root"]).resolve() / ("pilot" if args.pilot else "full")
    out = root / "runs" / args.variant
    calibration_path = root / "runs" / "full" / "calibration.json"
    fixed_ref = fixed_win = None
    if args.variant == "fixed_regularization":
        cal = json.loads(calibration_path.read_text(encoding="utf-8"))
        if cal["recipe_id"] != signature or cal["variant"] != "full":
            raise ValueError("Calibration from a different recipe is not a controlled ablation")
        fixed_ref = cal["fixed_ref"]
        fixed_win = cal["fixed_win"]
        if cal["world_size"] != world:
            raise ValueError("Calibration DDP topology differs from ablation")
        if not math.isfinite(fixed_ref) or fixed_ref < 0:
            raise ValueError("Invalid calibration coefficient")
    print(json.dumps(dict(variant=args.variant, optimizer_steps=steps,
                          effective_batch=c["effective_batch"], accumulation=accumulation,
                          recipe_id=signature, local_shards=len(manifest["files"]),
                          fixed_ref=fixed_ref)), flush=True)
    if args.preflight:
        return
    import torch
    from functools import partial
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
    from hessian.streaming_pickapic import PickAPicStreamingDataset
    accelerator = Accelerator(mixed_precision="bf16")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU required; this trainer never CPU-offloads")
    if accelerator.num_processes != world:
        raise RuntimeError("Launcher world size mismatch")
    device, rank = accelerator.device, accelerator.process_index
    # Only rank 0 may decide whether a fresh run directory already exists.
    # Previously every rank performed this check while rank 0 immediately
    # created resolved_config.json, so slower ranks mistook the same new DDP
    # launch for a stale unfinished run.
    if accelerator.is_main_process:
        if (out / "completed.json").exists():
            raise FileExistsError(f"Run already completed: {out}; use a new work_root")
        if (out / "resolved_config.json").exists() and not args.resume:
            raise FileExistsError(f"Existing unfinished run: pass --resume or use a new work_root: {out}")
        out.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    resolved = dict(c, variant=args.variant, recipe_id=signature, world_size=world,
                    accumulation=accumulation, optimizer_steps=steps, fixed_ref=fixed_ref,
                    fixed_win=fixed_win,
                    reduction="latent_sum_pair_mean", weight_normalization="global_microbatch_mean",
                    rho_win=loss_config.rho_win, budget="per_pair_epsilon_output", optimizer="AdamW",
                    mixed_precision="bf16", cpu_offload=False)
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        meta = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
        if (checkpoint.parent != out.resolve() or meta["recipe_id"] != signature
                or meta["world_size"] != world or meta["variant"] != args.variant):
            raise ValueError("Resume must restore the SAME run, recipe and DDP topology")
    if accelerator.is_main_process:
        atomic_json(out / "resolved_config.json", resolved)
        atomic_json(out / "binary_manifest.json", manifest)
    accelerator.wait_for_everyone()
    set_seed(c["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    local = c["local_files_only"]
    pipeline_cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
    extras = {} if is_xl else dict(safety_checker=None, requires_safety_checker=False)
    pipe = pipeline_cls.from_pretrained(c["model"], torch_dtype=torch.bfloat16,
                                        local_files_only=local, **extras)
    # Separate FP32 policy/master weights; reference remains frozen BF16.
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(c["model"], subfolder="unet",
                                              local_files_only=local, torch_dtype=torch.float32)
    unet.enable_gradient_checkpointing()
    unet.train().requires_grad_(True)
    # Dropout off in BOTH paths; training mode still enables checkpointing.
    for module in unet.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    reference = pipe.unet.requires_grad_(False).eval().to(device)
    vae_kwargs = {} if c["vae"] else {"subfolder": "vae"}
    vae = AutoencoderKL.from_pretrained(c["vae"] or c["model"], **vae_kwargs, local_files_only=local,
                                       torch_dtype=torch.float32).requires_grad_(False).eval().to(device)
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2] if is_xl else [pipe.text_encoder]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2] if is_xl else [pipe.tokenizer]
    for encoder in text_encoders:
        encoder.requires_grad_(False).eval().to(device)
    del pipe
    scheduler = DDPMScheduler.from_pretrained(c["model"], subfolder="scheduler", local_files_only=local)
    ab = scheduler.alphas_cumprod.to(device)
    omega, valid_t = vp_omega(ab.cpu())
    omega, valid_t = omega.to(device), valid_t.to(device)
    head = ConfidenceHead(channels=unet.config.in_channels)
    opt = torch.optim.AdamW(unet.parameters(), lr=c["learning_rate"], weight_decay=0.01)
    head_opt = torch.optim.AdamW(head.parameters(), lr=c["head_learning_rate"], weight_decay=0.0)
    def lr_factor(step):
        return min(1.0, (step + 1) / max(1, c["warmup_steps"]))
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    unet, head, opt, head_opt = accelerator.prepare(unet, head, opt, head_opt)
    calibration = Calibration()
    win_calibration = Calibration()
    accelerator.register_for_checkpointing(lr_schedule, calibration, win_calibration)
    # Independent corruption/latent RNG from model and shard order.
    corruption = torch.Generator(device=device).manual_seed(c["seed"] + 100003 + rank)
    latent_rng = torch.Generator(device=device).manual_seed(c["seed"] + 200003 + rank)
    step = 0
    if args.resume:
        accelerator.load_state(str(checkpoint))
        step = meta["step"]
        states = torch.load(checkpoint / f"pair_rng_rank{rank}.pt", map_location="cpu", weights_only=True)
        corruption.set_state(states["corruption"])
        latent_rng.set_state(states["latent"])
    opt.zero_grad(); head_opt.zero_grad()

    def save_state():
        destination = out / f"checkpoint-{step}"
        if (destination / "state.json").exists():
            return
        accelerator.save_state(str(destination))
        torch.save(dict(corruption=corruption.get_state(), latent=latent_rng.get_state()),
                   destination / f"pair_rng_rank{rank}.pt")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            atomic_json(destination / "state.json", dict(step=step, recipe_id=signature,
                                                         world_size=world, variant=args.variant))
        accelerator.wait_for_everyone()

    def save_calibration():
        if accelerator.is_main_process:
            atomic_json(calibration_path, dict(variant="full", recipe_id=signature,
                        world_size=world, model_family=c["model_family"],
                        fixed_ref=calibration.coefficient(),
                        fixed_win=win_calibration.coefficient() if loss_config.winner_enabled else 0,
                        win_sums=win_calibration.state_dict(),
                        window=[c["calibration_start"], c["calibration_end"]],
                        precision="float64_sums_global_allreduce", **calibration.state_dict()))

    start_time = time.monotonic()
    for epoch in range(step // epoch_steps, c["epochs"]):
        if step >= steps:
            break
        start_sample = (step % epoch_steps) * c["micro_batch"] * accumulation
        dataset = PickAPicStreamingDataset(out / "binary_manifest.json", c["data_dir"],
                    out / "unused_stream_cache", rank, world, c["seed"] + epoch,
                    start_sample=start_sample, retries=1,
                    pad_to_multiple=c["micro_batch"] * accumulation)
        # Even if a shard disappears after preflight, do not fall back to network.
        def local_only(name):
            path = Path(c["data_dir"]) / Path(name).name
            if not path.is_file():
                raise FileNotFoundError(path)
            return path, False
        dataset.cache.obtain = local_only
        loader = torch.utils.data.DataLoader(dataset, batch_size=c["micro_batch"], num_workers=0,
                    collate_fn=partial(collate, resolution=c["resolution"]), pin_memory=device.type == "cuda")
        for micro, (pixels, time_ids, captions) in enumerate(loader):
            sync = (micro + 1) % accumulation == 0
            with torch.no_grad():
                latent_parts = []
                for chunk in pixels.split(c["vae_batch_size"]):
                    posterior = vae.encode(chunk.to(device, dtype=torch.float32)).latent_dist
                    latent_parts.append(posterior.sample(generator=latent_rng) * vae.config.scaling_factor)
                latents = torch.cat(latent_parts).to(torch.bfloat16)
                embeddings = []
                for tokenizer, encoder in zip(tokenizers, text_encoders):
                    ids = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt").input_ids.to(device)
                    enc = encoder(ids, output_hidden_states=True)
                    embeddings.append(enc.hidden_states[-2] if is_xl else enc.last_hidden_state)
                context = torch.cat(embeddings, dim=-1).repeat(2, 1, 1)
                conditions = (dict(text_embeds=enc.text_embeds.repeat(2, 1),
                                  time_ids=time_ids.to(device, dtype=torch.bfloat16)) if is_xl else None)
                pair_t = valid_t[torch.randint(len(valid_t), (len(captions),), device=device, generator=corruption)]
                t = pair_t.repeat(2)
                noise = torch.randn(latents.shape, device=device, dtype=torch.float32, generator=corruption)
                noisy = scheduler.add_noise(latents.float(), noise, t).to(torch.bfloat16)
                ref = reference(noisy, t, context, added_cond_kwargs=conditions).sample
                ref = as_epsilon(ref, noisy, t, ab, scheduler.config.prediction_type)
                rw, rl = ref.chunk(2); nw, nl = noise.chunk(2); xw, xl = noisy.chunk(2)
                features = confidence_features(xw, xl, rw, rl, pair_t, len(ab))
            with ExitStack() as stack:
                if not sync:
                    stack.enter_context(accelerator.no_sync(unet))
                    stack.enter_context(accelerator.no_sync(head))
                with accelerator.autocast():
                    pred = unet(noisy, t, context, added_cond_kwargs=conditions).sample
                pred = as_epsilon(pred, noisy, t, ab, scheduler.config.prediction_type)
                with torch.autocast(device_type="cuda", enabled=False):
                    logits = head(features)
                    weights = normalized_weights(logits, loss_config.w_min)
                    policy_loss, stats = denoiser_loss(*pred.chunk(2), rw, rl, nw, nl,
                        omega[pair_t], weights, loss_config, args.variant, fixed_ref, fixed_win)
                    conf = confidence_loss(logits, rw, rl, nw, nl, loss_config)
                    loss = policy_loss + loss_config.lambda_conf * conf
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite total loss")
                if args.variant == "full" and c["calibration_start"] <= step + 1 <= c["calibration_end"]:
                    calibration.update(stats)
                    if loss_config.winner_enabled:
                        win_calibration.update(stats, "win")
                accelerator.backward(loss / accumulation)
            if sync:
                grad = accelerator.clip_grad_norm_(unet.parameters(), c["max_grad_norm"])
                head_grad = accelerator.clip_grad_norm_(head.parameters(), c["max_grad_norm"])
                if not (torch.isfinite(grad) and torch.isfinite(head_grad)):
                    raise FloatingPointError("Nonfinite parameter gradient; no skipped batches")
                opt.step(); head_opt.step(); lr_schedule.step()
                opt.zero_grad(); head_opt.zero_grad()
                step += 1
                if step % c["log_every"] == 0 or step == steps:
                    # Gather quantiles across ranks for the last global microbatch.
                    record = dict(step=step, total_steps=steps, elapsed_seconds=time.monotonic()-start_time,
                                  loss=float(loss.detach()), confidence_bce=float(conf.detach()),
                                  lr=opt.param_groups[0]["lr"], parameter_grad_norm=float(grad),
                                  cuda_peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                                  diagnostic_scope="last_global_microbatch")
                    for name, values in stats.items():
                        values = accelerator.gather(values.float())
                        record[name] = dict(mean=float(values.mean()), quantiles=torch.quantile(values,
                            values.new_tensor([0, .05, .5, .95, 1])).cpu().tolist())
                    if accelerator.is_main_process:
                        line = json.dumps(record, allow_nan=False)
                        with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
                            log.write(line + "\n")
                        print(f"step={step}/{steps} loss={loss.item():.6g} conf={conf.item():.4g} "
                              f"z_q={record['log_ratio']['quantiles']} peakGB={record['cuda_peak_allocated_gb']:.2f}", flush=True)
                if args.variant == "full" and step == c["calibration_end"]:
                    save_calibration()
                if step % c["checkpoint_steps"] == 0 or step == steps:
                    save_state()
                if step >= steps:
                    break
    if step != steps:
        raise RuntimeError(f"Data exhausted at {step}/{steps}; not marking complete")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(unet).save_pretrained(out / "unet", safe_serialization=True)
        torch.save(accelerator.unwrap_model(head).state_dict(), out / "confidence_head.pt")
        if args.variant == "full":
            save_calibration()
        atomic_json(out / "completed.json", dict(step=step, recipe_id=signature, variant=args.variant))
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
