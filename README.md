# RatioDiff SDXL — single H100

Clean, resumable implementation of the corrected RatioDiff SDXL recipe for one
H100 with at least 80 GB VRAM. This repository contains SDXL code only.

## Training recipe

- SDXL base 1.0 with `madebyollin/sdxl-vae-fp16-fix`
- 85,000 binary Pick-a-Pic v2 preference pairs, one epoch
- 1024 × 1024, BF16 forward, FP32 trainable UNet/master weights
- one H100, micro-batch 4, accumulation 16, effective batch 64
- 1,329 optimizer steps, checkpoint every 250 steps
- full SDXL denoiser fine-tuning with frozen reference, VAE, and both text encoders
- adaptive reference regularization without a winner-denoising auxiliary
- no CPU offload

The configuration is in `hessian/ratiodiff_xl_h100_1gpu.json`. Resume requires
the same code, configuration, and one-GPU topology.

## Environment

```bash
git clone https://github.com/Vinhable/ratiodiff-h100-1gpu.git
cd ratiodiff-h100-1gpu
bash scripts/setup_h100.sh
source .venv/bin/activate
```

The host needs Linux, Python 3.11, Git, `flock`, a CUDA-compatible NVIDIA driver,
one H100 with at least 75 GiB visible VRAM, and enough disk for models, parquet
data, checkpoints, evaluation images, and reward-model caches.

## Dataset

The expected source is
[liuhuohuo2/pick-a-pic-v2](https://huggingface.co/datasets/liuhuohuo2/pick-a-pic-v2),
pinned to revision `f602d48`. Training consumes materialized parquet shards with
`jpg_0`, `jpg_1`, `label_0`, and `caption`; it never downloads image URLs.

```bash
hf download liuhuohuo2/pick-a-pic-v2 \
  --repo-type dataset \
  --revision f602d48 \
  --include 'data/train-000*.parquet' \
  --local-dir /data/pickapic_v2_source

python hessian/prepare_binary_local_manifest.py \
  --data-dir /data/pickapic_v2_source/data \
  --target-rows 85000 \
  --workers 16 \
  --repo-id liuhuohuo2/pick-a-pic-v2 \
  --revision f602d48 \
  --output /experiments/ratiodiff_h100/assets/binary_manifest.json
```

The first launcher invocation downloads or reuses SDXL, the SDXL VAE, bundled
evaluation prompts, and the five evaluator model caches. To bind existing local
assets manually:

```bash
python hessian/prepare_ratiodiff_experiments.py \
  --output-dir /experiments/ratiodiff_h100/setup \
  --data-dir /data/pickapic_v2_source/data \
  --manifest /experiments/ratiodiff_h100/assets/binary_manifest.json \
  --sdxl-model /models/stable-diffusion-xl-base-1.0 \
  --sdxl-vae /models/sdxl-vae-fp16-fix \
  --prompt-dir /data/eval_prompts
```

## Pilot and full run

Run the eight-step end-to-end pilot first:

```bash
GPU_ID=0 bash scripts/run_sdxl_h100.sh \
  /experiments/ratiodiff_h100/setup pilot
```

Then run the complete 85k × 1 epoch pipeline:

```bash
GPU_ID=0 bash scripts/run_sdxl_h100.sh \
  /experiments/ratiodiff_h100/setup full
```

For a detached run:

```bash
nohup env GPU_ID=0 bash scripts/run_sdxl_h100.sh \
  /experiments/ratiodiff_h100/setup full \
  > /experiments/ratiodiff_h100/controller.log 2>&1 &
```

Set `MODEL_ROOT`, `SDXL_MODEL`, `SDXL_VAE`, `DATA_DIR`, `MANIFEST`, or
`PROMPT_DIR` to override inferred paths. Set `RATIODIFF_OFFLINE=1` to prohibit
downloads and fail immediately when an asset is absent.

## Logs, checkpoints, and resume

- Controller: `/experiments/ratiodiff_h100/setup/h100-sdxl-pipeline.log`
- Status: `<work_root>/pipeline-status.log`
- Training: `<work_root>/{pilot,full}/logs/full/train.log`
- Diagnostics: `<work_root>/{pilot,full}/runs/full/metrics.jsonl`
- Checkpoints: `<work_root>/full/runs/full/checkpoint-{250,500,...}`

```bash
tail -F /experiments/ratiodiff_h100/setup/h100-sdxl-pipeline.log
```

Rerunning the same full command resumes from the newest complete checkpoint.
Use a new `work_root` for an intentionally fresh experiment or a changed recipe.

## Evaluation

The final checkpoint is compared with pretrained SDXL using identical prompts
and seeds: Euler scheduler, 50 steps, CFG 7.5, 1024 px, and seed
`42 + prompt_id`. Reports include PickScore, HPSv2, Aesthetics, CLIP, and
ImageReward, followed by a compact result archive.

Do not increase the training micro-batch above 4 on an 80 GB card without a new
memory pilot. A previous micro-batch-8 probe exhausted the available safety
margin.

The implementation is derived from Salesforce DiffusionDPO and retains its
upstream license in `LICENSE.txt`.
