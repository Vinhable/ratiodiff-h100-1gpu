# RatioDiff SDXL + SD1.5 ablations — single H100 release

Clean, resumable 85k-pair implementation for **one H100 with at least 80 GB VRAM**.
This repository contains code and configuration only: no datasets, pretrained weights,
checkpoints, generated images, caches, or historical results.

## Experiment matrix

All runs use 85,000 binary preference pairs, one epoch, BF16 forward, FP32 master
weights, effective batch 64, and 1,329 optimizer updates.

| Order | Backbone | Variant | Resolution | Micro-batch | Accumulation |
|---:|---|---|---:|---:|---:|
| 1 | SDXL | `full` | 1024 | 4 | 16 |
| 2 | SD1.5 | `full` | 512 | 16 | 4 |
| 3 | SD1.5 | `no_ratio_correction` | 512 | 16 | 4 |
| 4 | SD1.5 | `fixed_regularization` | 512 | 16 | 4 |

The per-GPU micro-batches are the largest safe settings carried over from the
validated 80 GB Modal run. The larger accumulation values are required because
this release uses one GPU instead of four. Changing accumulation changes the
effective batch and invalidates the intended comparison.

Each full run writes a complete Accelerate checkpoint every **250 optimizer
steps** and at the final step. Relaunching the same command automatically finds
the newest complete checkpoint. Resume requires the same code, configuration,
variant, micro-batch, and one-GPU topology.

## Host requirements

- Linux, Python 3.11, Git, `flock`, and an NVIDIA driver compatible with CUDA 12.1.
- Exactly one H100 must be visible to the pipeline; at least 75 GiB VRAM and BF16.
- Enough local disk for the pretrained models, 85k parquet subset, optimizer
  checkpoints, generated evaluation images, and reward-model caches.
- Internet during initial environment/model/reward-weight setup, unless all assets
  are already mirrored locally.

## 1. Environment

```bash
git clone <THIS_REPOSITORY_URL> ratiodiff-h100-1gpu
cd ratiodiff-h100-1gpu
bash scripts/setup_h100.sh
source .venv/bin/activate
```

The setup script installs the CUDA 12.1 build of PyTorch 2.3.1 and the dependency
versions used by the successful Modal pilot. If the host provides a managed
PyTorch environment, it may be used instead, but run the tests and pilot again.

## 2. Required local assets

Prepare these paths on the H100 host:

- materialized Pick-a-Pic v2 parquet shards containing `jpg_0`, `jpg_1`,
  `label_0`, and `caption`;
- the binary manifest created below.

The first pipeline launch now bootstraps the other assets automatically. It
downloads SDXL base 1.0, `madebyollin/sdxl-vae-fp16-fix`, and Stable Diffusion
v1.5. The three exact evaluation prompt files are bundled under
`assets/eval_prompts`; their checksums are verified before the launcher creates
`setup/sdxl.json` and `setup/sd15.json`. Existing complete model assets are
reused, so an interrupted download can be resumed.

### Hugging Face dataset source

The exact parquet source expected by this release is:

- [liuhuohuo2/pick-a-pic-v2](https://huggingface.co/datasets/liuhuohuo2/pick-a-pic-v2)
- pinned source revision: `f602d48`

The complete repository is approximately 335 GB. For the 85k experiment it is
enough to stage the first 100 training shards; the manifest builder skips shard 0,
filters ties, and stops selecting files as soon as it has at least 85,000 binary
pairs:

```bash
hf download liuhuohuo2/pick-a-pic-v2 \
  --repo-type dataset \
  --revision f602d48 \
  --include 'data/train-000*.parquet' \
  --local-dir /data/pickapic_v2_source
```

This command stores the parquet files under `/data/pickapic_v2_source/data`.
If an equivalent materialized 85k subset is already available, reuse it instead
of downloading the full dataset. Training never reads image URLs: `jpg_0` and
`jpg_1` bytes must exist in the parquet files.

Create the exact binary-row manifest. Ties are excluded and no network fallback
is used during training:

```bash
python hessian/prepare_binary_local_manifest.py \
  --data-dir /data/pickapic_v2_source/data \
  --target-rows 85000 \
  --workers 16 \
  --repo-id liuhuohuo2/pick-a-pic-v2 \
  --revision f602d48 \
  --output /experiments/ratiodiff_h100/assets/binary_manifest.json
```

You may still bind the portable templates manually when models already exist at
custom paths:

```bash
python hessian/prepare_ratiodiff_experiments.py \
  --output-dir /experiments/ratiodiff_h100/setup \
  --data-dir /data/pickapic_v2_source/data \
  --manifest /experiments/ratiodiff_h100/assets/binary_manifest.json \
  --sdxl-model /models/stable-diffusion-xl-base-1.0 \
  --sdxl-vae /models/sdxl-vae-fp16-fix \
  --sd15-model /models/stable-diffusion-v1-5 \
  --prompt-dir /data/eval_prompts
```

Use `--prompt-scores-csv /path/to/full_scores.csv` instead of `--prompt-dir`
when recovering the exact established evaluation prompts.

The command creates `sdxl.json`, `sd15.json`, and absolute, separate work roots
for both families. Pass `--overwrite` to regenerate existing config files.

## 3. Tests and eight-step end-to-end pilot

Choose the physical H100 index with `GPU_ID` (default `0`):

```bash
GPU_ID=0 bash scripts/run_all_h100.sh /experiments/ratiodiff_h100/setup pilot
```

If the setup configs are absent, this command infers the parquet directory from
`/experiments/ratiodiff_h100/assets/binary_manifest.json` and performs the
bootstrap automatically. To use pre-staged assets, set `MODEL_ROOT`, or set the
individual `SDXL_MODEL`, `SDXL_VAE`, `SD15_MODEL`, and `PROMPT_DIR` variables.
Set `RATIODIFF_OFFLINE=1` to prohibit downloads and fail immediately if anything
is missing.

The pilot must complete training, inference, all five reported metrics, and report
generation for SDXL and all three SD1.5 variants. Pilot outputs are never reused as
full checkpoints or treated as quality evidence.

## 4. Full sequential pipeline

```bash
GPU_ID=0 bash scripts/run_all_h100.sh /experiments/ratiodiff_h100/setup full
```

For a detached run, use `tmux` or the scheduler supplied by the GPU provider. A
plain `nohup` example is:

```bash
nohup env GPU_ID=0 bash scripts/run_all_h100.sh \
  /experiments/ratiodiff_h100/setup full \
  > /experiments/ratiodiff_h100/controller.log 2>&1 &
```

Run only one copy. The lock file rejects accidental duplicate launchers. The order
is SDXL full first, then the three SD1.5 variants. Every variant starts from its
declared pretrained base; ablations do not continue from the preceding tuned model.

## Logs, checkpoints, and resume

- Controller: `/experiments/ratiodiff_h100/setup/h100-pipeline.log`
- Stage status: `<work_root>/pipeline-status.log`
- Training: `<work_root>/{pilot,full}/logs/<variant>/train.log`
- Diagnostics: `<work_root>/{pilot,full}/runs/<variant>/metrics.jsonl`
- Resume state: `<work_root>/full/runs/<variant>/checkpoint-{250,500,...}`

Follow the active controller log:

```bash
tail -F /experiments/ratiodiff_h100/setup/h100-pipeline.log
```

Rerun the same full command after an interruption. The launcher resumes from the
newest checkpoint containing `state.json`; it does not silently reuse incomplete
state or a checkpoint from another topology.

## Evaluation protocol

Each final checkpoint is compared with its corresponding pretrained base using
the same prompts and seeds: Euler scheduler, 50 steps, CFG 7.5, seed
`42 + prompt_id`; 1024 px for SDXL and 512 px for SD1.5. Reports contain
PickScore, HPSv2, Aesthetics, CLIP, and ImageReward. Full generated images are not
included in the compact archive.

## Safety notes

- Do not raise SDXL to micro-batch 8 on an 80 GB card: the Modal probe reached
  approximately 80–81 GiB and left no long-run safety margin.
- Do not run SDXL and SD1.5 concurrently on this one-GPU release.
- Do not change GPU topology when resuming an existing checkpoint.
- `local_files_only=true` is intentional for diffusion assets during expensive
  stages; missing assets fail before training instead of downloading mid-run.

The implementation is derived from Salesforce DiffusionDPO and retains its
upstream license in `LICENSE.txt`.
