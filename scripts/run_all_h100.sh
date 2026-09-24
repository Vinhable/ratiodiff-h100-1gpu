#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_DIR="${1:?usage: bash scripts/run_all_h100.sh /absolute/setup/dir [pilot|full]}"
MODE="${2:-full}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"

if [[ "$MODE" != "pilot" && "$MODE" != "full" ]]; then
  echo "mode must be pilot or full" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_HOME="${HF_HOME:-$SETUP_DIR/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TORCH_HOME="${TORCH_HOME:-$SETUP_DIR/cache/torch}"
export HPS_ROOT="${HPS_ROOT:-$SETUP_DIR/cache/hpsv2}"

SDXL_CONFIG="$SETUP_DIR/sdxl.json"
SD15_CONFIG="$SETUP_DIR/sd15.json"
LOG="$SETUP_DIR/h100-pipeline.log"

if [[ ! -f "$SDXL_CONFIG" || ! -f "$SD15_CONFIG" ]]; then
  echo "setup configs missing; bootstrapping prompts/models and generating configs"
  BOOTSTRAP=("$PYTHON" "$ROOT/hessian/bootstrap_h100_assets.py" --setup-dir "$SETUP_DIR")
  [[ -n "${DATA_DIR:-}" ]] && BOOTSTRAP+=(--data-dir "$DATA_DIR")
  [[ -n "${MANIFEST:-}" ]] && BOOTSTRAP+=(--manifest "$MANIFEST")
  [[ -n "${MODEL_ROOT:-}" ]] && BOOTSTRAP+=(--model-root "$MODEL_ROOT")
  [[ -n "${SDXL_MODEL:-}" ]] && BOOTSTRAP+=(--sdxl-model "$SDXL_MODEL")
  [[ -n "${SDXL_VAE:-}" ]] && BOOTSTRAP+=(--sdxl-vae "$SDXL_VAE")
  [[ -n "${SD15_MODEL:-}" ]] && BOOTSTRAP+=(--sd15-model "$SD15_MODEL")
  [[ -n "${PROMPT_DIR:-}" ]] && BOOTSTRAP+=(--prompt-dir "$PROMPT_DIR")
  [[ "${RATIODIFF_OFFLINE:-0}" == "1" ]] && BOOTSTRAP+=(--offline)
  "${BOOTSTRAP[@]}"
fi

mkdir -p "$SETUP_DIR"

exec 9>"$SETUP_DIR/h100-pipeline.lock"
flock -n 9 || { echo "another H100 pipeline is already running" >&2; exit 1; }

"$PYTHON" "$ROOT/hessian/preflight_h100.py" \
  --sdxl-config "$SDXL_CONFIG" --sd15-config "$SD15_CONFIG" | tee -a "$LOG"

FLAGS=(--gpus "$GPU_ID" --execute --eval-python "$PYTHON")
if [[ "$MODE" == "pilot" ]]; then FLAGS+=(--pilot-only); fi

"$PYTHON" "$ROOT/hessian/run_ratiodiff_sdxl_matrix.py" \
  --config "$SDXL_CONFIG" "${FLAGS[@]}" 2>&1 | tee -a "$LOG"
"$PYTHON" "$ROOT/hessian/run_ratiodiff_sd15_ablations.py" \
  --config "$SD15_CONFIG" "${FLAGS[@]}" 2>&1 | tee -a "$LOG"
