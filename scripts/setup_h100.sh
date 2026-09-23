#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$ROOT/.venv}"

python3.11 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip 'setuptools<81' wheel
"$VENV/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.3.1+cu121' 'torchvision==0.18.1+cu121'
"$VENV/bin/python" -m pip install -r "$ROOT/hessian/requirements_h100.txt"
"$VENV/bin/python" -m pip install 'git+https://github.com/openai/CLIP.git'
"$VENV/bin/python" - <<'PY'
import pathlib, site, urllib.request
p = pathlib.Path(site.getsitepackages()[0]) / "hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
p.parent.mkdir(parents=True, exist_ok=True)
if not p.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tgxs002/HPSv2/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz", p
    )
print(f"environment_ready={p}")
PY
