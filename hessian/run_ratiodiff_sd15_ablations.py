#!/usr/bin/env python3
"""SD1.5 full -> no-ratio-correction -> calibrated fixed regularization."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hessian.run_ratiodiff_sdxl_matrix import main

if __name__ == "__main__":
    main(default_config="ratiodiff_sd15_h100_1gpu.json")
