#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/build_dataset.py \
  --output artifacts/euroc_corrections_h20.npz \
  --horizon 20 \
  --stride 5 \
  --max-est-translation 2.0 \
  --max-gt-translation 1.0
python scripts/train_correction.py --data artifacts/euroc_corrections_h20.npz --epochs 20
python scripts/train_diffusion_correction.py --data artifacts/euroc_corrections_h20.npz --epochs 20 --samples 16
python scripts/visualize_v0.py
python scripts/visualize_trajectory_v0.py
