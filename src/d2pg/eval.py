from __future__ import annotations

import numpy as np

from .pose import rotation_error_deg, translation_error


def residual_targets(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # First prototype uses additive residuals in se(3)-like vector space.
    return y - x


def apply_residual(x: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return x + residual


def metrics(pred: np.ndarray, target: np.ndarray, prefix: str = "") -> dict[str, float]:
    trans = np.asarray([translation_error(p, t) for p, t in zip(pred, target)], dtype=np.float64)
    rot = np.asarray([rotation_error_deg(p, t) for p, t in zip(pred, target)], dtype=np.float64)
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}trans_rmse": float(np.sqrt(np.mean(trans**2))),
        f"{key}trans_median": float(np.median(trans)),
        f"{key}rot_rmse_deg": float(np.sqrt(np.mean(rot**2))),
        f"{key}rot_median_deg": float(np.median(rot)),
    }
