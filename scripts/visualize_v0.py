#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from d2pg.pose import rotation_error_deg


def trans_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)


def rot_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.asarray([rotation_error_deg(p, t) for p, t in zip(pred, gt)], dtype=np.float64)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize D2PG v0 smoke-test results.")
    parser.add_argument("--mlp", type=Path, default=Path("artifacts/correction_mlp_predictions.npz"))
    parser.add_argument("--diffusion", type=Path, default=Path("artifacts/diffusion_correction_predictions.npz"))
    parser.add_argument("--mlp-metrics", type=Path, default=Path("artifacts/correction_mlp_metrics.json"))
    parser.add_argument("--diffusion-metrics", type=Path, default=Path("artifacts/diffusion_correction_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()

    mlp = np.load(args.mlp)
    diff = np.load(args.diffusion)
    gt = diff["gt"]
    vio = diff["vio"]
    diffusion_mean = diff["diffusion_mean"]
    diffusion_best = diff["diffusion_best"]
    corrected = mlp["corrected"]
    sample_std = diff["sample_std"]

    series = {
        "Raw VIO/VO": vio,
        "MLP correction": corrected,
        "Diffusion mean": diffusion_mean,
        "Diffusion best-of-K": diffusion_best,
    }
    trans = {name: trans_errors(value, gt) for name, value in series.items()}
    rot = {name: rot_errors(value, gt) for name, value in series.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("ggplot")
    colors = {
        "Raw VIO/VO": "#3949ab",
        "MLP correction": "#d81b60",
        "Diffusion mean": "#00897b",
        "Diffusion best-of-K": "#f9a825",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(series)
    x = np.arange(len(names))
    rmse = [float(np.sqrt(np.mean(trans[name] ** 2))) for name in names]
    median = [float(np.median(trans[name])) for name in names]
    ax.bar(x - 0.18, rmse, width=0.36, label="RMSE", color=[colors[n] for n in names], alpha=0.9)
    ax.bar(x + 0.18, median, width=0.36, label="Median", color=[colors[n] for n in names], alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Translation error (m)")
    ax.set_title("D2PG v0 local translation correction")
    ax.legend()
    fig.tight_layout()
    bar_path = args.output_dir / "translation_bar.png"
    fig.savefig(bar_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, np.percentile(trans["Raw VIO/VO"], 99), 60)
    for name in names:
        ax.hist(trans[name], bins=bins, histtype="step", linewidth=2.0, label=name, color=colors[name], density=True)
    ax.set_xlabel("Translation error (m)")
    ax.set_ylabel("Density")
    ax.set_title("Translation error distribution")
    ax.legend()
    fig.tight_layout()
    hist_path = args.output_dir / "translation_hist.png"
    fig.savefig(hist_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    uncertainty = np.linalg.norm(sample_std[:, :3], axis=1)
    best_gain = trans["Raw VIO/VO"] - trans["Diffusion best-of-K"]
    ax.scatter(uncertainty, best_gain, s=12, alpha=0.55, color="#5e35b1")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Generated translation correction std")
    ax.set_ylabel("Best-of-K translation improvement (m)")
    ax.set_title("Does sample spread indicate useful correction candidates?")
    fig.tight_layout()
    scatter_path = args.output_dir / "uncertainty_vs_gain.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    rot_rmse = [float(np.sqrt(np.mean(rot[name] ** 2))) for name in names]
    ax.bar(names, rot_rmse, color=[colors[n] for n in names], alpha=0.85)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Rotation RMSE (deg)")
    ax.set_title("D2PG v0 local rotation error")
    fig.tight_layout()
    rot_path = args.output_dir / "rotation_bar.png"
    fig.savefig(rot_path, dpi=180)
    plt.close(fig)

    summary = {
        "figures": {
            "translation_bar": str(bar_path),
            "translation_hist": str(hist_path),
            "uncertainty_vs_gain": str(scatter_path),
            "rotation_bar": str(rot_path),
        },
        "translation_rmse": dict(zip(names, rmse)),
        "translation_median": dict(zip(names, median)),
        "rotation_rmse_deg": dict(zip(names, rot_rmse)),
        "metrics_json": {
            "mlp": load_json(args.mlp_metrics),
            "diffusion": load_json(args.diffusion_metrics),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
