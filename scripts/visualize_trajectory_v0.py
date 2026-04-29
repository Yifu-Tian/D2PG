#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from d2pg.pose import vec6_to_pose


def integrate_deltas(deltas: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    positions = [pose[:3, 3].copy()]
    for delta in deltas:
        pose = pose @ vec6_to_pose(delta)
        positions.append(pose[:3, 3].copy())
    return np.asarray(positions, dtype=np.float64)


def trajectory_ate(pred: np.ndarray, gt: np.ndarray) -> float:
    n = min(len(pred), len(gt))
    return float(np.sqrt(np.mean(np.sum((pred[:n] - gt[:n]) ** 2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render D2PG v0 integrated local trajectory comparison.")
    parser.add_argument("--meta", type=Path, default=Path("artifacts/euroc_corrections_h20.json"))
    parser.add_argument("--mlp", type=Path, default=Path("artifacts/correction_mlp_predictions.npz"))
    parser.add_argument("--diffusion", type=Path, default=Path("artifacts/diffusion_correction_predictions.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--sequence-id", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=160)
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    mlp = np.load(args.mlp)
    diff = np.load(args.diffusion)
    seq_ids = diff["test_seq_ids"].astype(int)
    sequence_id = args.sequence_id
    if sequence_id is None:
        sequence_id = Counter(seq_ids.tolist()).most_common(1)[0][0]
    keep = np.where(seq_ids == sequence_id)[0]
    if len(keep) == 0:
        raise ValueError(f"No test samples found for sequence id {sequence_id}")
    keep = keep[: args.max_steps]

    series = {
        "GT": diff["gt"][keep],
        "Raw VIO/VO": diff["vio"][keep],
        "MLP correction": mlp["corrected"][keep],
        "Diffusion mean": diff["diffusion_mean"][keep],
        "Diffusion best-of-K": diff["diffusion_best"][keep],
    }
    traj = {name: integrate_deltas(value) for name, value in series.items()}
    seq_meta = meta["sequences"][sequence_id]
    title = f"{seq_meta['dataset']} / {seq_meta['sequence']} / {seq_meta['method']} (integrated local deltas)"
    colors = {
        "GT": "#111111",
        "Raw VIO/VO": "#3949ab",
        "MLP correction": "#d81b60",
        "Diffusion mean": "#00897b",
        "Diffusion best-of-K": "#f9a825",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, positions in traj.items():
        ax.plot(positions[:, 0], positions[:, 1], label=name, linewidth=2.0, color=colors[name])
        ax.scatter(positions[0, 0], positions[0, 1], s=32, color=colors[name], marker="o")
        ax.scatter(positions[-1, 0], positions[-1, 1], s=48, color=colors[name], marker="x")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    xy_path = args.output_dir / "trajectory_xy.png"
    fig.savefig(xy_path, dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    ax3 = fig.add_subplot(111, projection="3d")
    for name, positions in traj.items():
        ax3.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=name, linewidth=2.0, color=colors[name])
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    ax3.set_title(title)
    ax3.legend()
    fig.tight_layout()
    xyz_path = args.output_dir / "trajectory_3d.png"
    fig.savefig(xyz_path, dpi=180)
    plt.close(fig)

    gt_traj = traj["GT"]
    ate = {name: trajectory_ate(value, gt_traj) for name, value in traj.items() if name != "GT"}
    summary = {
        "sequence_id": int(sequence_id),
        "sequence": seq_meta,
        "num_integrated_deltas": int(len(keep)),
        "figures": {
            "trajectory_xy": str(xy_path),
            "trajectory_3d": str(xyz_path),
        },
        "integrated_local_ate": ate,
        "note": "This integrates overlapping horizon deltas as a visualization diagnostic, not a deployable SLAM backend.",
    }
    summary_path = args.output_dir / "trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
