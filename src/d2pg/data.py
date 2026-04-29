from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pose import interpolate_poses, load_tum_poses, pose_to_vec6, relative_pose


@dataclass(frozen=True)
class SequencePair:
    dataset: str
    sequence: str
    method: str
    gt_path: Path
    estimate_path: Path


def discover_alignanything_euroc(root: Path) -> list[SequencePair]:
    base = root / "AlignAnything" / "AlignAnything"
    gt_dir = base / "GT" / "euroc_mav"
    bench_dir = base / "benchmark" / "euroc_mav" / "pose"
    pairs: list[SequencePair] = []
    for method, filename in [("rovio", "rovio_poses.txt"), ("svo_stereo", "svo_poses.txt")]:
        method_dir = bench_dir / method
        if not method_dir.exists():
            continue
        for seq_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            gt_path = gt_dir / f"{seq_dir.name}.txt"
            estimate_path = seq_dir / filename
            if gt_path.exists() and estimate_path.exists():
                pairs.append(
                    SequencePair(
                        dataset="euroc_mav",
                        sequence=seq_dir.name,
                        method=method,
                        gt_path=gt_path,
                        estimate_path=estimate_path,
                    )
                )
    return pairs


def build_pair_samples(
    pair: SequencePair,
    stride: int = 1,
    horizon: int = 1,
    max_time_delta: float = 0.03,
    max_est_translation: float | None = None,
    max_gt_translation: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    est_t, est_poses = load_tum_poses(str(pair.estimate_path))
    gt_t, gt_poses = load_tum_poses(str(pair.gt_path))
    assoc_gt, assoc_dt = interpolate_poses(est_t, gt_t, gt_poses)
    valid = assoc_dt <= max_time_delta
    est_t = est_t[valid]
    est_poses = est_poses[valid]
    assoc_gt = assoc_gt[valid]
    if len(est_t) <= horizon:
        return np.empty((0, 6), dtype=np.float32), np.empty((0, 6), dtype=np.float32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(0, len(est_t) - horizon, stride):
        j = i + horizon
        est_rel = relative_pose(est_poses[i], est_poses[j])
        gt_rel = relative_pose(assoc_gt[i], assoc_gt[j])
        xs.append(pose_to_vec6(est_rel))
        ys.append(pose_to_vec6(gt_rel))
    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    keep = np.ones(len(x_arr), dtype=bool)
    if max_est_translation is not None:
        keep &= np.linalg.norm(x_arr[:, :3], axis=1) <= max_est_translation
    if max_gt_translation is not None:
        keep &= np.linalg.norm(y_arr[:, :3], axis=1) <= max_gt_translation
    return x_arr[keep], y_arr[keep]


def build_dataset(
    epa_root: Path,
    stride: int = 1,
    horizon: int = 1,
    max_time_delta: float = 0.03,
    max_est_translation: float | None = None,
    max_gt_translation: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[SequencePair], np.ndarray]:
    pairs = discover_alignanything_euroc(epa_root)
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    seq_ids: list[np.ndarray] = []
    kept_pairs: list[SequencePair] = []
    for idx, pair in enumerate(pairs):
        x, y = build_pair_samples(
            pair,
            stride=stride,
            horizon=horizon,
            max_time_delta=max_time_delta,
            max_est_translation=max_est_translation,
            max_gt_translation=max_gt_translation,
        )
        if len(x) == 0:
            continue
        all_x.append(x)
        all_y.append(y)
        seq_ids.append(np.full(len(x), idx, dtype=np.int64))
        kept_pairs.append(pair)
    if not all_x:
        raise RuntimeError(f"No usable AlignAnything/euroc_mav samples found under {epa_root}")
    return np.concatenate(all_x), np.concatenate(all_y), kept_pairs, np.concatenate(seq_ids)
