from __future__ import annotations

import math

import numpy as np


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out_shape = q.shape[:-1] + (3, 3)
    r = np.empty(out_shape, dtype=np.float64)
    r[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    r[..., 0, 1] = 2.0 * (xy - wz)
    r[..., 0, 2] = 2.0 * (xz + wy)
    r[..., 1, 0] = 2.0 * (xy + wz)
    r[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    r[..., 1, 2] = 2.0 * (yz - wx)
    r[..., 2, 0] = 2.0 * (xz - wy)
    r[..., 2, 1] = 2.0 * (yz + wx)
    r[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return r


def load_tum_poses(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts[:8]])
    if not rows:
        raise ValueError(f"No TUM poses found in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    t = arr[:, 0]
    trans = arr[:, 1:4]
    rot = quat_xyzw_to_rotmat(arr[:, 4:8])
    poses = np.tile(np.eye(4, dtype=np.float64), (len(arr), 1, 1))
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = trans
    finite = np.isfinite(t) & np.isfinite(poses.reshape(len(poses), -1)).all(axis=1)
    return t[finite], poses[finite]


def interpolate_poses(
    query_times: np.ndarray,
    source_times: np.ndarray,
    source_poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor pose association.

    For this first prototype the VIO outputs are sparse relative to GT.
    Nearest-neighbor association is enough to build local correction samples.
    """
    query_times = np.asarray(query_times, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    idx = np.searchsorted(source_times, query_times)
    idx0 = np.clip(idx - 1, 0, len(source_times) - 1)
    idx1 = np.clip(idx, 0, len(source_times) - 1)
    choose_prev = np.abs(source_times[idx0] - query_times) <= np.abs(source_times[idx1] - query_times)
    nearest = np.where(choose_prev, idx0, idx1)
    dt = np.abs(source_times[nearest] - query_times)
    return source_poses[nearest], dt


def relative_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.inv(a) @ b


def rotmat_to_rotvec(r: np.ndarray) -> np.ndarray:
    cos_theta = (np.trace(r) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    scale = theta / (2.0 * math.sin(theta))
    return scale * np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
        dtype=np.float64,
    )


def pose_to_vec6(t: np.ndarray) -> np.ndarray:
    return np.concatenate([t[:3, 3], rotmat_to_rotvec(t[:3, :3])])


def vec6_to_pose(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotvec_to_rotmat(v[3:6])
    t[:3, 3] = v[:3]
    return t


def rotvec_to_rotmat(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = v / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def compose_vec6(base_vec: np.ndarray, correction_vec: np.ndarray) -> np.ndarray:
    return pose_to_vec6(vec6_to_pose(correction_vec) @ vec6_to_pose(base_vec))


def translation_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3] - b[:3]))


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra = vec6_to_pose(a)[:3, :3]
    rb = vec6_to_pose(b)[:3, :3]
    dr = ra.T @ rb
    angle = np.linalg.norm(rotmat_to_rotvec(dr))
    return float(angle * 180.0 / math.pi)

