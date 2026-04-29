#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from d2pg.eval import apply_residual, metrics, residual_targets
from d2pg.models import DiffusionCorrectionMLP


def split_by_sequence(seq_ids: np.ndarray, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(int(x) for x in np.unique(seq_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    test_count = max(1, len(unique) // 5)
    test_ids = set(unique[:test_count])
    is_test = np.asarray([int(x) in test_ids for x in seq_ids], dtype=bool)
    return ~is_test, is_test


def standardize(train: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (value - mean) / std, mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny conditional diffusion correction model.")
    parser.add_argument("--data", type=Path, default=Path("artifacts/euroc_corrections.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/diffusion_correction_metrics.json"))
    parser.add_argument("--predictions-output", type=Path, default=Path("artifacts/diffusion_correction_predictions.npz"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.data)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    seq_ids = data["seq_ids"]
    residual = residual_targets(x, y).astype(np.float32)
    train_mask, test_mask = split_by_sequence(seq_ids, args.seed)

    x_train_raw, x_test_raw = x[train_mask], x[test_mask]
    r_train_raw = residual[train_mask]
    x_train, x_mean, x_std = standardize(x_train_raw, x_train_raw)
    x_test = (x_test_raw - x_mean) / x_std
    r_train, r_mean, r_std = standardize(r_train_raw, r_train_raw)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(r_train))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = DiffusionCorrectionMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for cond, clean in loader:
            t = torch.rand(len(cond))
            noise = torch.randn_like(clean)
            noisy = (1.0 - t[:, None]) * clean + t[:, None] * noise
            pred_clean = model(noisy, t, cond)
            loss = loss_fn(pred_clean, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if epoch in {0, args.epochs - 1}:
            print(f"epoch {epoch + 1:03d}/{args.epochs}: loss={np.mean(losses):.6f}")

    model.eval()
    cond = torch.from_numpy(x_test)
    sample_preds = []
    with torch.no_grad():
        for _ in range(args.samples):
            noisy = torch.randn((len(x_test), 6), dtype=torch.float32)
            t = torch.ones(len(x_test), dtype=torch.float32)
            pred = model(noisy, t, cond).numpy()
            sample_preds.append(pred * r_std + r_mean)
    sample_preds_np = np.stack(sample_preds, axis=0)
    mean_residual = sample_preds_np.mean(axis=0)
    corrected_mean = apply_residual(x_test_raw, mean_residual)

    # Oracle best-of-K is not deployable, but it measures whether the generative
    # samples contain useful hypotheses.
    gt = y[test_mask]
    best_indices = []
    for i in range(len(x_test_raw)):
        candidates = apply_residual(
            np.repeat(x_test_raw[i : i + 1], args.samples, axis=0),
            sample_preds_np[:, i, :],
        )
        trans_err = np.linalg.norm(candidates[:, :3] - gt[i, :3], axis=1)
        best_indices.append(int(np.argmin(trans_err)))
    best_corrected = np.asarray(
        [apply_residual(x_test_raw[i : i + 1], sample_preds_np[j, i : i + 1])[0] for i, j in enumerate(best_indices)],
        dtype=np.float32,
    )

    report = {
        "num_train": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "epochs": args.epochs,
        "samples": args.samples,
        "baseline": metrics(x_test_raw, gt, prefix="vio"),
        "diffusion_mean": metrics(corrected_mean, gt, prefix="diffusion_mean"),
        "diffusion_oracle_best_of_k": metrics(best_corrected, gt, prefix="diffusion_best"),
        "sample_std_mean": sample_preds_np.std(axis=0).mean(axis=0).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.predictions_output,
        vio=x_test_raw,
        gt=gt,
        diffusion_mean=corrected_mean,
        diffusion_best=best_corrected,
        sample_residuals=sample_preds_np,
        sample_std=sample_preds_np.std(axis=0),
        best_indices=np.asarray(best_indices, dtype=np.int64),
        test_seq_ids=seq_ids[test_mask],
    )
    print(json.dumps(report, indent=2))
    _ = rng


if __name__ == "__main__":
    main()
