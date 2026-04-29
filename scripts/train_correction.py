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
from d2pg.models import CorrectionMLP


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
    parser = argparse.ArgumentParser(description="Train deterministic D2PG correction baseline.")
    parser.add_argument("--data", type=Path, default=Path("artifacts/euroc_corrections.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/correction_mlp_metrics.json"))
    parser.add_argument("--predictions-output", type=Path, default=Path("artifacts/correction_mlp_predictions.npz"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    r_train_raw, r_test_raw = residual[train_mask], residual[test_mask]
    x_train, x_mean, x_std = standardize(x_train_raw, x_train_raw)
    x_test = (x_test_raw - x_mean) / x_std
    r_train, r_mean, r_std = standardize(r_train_raw, r_train_raw)
    r_test = (r_test_raw - r_mean) / r_std

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(r_train))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = CorrectionMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for xb, rb in loader:
            pred = model(xb)
            loss = loss_fn(pred, rb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if epoch in {0, args.epochs - 1}:
            print(f"epoch {epoch + 1:03d}/{args.epochs}: loss={np.mean(losses):.6f}")

    model.eval()
    with torch.no_grad():
        pred_r_norm = model(torch.from_numpy(x_test)).numpy()
    pred_r = pred_r_norm * r_std + r_mean
    corrected = apply_residual(x_test_raw, pred_r)
    baseline = metrics(x_test_raw, y[test_mask], prefix="vio")
    corrected_metrics = metrics(corrected, y[test_mask], prefix="corrected")
    zero_residual = apply_residual(x_test_raw, np.tile(r_train_raw.mean(axis=0, keepdims=True), (len(x_test_raw), 1)))
    mean_metrics = metrics(zero_residual, y[test_mask], prefix="mean_residual")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "num_train": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "epochs": args.epochs,
        "baseline": baseline,
        "mean_residual_baseline": mean_metrics,
        "correction_mlp": corrected_metrics,
        "sample_prediction": {
            "vio_delta": x_test_raw[0].tolist(),
            "gt_delta": y[test_mask][0].tolist(),
            "corrected_delta": corrected[0].tolist(),
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.predictions_output,
        vio=x_test_raw,
        gt=y[test_mask],
        corrected=corrected,
        predicted_residual=pred_r,
        test_seq_ids=seq_ids[test_mask],
    )
    print(json.dumps(report, indent=2))
    _ = rng


if __name__ == "__main__":
    main()
