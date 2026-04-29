#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from d2pg.data import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build D2PG v0 trajectory-correction dataset.")
    parser.add_argument("--epa-root", type=Path, default=Path("/home/yifu/epa_data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/euroc_corrections.npz"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--max-time-delta", type=float, default=0.03)
    parser.add_argument("--max-est-translation", type=float, default=None)
    parser.add_argument("--max-gt-translation", type=float, default=None)
    args = parser.parse_args()

    x, y, pairs, seq_ids = build_dataset(
        args.epa_root,
        stride=args.stride,
        horizon=args.horizon,
        max_time_delta=args.max_time_delta,
        max_est_translation=args.max_est_translation,
        max_gt_translation=args.max_gt_translation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, seq_ids=seq_ids)
    meta = {
        "num_samples": int(len(x)),
        "input": "local VIO/VO relative pose vector [tx,ty,tz,rx,ry,rz]",
        "target": "matched GT local relative pose vector [tx,ty,tz,rx,ry,rz]",
        "stride": args.stride,
        "horizon": args.horizon,
        "max_time_delta": args.max_time_delta,
        "max_est_translation": args.max_est_translation,
        "max_gt_translation": args.max_gt_translation,
        "sequences": [
            {
                "id": i,
                "dataset": p.dataset,
                "sequence": p.sequence,
                "method": p.method,
                "gt_path": str(p.gt_path),
                "estimate_path": str(p.estimate_path),
            }
            for i, p in enumerate(pairs)
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {args.output} with {len(x)} samples from {len(pairs)} sequence/method pairs")


if __name__ == "__main__":
    main()
