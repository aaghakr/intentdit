"""Build ground-truth layouts with elements removed at random (controlled density intervention).

Each annotated test layout is loaded, every element is kept independently with
probability ``1 - drop_rate``, and the result is saved in the shared prediction
format ({'img_names', 'test_output'}, class + normalized cxcywh) so it can be
scored by ``evaluate_saved_predictions.py`` exactly like a model output.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_ROOT = Path(__file__).resolve().parents[1]

from utils.benchmark_metrics import load_ground_truth_layouts
from utils.util import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--drop-rates", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--path-profile", choices=("local", "server"), default="local")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(
        str(CODE_ROOT / "configs" / f"{args.dataset}_anno_test.yaml"),
        path_profile=args.path_profile,
    )
    frame = pd.read_csv(cfg.paths.test.annotated_dir)
    img_names = sorted(frame.poster_path.unique().tolist())
    layouts = load_ground_truth_layouts(img_names, cfg)
    max_elem = int(cfg.max_elem)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rate in args.drop_rates:
        for seed in args.seeds if rate > 0 else [0]:
            rng = np.random.default_rng(seed)
            output = torch.zeros(len(img_names), max_elem, 5)
            for i, layout in enumerate(layouts):
                keep = rng.random(len(layout["classes"])) >= rate
                classes = layout["classes"][keep]
                boxes = layout["boxes"][keep]
                n = len(classes)
                output[i, :n, 0] = torch.as_tensor(classes, dtype=torch.float32)
                output[i, :n, 1] = torch.as_tensor((boxes[:, 0] + boxes[:, 2]) / 2)
                output[i, :n, 2] = torch.as_tensor((boxes[:, 1] + boxes[:, 3]) / 2)
                output[i, :n, 3] = torch.as_tensor(boxes[:, 2] - boxes[:, 0])
                output[i, :n, 4] = torch.as_tensor(boxes[:, 3] - boxes[:, 1])
            name = f"gt_drop{int(round(rate * 100)):02d}_seed{seed}.pt"
            torch.save({"img_names": img_names, "test_output": output}, out_dir / name)
            print(out_dir / name, float((output[..., 0] > 0).sum(1).float().mean()))


if __name__ == "__main__":
    main()
