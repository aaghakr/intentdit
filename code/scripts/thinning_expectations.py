"""Exact expectations of n, Uti and Type-F1 under independent thinning (Proposition 3).

Each element of a designer layout is kept independently with probability q = 1 - p.
With d(x) the number of boxes covering pixel x and s the saliency map,
    E[n]       = q n,
    E[Uti]     = sum_x (1 - s(x)) (1 - p^d(x)) / sum_x (1 - s(x)),
    E[Type-F1] = sum_b Binom(b; n, q) 2b / (n + b).
The script evaluates these per test layout with the evaluator's rasterization
(utils.benchmark_metrics.content_records) and compares the test-set means with
the Monte Carlo thinning results in experiments/paper_results/gt_drop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.benchmark_metrics import decode_layout_tensor
from utils.util import load_config

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent


def expected_type_f1(n: int, q: float) -> float:
    if n == 0:
        return 1.0
    return sum(comb(n, b) * q**b * (1 - q) ** (n - b) * 2 * b / (n + b) for b in range(n + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(str(CODE_ROOT / "configs" / f"{args.dataset}_anno_test.yaml"), path_profile="local")
    drop_dir = ROOT / "experiments/paper_results/gt_drop" / args.dataset
    payload = torch.load(drop_dir / "gt_drop00_seed0.pt", map_location="cpu", weights_only=False)
    classes, _, boxes = decode_layout_tensor(payload["test_output"])
    size = (int(cfg.width), int(cfg.height))
    sums = {p: {"n": 0.0, "uti": 0.0, "f1": 0.0} for p in args.rates}
    uti0 = 0.0
    names = payload["img_names"]
    for i, name in enumerate(names):
        name = os.path.basename(str(name))
        sal = np.maximum(
            np.asarray(Image.open(os.path.join(cfg.paths.test.sal_dir, name)).convert("L").resize(size), dtype=np.float64),
            np.asarray(Image.open(os.path.join(cfg.paths.test.sal_sub_dir, name)).convert("L").resize(size), dtype=np.float64),
        ) / 255.0
        weight = 1.0 - sal
        pix = boxes[i].copy()
        pix[:, [0, 2]] *= size[0]
        pix[:, [1, 3]] *= size[1]
        pix = np.rint(pix).astype(int)
        depth = np.zeros(sal.shape, dtype=np.int32)
        n = 0
        for c, (x1, y1, x2, y2) in zip(classes[i], pix):
            if c <= 0:
                continue
            n += 1
            if x2 > x1 and y2 > y1:
                depth[y1:y2, x1:x2] += 1
        total = weight.sum()
        uti0 += float((weight * (depth > 0)).sum() / total)
        for p in args.rates:
            q = 1.0 - p
            sums[p]["n"] += q * n
            sums[p]["uti"] += float((weight * (1.0 - p ** depth)).sum() / total)
            sums[p]["f1"] += expected_type_f1(n, q)
    count = len(names)
    empirical = json.loads((ROOT / "experiments/paper_results/gt_drop/gt_drop_summary.json").read_text())
    out = {"dataset": args.dataset, "n_layouts": count, "uti_p0": uti0 / count, "rates": {}}
    for p in args.rates:
        pred = {k: v / count for k, v in sums[p].items()}
        out["rates"][str(p)] = {"predicted": pred, "bounds_uti": [(1 - p) * uti0 / count, uti0 / count],
                                "type_f1_jensen_bound": 2 * (1 - p) / (2 - p)}
        print(f"{args.dataset} p={p:.1f}  E[n]={pred['n']:.3f}  E[Uti]={pred['uti']:.4f}  E[F1]={pred['f1']:.4f}")
    out["empirical_summary_keys"] = list(empirical)[:5] if isinstance(empirical, dict) else None
    Path(args.output).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
