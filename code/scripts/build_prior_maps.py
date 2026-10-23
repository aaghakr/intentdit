"""Write simple spatial priors as drop-in replacements for learned intent maps.

Used to test whether the learned placement-suitability map adds information
beyond simple priors when the generator is retrained with each prior:

* ``train_density``: the mean annotated-layout density over the training split
  (identical for every image; same construction as in analyze_intent_maps.py);
* ``inverse_saliency``: one minus the saliency map used by the evaluator
  (the per-pixel maximum of the two saliency detectors).

Maps are written as 8-bit PNGs under ``<split>/<prior>_map`` with the image's
file name, so a config only needs to point ``intent_map_dir`` at that folder.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyze_intent_maps import normalize_map, rasterize_density

SPLITS = {"train": "train", "val": "val", "test_anno": "test_anno"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--prior", choices=("train_density", "inverse_saliency"), required=True)
    parser.add_argument("--dataset-root", default=str(Path(__file__).resolve().parents[2] / "data" / "dataset"))
    parser.add_argument("--width", type=int, default=513)
    parser.add_argument("--height", type=int, default=750)
    parser.add_argument("--sigma", type=float, default=10.0)
    args = parser.parse_args()

    base = Path(args.dataset_root) / args.dataset / "split"
    constant = None
    if args.prior == "train_density":
        train = pd.read_csv(base / "csv" / "train.csv")
        constant = normalize_map(rasterize_density(train, args.width, args.height, args.sigma, accumulate=True))
        constant = np.rint(constant * 255).astype(np.uint8)

    for split in SPLITS.values():
        out_dir = base / split / f"{args.prior}_map"
        out_dir.mkdir(parents=True, exist_ok=True)
        names = sorted(os.listdir(base / split / "inpaint"))
        for name in names:
            if constant is not None:
                prior = constant
            else:
                sal_a = cv2.imread(str(base / split / "saliency" / name), cv2.IMREAD_GRAYSCALE)
                sal_b = cv2.imread(str(base / split / "saliency_sub" / name), cv2.IMREAD_GRAYSCALE)
                saliency = np.maximum(sal_a, sal_b)
                saliency = cv2.resize(saliency, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
                prior = 255 - saliency
            cv2.imwrite(str(out_dir / name), prior)
        print(f"{out_dir}: {len(names)} maps")


if __name__ == "__main__":
    main()
