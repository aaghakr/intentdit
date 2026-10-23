"""Convert RALF's released test outputs (``test_<seed>.pkl``) to the shared prediction format.

RALF (Horita et al., CVPR 2024) uses the same PKU/CGL train/validation/test splits as
this project. Its pickles store, per test image, RALF class ids (alphabetical class
names) and normalized centre/size boxes; this script maps them to our class ids and
writes {'img_names', 'test_output'} with rows (class, cx, cy, w, h).
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RALF_TO_OURS = {
    "pku": {0: 2, 1: 1, 2: 3},  # logo, text, underlay -> 2, 1, 3
    "cgl": {0: 4, 1: 2, 2: 1, 3: 3},  # embellishment, logo, text, underlay -> 4, 2, 1, 3
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--pickle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-elements", type=int, default=16)
    args = parser.parse_args()

    if args.dataset == "cgl":
        csv = pd.read_csv(ROOT / "data/dataset/cgl/split/csv/test.csv")
        id_to_name = {os.path.splitext(f)[0]: p for f, p in zip(csv.file_name, csv.poster_path)}
    results = pickle.load(open(args.pickle, "rb"))["results"]
    names, rows = [], torch.zeros(len(results), args.max_elements, 5)
    for i, r in enumerate(results):
        names.append(id_to_name[r["id"]] if args.dataset == "cgl" else f"{r['id']}.png")
        for j, (label, cx, cy, w, h) in enumerate(zip(r["label"], r["center_x"], r["center_y"], r["width"], r["height"])):
            if j >= args.max_elements:
                break
            rows[i, j] = torch.tensor([RALF_TO_OURS[args.dataset][int(label)], cx, cy, w, h])
    torch.save({"img_names": names, "test_output": rows, "source": str(args.pickle)}, args.output)
    print(f"{args.output}: {len(names)} layouts")


if __name__ == "__main__":
    main()
