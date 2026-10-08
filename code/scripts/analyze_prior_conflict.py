"""Do explicit positions override the learned placement prior?

For every requested (class, cell) pair of a prompt, the nine grid cells of the
image are ranked by the mean of the learned placement-suitability map. A request
is a *conflict* when its cell is among the three least suitable cells of that
image and *agreement* when it is among the three most suitable. Spatial
adherence (the SPLA hit rate, multiset matching per class and cell as in
utils/spatial_pla.py) is then reported separately for the two groups, for any
set of saved prediction files. No model is run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgbdm.text_spatial import parse_positions_from_prompt
from utils.spatial_pla import GRID_COLS, GRID_ROWS, _cell

ROOT = Path(__file__).resolve().parents[2]
MAPS = {"pku": "data/dataset/pku/split/test_anno/intent_map_v2", "cgl": "data/dataset/cgl/split/test_anno/intent_map"}
NAMES = {1: "Text", 2: "Logo", 3: "Underlay", 4: "Embellishment"}
CELLS = [f"{r}-{c}" for r in GRID_ROWS for c in GRID_COLS]


def cell_ranks(path: Path) -> dict[str, int]:
    """Rank 0 = least suitable cell of this image, 8 = most suitable."""
    m = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    h, w = m.shape
    means = {f"{GRID_ROWS[r]}-{GRID_COLS[c]}": m[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3].mean()
             for r in range(3) for c in range(3)}
    order = sorted(CELLS, key=lambda k: means[k])
    return {k: i for i, k in enumerate(order)}


def hits_by_group(pred_path: Path, prompts: dict[str, str], ranks: dict[str, dict[str, int]]):
    payload = torch.load(pred_path, map_location="cpu", weights_only=False)
    out = defaultdict(lambda: [0.0, 0.0])  # group -> [hits, requested]
    for name, rows in zip(payload["img_names"], payload["test_output"]):
        name = os.path.basename(str(name))
        expected = parse_positions_from_prompt(prompts.get(name, ""))
        if not expected or name not in ranks:
            continue
        predicted = defaultdict(Counter)
        for cls, cx, cy, w, h in rows.tolist():
            cls = int(round(cls))
            if cls in NAMES:
                x1, x2 = max(cx - w / 2, 0.0), min(cx + w / 2, 1.0)
                y1, y2 = max(cy - h / 2, 0.0), min(cy + h / 2, 1.0)
                predicted[NAMES[cls]][_cell((x1 + x2) / 2, (y1 + y2) / 2)] += 1
        for cls, cells in expected.items():
            for cell, count in Counter(cells).items():
                rank = ranks[name][cell]
                group = "conflict" if rank <= 2 else "agreement" if rank >= 6 else "neutral"
                hit = min(count, predicted[cls].get(cell, 0))
                for g in (group, "all"):
                    out[g][0] += hit
                    out[g][1] += count
    return {g: {"spla": h / r if r else float("nan"), "requested": r} for g, (h, r) in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--prompts-csv", required=True)
    parser.add_argument("--prediction", nargs=2, action="append", metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.prompts_csv)
    column = "text_prompt" if "text_prompt" in frame.columns else "prompt"
    prompts = {os.path.basename(str(k)): str(v) for k, v in frame.groupby("poster_path")[column].first().items()}
    map_dir = ROOT / MAPS[args.dataset]
    ranks = {n: cell_ranks(map_dir / n) for n in prompts if (map_dir / n).is_file()}
    results = {"dataset": args.dataset, "prompts_csv": args.prompts_csv, "n_images_with_maps": len(ranks), "methods": {}}
    for label, path in args.prediction:
        results["methods"][label] = hits_by_group(Path(path), prompts, ranks)
        r = results["methods"][label]
        print(f"{label:40s} " + "  ".join(f"{g}={r[g]['spla']:.3f} (n={int(r[g]['requested'])})"
                                          for g in ("conflict", "neutral", "agreement", "all") if g in r))
    Path(args.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
