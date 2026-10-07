"""Check that saved predictions are labelled with the images they were generated for.

test.py writes the evaluation image order to a file whose name depends only on the
dataset and inference seed, then reads it back after sampling. Concurrent runs with
different image sets (full test set vs. prompt subsets) can overwrite that file, so
outputs get labelled with the wrong images. The correct order is deterministic:
sorted test images, restricted to the prompt CSV for --prompt-subset-only runs.
This script compares every saved ``*_test_output.pt`` with that order.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SUBSET_STYLES = {
    "freeform": "data/prompts/free_form_{ds}.csv",
    "stress": "data/prompts/stress_{ds}.csv",
    "edit_base": "data/prompts/edit_base_{ds}.csv",
    "edit_changed": "data/prompts/edit_changed_{ds}.csv",
}


def expected_names(name: str) -> tuple[str, list[str]] | None:
    match = re.search(r"cross_(pku|cgl)_to_(pku|cgl)", name)
    if match:
        dataset = match.group(2)
    else:
        match = re.match(r"^(?:ivc|rev)_(?:prompt_|diversity_|edit_|textscale_|efficiency_|oracle_intent_)?(pku|cgl)_", name)
        if not match:
            return None
        dataset = match.group(1)
    images = sorted(os.listdir(ROOT / f"data/dataset/{dataset}/split/test_anno/inpaint"))
    style = None
    if "_diversity_" in f"_{name}" or "freeform" in name:
        style = "freeform"
    elif "stress" in name:
        style = "stress"
    elif re.search(r"edit_(?:(?:pku|cgl)_)?(base|changed)", name):
        style = "edit_" + re.search(r"edit_(?:(?:pku|cgl)_)?(base|changed)", name).group(1)
    if style:
        csv = pd.read_csv(ROOT / SUBSET_STYLES[style].format(ds=dataset))
        allowed = {os.path.basename(p) for p in csv.poster_path}
        images = [n for n in images if n in allowed]
    return dataset, images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default=str(ROOT / "experiments" / "paper_figures"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = {}
    for path in sorted(Path(args.prediction_dir).glob("*_test_output.pt")):
        name = path.name[: -len("_test_output.pt")]
        expected = expected_names(name)
        if expected is None:
            continue
        _, images = expected
        payload = torch.load(path, map_location="cpu")
        saved = [os.path.basename(str(n)) for n in payload["img_names"]]
        n_out = int(payload["test_output"].shape[0])
        ok = saved == images[: len(saved)] and len(saved) == len(images) == n_out
        report[name] = {"ok": ok, "n_saved_names": len(saved), "n_outputs": n_out, "n_expected": len(images)}
    bad = sorted(k for k, v in report.items() if not v["ok"])
    Path(args.output).write_text(json.dumps({"n_checked": len(report), "bad": bad, "detail": report}, indent=1))
    print(f"checked {len(report)} prediction files, {len(bad)} mislabelled")
    for name in bad:
        d = report[name]
        print(f"  BAD {name}: names={d['n_saved_names']} outputs={d['n_outputs']} expected={d['n_expected']}")


if __name__ == "__main__":
    main()
