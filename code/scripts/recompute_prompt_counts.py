"""Re-score prompt-count adherence (PLA, ExactCount) of saved predictions.

Only the count metrics depend on the prompt parser, so this script re-evaluates
them from saved ``*_test_output.pt`` files without re-running the geometry and
content metrics. Used after fixing the digit-count parsing bug in
``utils.metric._parse_prompt_counts``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent

from utils.metric import _prompt_count_records, tla_cal
from utils.util import load_config

FAMILY_CSV = {
    "basic": "test_with_prompts_basic.csv",
    "enhanced": "test_with_prompts_enhanced.csv",
    "advanced": "test_with_prompts_advanced.csv",
    "spatial": "test_with_prompts_spatial.csv",
    "rich": "test_with_rich_prompts.csv",
}
NAME = re.compile(
    r"^ivc_(?:prompt_)?(?P<dataset>pku|cgl)_(?P<variant>.+?)"
    r"(?:_(?P<family>basic|enhanced|advanced|spatial|rich))?"
    r"_trainseed(?P<seed>\d+)_inferseed1_test_output\.pt$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default=str(PROJECT_ROOT / "experiments" / "paper_figures"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(Path(args.prediction_dir).glob("ivc_*_test_output.pt")):
        match = NAME.match(path.name)
        if not match:
            continue
        variant = match["variant"]
        # Unprompted image-only runs have no count metrics; free-form and
        # stress prompts are unaffected by the parser fix and use other CSVs.
        if not any(tag in variant for tag in ("text", "lambda", "pooled")):
            continue
        if "freeform" in variant or "stress" in variant or "cross" in variant or "efficiency" in variant:
            continue
        protocol = "family" if path.name.startswith("ivc_prompt_") else "oracle"
        dataset, family = match["dataset"], match["family"] or "basic"
        cfg = load_config(str(CODE_ROOT / "configs" / f"{dataset}_anno_test.yaml"), path_profile="local")
        cfg.text_control = True
        cfg.paths.test.all_prompts = str(
            PROJECT_ROOT / "data" / "dataset" / dataset / "split" / "csv" / FAMILY_CSV[family]
        )
        payload = torch.load(path, map_location="cpu")
        clses = payload["test_output"][:, :, :1]
        records = _prompt_count_records(list(payload["img_names"]), clses, cfg)
        grouped[f"{protocol}_{dataset}_{variant}_{family}"].append(
            {
                "seed": int(match["seed"]),
                "pla_count": tla_cal(list(payload["img_names"]), clses, cfg),
                "exact_count_match": float(np.mean([r["exact_count_match"] for r in records])),
                "n": len(records),
            }
        )

    summary = {}
    for key, runs in sorted(grouped.items()):
        entry = {"seeds": [r["seed"] for r in runs], "n": runs[0]["n"]}
        for metric in ("pla_count", "exact_count_match"):
            values = [r[metric] for r in runs]
            entry[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
        summary[key] = entry
        print(f"{key:55s} seeds={entry['seeds']} PLA={entry['pla_count']['mean']:.4f}±{entry['pla_count']['std']:.4f} "
              f"Exact={entry['exact_count_match']['mean']:.4f}±{entry['exact_count_match']['std']:.4f}")
    Path(args.output).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
