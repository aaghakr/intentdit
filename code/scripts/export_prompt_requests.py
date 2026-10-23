"""Export requested element classes per prompt for external baselines.

Reads a prompt CSV (poster_path, text_prompt) and writes a JSON list with the
prompt text and the class multiset read from the prompt by the same rule-based
count reader used in evaluation. External generators (e.g. PosterO) can then be
given the instruction and the requested elements without access to annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metric import _parse_prompt_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.prompts_csv)
    column = "text_prompt" if "text_prompt" in frame.columns else "prompt"
    rows = []
    for _, row in frame.iterrows():
        counts = {name.lower(): int(n) for name, n in _parse_prompt_counts(str(row[column])).items() if n}
        rows.append({"poster_path": os.path.basename(str(row.poster_path)), "prompt": str(row[column]),
                     "requested": counts})
    with open(args.output, "w") as handle:
        json.dump(rows, handle, indent=1)
    empty = sum(1 for r in rows if not r["requested"])
    print(f"{args.output}: {len(rows)} prompts ({empty} with no parsed request)")


if __name__ == "__main__":
    main()
