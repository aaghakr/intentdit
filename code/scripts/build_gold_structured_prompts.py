"""Rewrite free-form prompts as structured prompts from the manual references.

Same wording as build_structured_prompts.py, but the element list comes from the
manual reference counts and positions of the free-form audit instead of the
rule-based readers, so every prompt carries its gold element list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_structured_prompts import ORDER, element_phrase


def gold_prompt(counts: dict, positions: dict) -> str:
    phrases = [element_phrase(name, int(counts[name]), list(positions.get(name, []))[: int(counts[name])])
               for name in ORDER if int(counts.get(name, 0)) > 0]
    return "A layout with " + ", ".join(phrases) + "."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-csv", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.prompts_csv)
    audit = pd.read_csv(args.audit_csv).set_index("poster_path")
    frame["free_form_prompt"] = frame["text_prompt"]
    frame["text_prompt"] = [gold_prompt(json.loads(audit.loc[p, "manual_counts_json"]),
                                        json.loads(audit.loc[p, "manual_positions_json"]))
                            for p in frame["poster_path"]]
    frame.to_csv(args.output, index=False)
    print(f"{args.output}: {len(frame)} prompts")
    for before, after in list(zip(frame["free_form_prompt"], frame["text_prompt"]))[:2]:
        print(f"  {before}\n  -> {after}")


if __name__ == "__main__":
    main()
