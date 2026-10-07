"""Rewrite free-form prompts as structured template prompts from their parsed request.

Gives IntentDiT the same structured information that the instruction-conditioned
PosterO baseline receives: the class counts read from the prompt by the rule-based
count reader, plus the grid positions read by the text-spatial parser. The output
uses the wording of the training template families, e.g.
"A layout with 3 Texts with 1 at top-center and 2 at middle-center, 1 Logo at bottom-left."
No annotation is used.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgbdm.text_spatial import parse_positions_from_prompt
from utils.metric import _parse_prompt_counts

ORDER = ("Logo", "Text", "Underlay", "Embellishment")


def element_phrase(name: str, count: int, positions: list[str]) -> str:
    noun = name if count == 1 else f"{name}s"
    if not positions:
        return f"{count} {noun}"
    cells = Counter(positions)
    if len(cells) == 1 and sum(cells.values()) == count:
        return f"{count} {noun} at {positions[0]}"
    parts = [f"{n} at {cell}" for cell, n in cells.items()]
    return f"{count} {noun} with " + " and ".join(parts)


def structured_prompt(text: str) -> str:
    counts = _parse_prompt_counts(text)
    positions = parse_positions_from_prompt(text)
    phrases = [element_phrase(name, counts[name], positions.get(name, [])[: counts[name]])
               for name in ORDER if counts.get(name, 0) > 0]
    return "A layout with " + ", ".join(phrases) + "." if phrases else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.prompts_csv)
    column = "text_prompt" if "text_prompt" in frame.columns else "prompt"
    frame["free_form_prompt"] = frame[column]
    frame[column] = [structured_prompt(str(t)) for t in frame[column]]
    frame.to_csv(args.output, index=False)
    print(f"{args.output}: {len(frame)} prompts")
    for before, after in list(zip(frame["free_form_prompt"], frame[column]))[:3]:
        print(f"  {before}\n  -> {after}")


if __name__ == "__main__":
    main()
