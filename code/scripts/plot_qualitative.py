"""Qualitative figures: image-only comparison and prompt-controlled layouts.

Examples are drawn at random with a fixed seed (no hand-picking) from the PKU and
CGL test sets (image-only) and from the free-form prompt sets (prompt control).
All methods are rendered with one class colour scheme on the same background.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "experiments" / "paper_figures"
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#4a3aa7"}  # text, logo, underlay, embellishment
NAMES = {1: "Text", 2: "Logo", 3: "Underlay", 4: "Embellishment"}
INK, INK_2 = "#0b0b0b", "#52514e"


def load(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {os.path.basename(str(n)): payload["test_output"][i] for i, n in enumerate(payload["img_names"])}


def ground_truth(dataset: str) -> dict[str, list[tuple[int, list[float]]]]:
    frame = pd.read_csv(ROOT / f"data/dataset/{dataset}/split/csv/test.csv")
    out: dict[str, list] = {}
    for _, row in frame.iterrows():
        out.setdefault(row.poster_path, []).append((int(row.cls_elem), json.loads(str(row.box_elem).replace("(", "[").replace(")", "]"))))
    return out


def render(dataset: str, name: str, layout, size=(256, 375), from_gt=False) -> Image.Image:
    image = Image.open(ROOT / f"data/dataset/{dataset}/split/test_anno/inpaint/{name}").convert("RGB")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    boxes = []
    if from_gt:
        for cls, (x1, y1, x2, y2) in layout:
            sx, sy = width / 513.0, height / 750.0
            boxes.append((cls, (x1 * sx, y1 * sy, x2 * sx, y2 * sy)))
    else:
        for row in layout.tolist():
            cls = int(round(row[0]))
            if cls <= 0:
                continue
            cx, cy, w, h = row[1:]
            x1, y1 = max(0, cx - w / 2) * width, max(0, cy - h / 2) * height
            x2, y2 = min(1, cx + w / 2) * width, min(1, cy + h / 2) * height
            boxes.append((cls, (x1, y1, x2, y2)))
    stroke = max(2, int(round(width / 110)))
    for cls, box in sorted(boxes, key=lambda b: b[0] != 3):  # underlays first, underneath
        rgb = tuple(int(COLORS.get(cls, "#888888")[i : i + 2], 16) for i in (1, 3, 5))
        draw.rectangle(box, fill=rgb + (80,), outline=rgb + (255,), width=stroke)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").resize(size, Image.LANCZOS)


def grid(rows, row_labels, col_labels, output: Path) -> None:
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.18 * n_cols, 1.72 * n_rows + 0.45))
    plt.rcParams.update({"font.family": "serif"})
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.imshow(rows[r][c])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#c9c8c3"); spine.set_linewidth(0.5)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=7, color=INK, pad=3)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=6.5, color=INK, rotation=90, labelpad=3)
    handles = [Patch(facecolor=COLORS[k] + "55", edgecolor=COLORS[k], label=NAMES[k]) for k in (1, 2, 3, 4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.95, bottom=0.06, wspace=0.04, hspace=0.04)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per-dataset", type=int, default=3)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)
    record = {"seed": args.seed}

    # ---------------- image-only comparison (sampler S_L, training seed 1)
    columns, rows = [], [[], [], [], []]
    for dataset in ("pku", "cgl"):
        gt = ground_truth(dataset)
        released = load(ROOT / f"other_baselines/layoutidit/{dataset}_anno_uncond_test_output.pt")
        config = load(FIGS / f"rev_{dataset}_vit_saliency_linear_trainseed1_inferseed1_test_output.pt")
        ours = load(FIGS / f"rev_{dataset}_vit_both_linear_trainseed1_inferseed1_test_output.pt")
        names = sorted(set(gt) & set(released) & set(config) & set(ours))
        chosen = list(rng.choice(names, size=args.per_dataset, replace=False))
        record[f"image_only_{dataset}"] = chosen
        for name in chosen:
            columns.append(f"{dataset.upper()} {name.split('.')[0]}")
            rows[0].append(render(dataset, name, gt[name], from_gt=True))
            rows[1].append(render(dataset, name, released[name]))
            rows[2].append(render(dataset, name, config[name]))
            rows[3].append(render(dataset, name, ours[name]))
    grid(rows, ["Designer layout", "LayoutDiT (released)", "LayoutDiT config.", "IntentDiT"], columns,
         out_dir / "fig05_qualitative_image_only.pdf")

    # ---------------- prompt-controlled layouts (sampler S_C, training seed 1)
    columns, rows, prompts = [], [[], [], []], []
    for dataset in ("pku", "cgl"):
        free = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}.csv").set_index("poster_path").text_prompt
        structured = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}_structured.csv").set_index("poster_path").text_prompt
        raw = load(FIGS / f"rev_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_test_output.pt")
        listed = load(FIGS / f"rev_prompt_{dataset}_vit_both_text_structured_trainseed1_inferseed1_test_output.pt")
        postero = load(ROOT / f"other_baselines/standardized/postero_prompted_{dataset}_freeform_subset.pt")
        names = sorted(set(raw) & set(listed) & set(postero))
        chosen = list(rng.choice(names, size=args.per_dataset, replace=False))
        for name in chosen:
            label = f"({chr(ord('a') + len(prompts))}) {dataset.upper()}"
            columns.append(label)
            prompts.append({"label": label, "image": name, "prompt": free[name], "element_list": structured[name]})
            rows[0].append(render(dataset, name, raw[name]))
            rows[1].append(render(dataset, name, listed[name]))
            rows[2].append(render(dataset, name, postero[name]))
    grid(rows, ["IntentDiT, instruction", "IntentDiT, + list", "PosterO, + list"], columns,
         out_dir / "fig03_prompt_control.pdf")
    record["prompt_examples"] = prompts
    (out_dir / "qualitative_selection.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
