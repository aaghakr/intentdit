"""Qualitative figures: image-only comparison and prompt-controlled layouts.

Examples are either drawn at random with a fixed seed (--selection random, no
hand-picking) or taken from a curated list with its selection criterion
(--selection curated, configs/qualitative_examples.json). All methods are rendered
with one class colour scheme on the same background.
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
# Solid class colours (text, logo, underlay, embellishment), as in common poster-layout figures.
COLORS = {1: "#63b748", 2: "#ed9ba7", 3: "#78bbe6", 4: "#f2c14e"}
OUTLINE_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#4a3aa7"}
NAMES = {1: "Text", 2: "Logo", 3: "Underlay", 4: "Embellishment"}
# Translucent fills with a darker edge of the same hue (LayoutDiT-style figures): the image stays visible.
EDGE_COLORS = {1: "#3d8a2b", 2: "#c8667a", 3: "#3b8bc4", 4: "#c08f1c"}
FILL_ALPHA = 0.80
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


def render(dataset: str, name: str, layout, size=(256, 375), from_gt=False, style="solid") -> Image.Image:
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
    ordered = sorted(boxes, key=lambda b: b[0] != 3)  # underlays first, underneath
    if style == "translucent":
        # Each box is composited on its own so that overlapping boxes blend instead of hiding each other.
        out = image.convert("RGBA")
        edge = max(2, int(round(width / 256)))
        for cls, box in ordered:
            layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            fill = tuple(int(COLORS.get(cls, "#888888")[i : i + 2], 16) for i in (1, 3, 5))
            line = tuple(int(EDGE_COLORS.get(cls, "#555555")[i : i + 2], 16) for i in (1, 3, 5))
            ImageDraw.Draw(layer).rectangle(box, fill=fill + (int(255 * FILL_ALPHA),), outline=line + (255,), width=edge)
            out = Image.alpha_composite(out, layer)
        out = out.convert("RGB")
        return out if size is None else out.resize(size, Image.LANCZOS)
    palette = COLORS if style == "solid" else OUTLINE_COLORS
    for cls, box in ordered:
        rgb = tuple(int(palette.get(cls, "#888888")[i : i + 2], 16) for i in (1, 3, 5))
        if style == "solid":
            draw.rectangle(box, fill=rgb + (255,))
        else:
            draw.rectangle(box, fill=rgb + (80,), outline=rgb + (255,), width=stroke)
    out = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return out if size is None else out.resize(size, Image.LANCZOS)


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
    handles = [Patch(facecolor=COLORS[k], edgecolor="none", label=NAMES[k]) for k in (1, 2, 3, 4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.95, bottom=0.06, wspace=0.04, hspace=0.04)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)


def strip(panels, col_labels, output: Path, tile=(1.02, 1.49)) -> None:
    """LayoutDiT-style strip: one row per image, one column per method, datasets side by side."""
    n_rows = max(len(rows) for _, rows in panels)
    n_cols = len(col_labels)
    gap = 0.35  # empty column between the dataset panels
    width_ratios = sum(([1] * n_cols + [gap] for _ in panels), [])[:-1]
    fig = plt.figure(figsize=(tile[0] * (n_cols * len(panels) + gap * (len(panels) - 1)), tile[1] * n_rows + 0.75))
    plt.rcParams.update({"font.family": "serif"})
    grid_spec = fig.add_gridspec(n_rows, len(width_ratios), width_ratios=width_ratios, wspace=0.04, hspace=0.04,
                                 left=0.005, right=0.995, top=0.90, bottom=0.13)
    for p, (dataset_label, rows) in enumerate(panels):
        offset = p * (n_cols + 1)
        for r, row in enumerate(rows):
            for c, tile_image in enumerate(row):
                ax = fig.add_subplot(grid_spec[r, offset + c])
                ax.imshow(tile_image)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#c9c8c3"); spine.set_linewidth(0.5)
                if r == 0:
                    bold = col_labels[c] == "IntentDiT"
                    ax.set_title(col_labels[c], fontsize=10.5, color=INK, pad=3, linespacing=1.05, fontweight="bold" if bold else "normal")
        left = grid_spec[0, offset].get_position(fig).x0
        right = grid_spec[0, offset + n_cols - 1].get_position(fig).x1
        fig.text((left + right) / 2, 0.075, dataset_label, ha="center", fontsize=12, color=INK, fontweight="bold")
    handles = [Patch(facecolor=COLORS[k], edgecolor="none", label=NAMES[k]) for k in (1, 2, 3, 4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.005))
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)


# Print-ready layout: drawn at the final MDPI full width (7.27 in = \\fulllength) with
# sans-serif labels, so nothing is rescaled when the PDF is placed in the paper.
PAGE_WIDTH = 7.27
FRAME, HIGHLIGHT, RULE = "#bdbcb7", "#1d1d1b", "#8c8b86"


def panel_figure(tiles, col_labels, output: Path, row_labels=None, groups=None, highlight=None,
                 classes=(1, 2, 3, 4), width=PAGE_WIDTH) -> None:
    """Grid of image tiles (rows x columns) with column headers, optional row labels and column groups.

    groups: list of (label, first_col, last_col); a wider gap separates consecutive groups.
    """
    plt.rcParams.update({"font.family": "Nimbus Sans", "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold", "pdf.fonttype": 42})
    n_rows, n_cols = len(tiles), len(tiles[0])
    left = 0.30 if row_labels else 0.04
    right, gx, gy, group_gap = 0.04, 0.05, 0.05, 0.18
    n_gaps = len(groups) - 1 if groups else 0
    tile_w = (width - left - right - gx * (n_cols - 1) - (group_gap - gx) * n_gaps) / n_cols
    tile_h = tile_w * tiles[0][0].height / tiles[0][0].width
    header = 0.36 if any("\n" in label for label in col_labels) else 0.22
    group_header = 0.24 if groups and any(label for label, _, _ in groups) else 0.0
    legend = 0.32
    height = group_header + header + n_rows * tile_h + (n_rows - 1) * gy + legend
    fig = plt.figure(figsize=(width, height))

    def col_x(c):
        extra = 0.0
        if groups:
            extra = sum(group_gap - gx for _, first, _ in groups[1:] if c >= first)
        return left + c * (tile_w + gx) + extra

    top = height - group_header - header
    for r, row in enumerate(tiles):
        y = top - (r + 1) * tile_h - r * gy
        for c, tile_image in enumerate(row):
            ax = fig.add_axes([col_x(c) / width, y / height, tile_w / width, tile_h / height])
            ax.imshow(tile_image)
            ax.set_xticks([]); ax.set_yticks([])
            strong = highlight is not None and c == highlight
            for spine in ax.spines.values():
                spine.set_color(HIGHLIGHT if strong else FRAME)
                spine.set_linewidth(1.3 if strong else 0.6)
        if row_labels:
            fig.text((left - 0.09) / width, (y + tile_h / 2) / height, row_labels[r], rotation=90,
                     ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.1)
    for c, label in enumerate(col_labels):
        strong = highlight is not None and c == highlight
        fig.text((col_x(c) + tile_w / 2) / width, (top + 0.05) / height, label, ha="center", va="bottom",
                 fontsize=8, color=INK, fontweight="bold" if strong else "normal", linespacing=1.1)
    for label, first, last in groups or []:
        if not label:  # unlabeled group: only the wider gap separates it from its neighbours
            continue
        x0, x1 = col_x(first), col_x(last) + tile_w
        y_rule = top + header - 0.02
        fig.add_artist(plt.Line2D([x0 / width, x1 / width], [y_rule / height] * 2, color=RULE, lw=0.7))
        fig.text((x0 + x1) / 2 / width, (y_rule + 0.05) / height, label, ha="center", va="bottom",
                 fontsize=8.5, color=INK, fontweight="bold")
    handles = [Patch(facecolor=COLORS[k], edgecolor="none", label=NAMES[k]) for k in classes]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7.5,
               handlelength=1.6, handleheight=0.9, columnspacing=1.8, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)


FONT = {"font.family": "Nimbus Sans", "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold", "pdf.fonttype": 42}
GUTTER = 0.028  # inches between tiles (about 2 pt), as in LayoutDiT's comparison figures
TEXT_WIDTH = 394.35 / 72  # MDPI text-column width; every qualitative grid is drawn at this width


def legend_row(fig, width, y, classes, right=0.0):
    """Right-aligned class legend: swatch with the box style, then the label."""
    from matplotlib.patches import Rectangle
    sw, sh, pad, gap = 0.20, 0.085, 0.05, 0.16
    labels = [NAMES[k] for k in classes]
    # approximate label widths for Nimbus Sans 7 pt
    widths = [0.052 * len(t) + 0.02 for t in labels]
    total = sum(sw + pad + w for w in widths) + gap * (len(classes) - 1)
    x = width - right - total
    for k, w in zip(classes, widths):
        fig.add_artist(Rectangle((x / width, y / fig.get_figheight()), sw / width, sh / fig.get_figheight(),
                                 transform=fig.transFigure, facecolor=COLORS[k], alpha=FILL_ALPHA, edgecolor="none"))
        fig.add_artist(Rectangle((x / width, y / fig.get_figheight()), sw / width, sh / fig.get_figheight(),
                                 transform=fig.transFigure, facecolor="none", edgecolor=EDGE_COLORS[k], lw=0.6))
        fig.text((x + sw + pad) / width, (y + sh / 2) / fig.get_figheight(), NAMES[k], ha="left", va="center",
                 fontsize=7, color=INK)
        x += sw + pad + w + gap


def comparison_grid(tiles, headers, output: Path, bold=(), sep_after=None, row_ids=None, classes=(1, 2, 3),
                    text_col=None, text_width=0.0, width=PAGE_WIDTH) -> None:
    """LayoutDiT-style comparison grid: rows are images, columns are methods.

    tiles: rows of PIL images (None where the text column goes); headers: one single-line label per column;
    bold: indices of columns whose header is bold; sep_after: column after which a wider gap is left;
    row_ids: short labels written vertically left of each row; text_col/text_width: index and width (in) of a
    column holding text (one string per row, a list of (text, style) runs).
    """
    plt.rcParams.update(FONT)
    n_rows, n_cols = len(tiles), len(headers)
    left = 0.17 if row_ids else 0.0
    sep = 0.10 if sep_after is not None else 0.0
    n_img = n_cols - (1 if text_col is not None else 0)
    tile_w = (width - left - text_width - sep - GUTTER * (n_cols - 1)) / n_img
    sample = next(t for t in tiles[0] if t is not None)
    tile_h = tile_w * sample.height / sample.width
    head, foot = 0.20, 0.24
    height = head + n_rows * tile_h + (n_rows - 1) * GUTTER + foot
    fig = plt.figure(figsize=(width, height))
    xs, x = [], left
    for c in range(n_cols):
        xs.append(x)
        x += (text_width if c == text_col else tile_w) + GUTTER + (sep if c == sep_after else 0.0)
    top = height - head
    for c, label in enumerate(headers):
        cw = text_width if c == text_col else tile_w
        fig.text((xs[c] + cw / 2) / width, (top + 0.055) / height, label, ha="center", va="bottom", fontsize=7,
                 color=INK, fontweight="bold" if c in bold else "normal")
    for r, row in enumerate(tiles):
        y = top - (r + 1) * tile_h - r * GUTTER
        for c in range(n_cols):
            if c == text_col:
                ax = fig.add_axes([xs[c] / width, y / height, text_width / width, tile_h / height])
                ax.axis("off")
                # runs: (text, style) lines, style in {"label", "body", "note", ""}; block centred vertically
                runs = row[c]
                size, lead = 6.6, 1.3
                step = size * lead / 72 / tile_h
                y0 = 0.5 + step * (len(runs) - 1) / 2
                for k, (t, kind) in enumerate(runs):
                    ax.text(0.03, y0 - k * step, t, transform=ax.transAxes, ha="left", va="center", fontsize=size,
                            color=INK_2 if kind == "note" else INK, fontweight="bold" if kind == "label" else "normal",
                            fontstyle="normal")
                continue
            ax = fig.add_axes([xs[c] / width, y / height, tile_w / width, tile_h / height])
            ax.imshow(row[c], interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():  # hairline frame so white posters do not melt into the page
                spine.set_color("#d4d3cf"); spine.set_linewidth(0.35)
        if row_ids:
            fig.text((left - 0.075) / width, (y + tile_h / 2) / height, row_ids[r], rotation=90, ha="center",
                     va="center", fontsize=7, color=INK_2)
    legend_row(fig, width, 0.07, classes)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)


def wrap(text: str, width_in: float, size_pt: float = 6.6) -> list[str]:
    """Greedy word wrap by the measured width of Nimbus Sans at size_pt, so no line leaves its column."""
    from PIL import ImageFont
    font = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf", 100)
    limit = width_in * 72 / size_pt * 100  # available width in font units at size 100
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and font.getlength(trial) > limit:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def classes_shown(layouts) -> tuple:
    """Element classes that occur in the drawn layouts (designer lists of (cls, box) or tensors)."""
    found = set()
    for layout in layouts:
        if isinstance(layout, list):
            found.update(int(c) for c, _ in layout)
        else:
            found.update(int(round(r[0])) for r in layout.tolist() if int(round(r[0])) > 0)
    return tuple(sorted(c for c in found if c in NAMES))


def paper_figures_v2(args, pick, out_dir: Path) -> dict:
    """LayoutDiT-style figures: image-only and prompt-control comparisons, one figure per dataset."""
    tile = (360, 526)  # 513 x 750 poster aspect; about 300 dpi at the printed tile width
    style = "translucent"
    record = {}
    headers = ["Input", "Designer", "LayoutDiT", "LayoutDiT config.", "RALF", "IntentDiT"]
    for dataset in ("pku", "cgl"):
        gt = ground_truth(dataset)
        released = load(ROOT / f"other_baselines/layoutidit/{dataset}_anno_uncond_test_output.pt")
        config = load(FIGS / f"rev_{dataset}_vit_saliency_linear_trainseed1_inferseed1_test_output.pt")
        ralf = load(ROOT / f"other_baselines/ralf/ralf_{dataset}_test_seed0.pt")
        ours = load(FIGS / f"rev_{dataset}_vit_both_linear_trainseed1_inferseed1_test_output.pt")
        names = sorted(set(gt) & set(released) & set(config) & set(ralf) & set(ours))
        chosen = pick(names, "image_only", dataset)
        record[f"image_only_{dataset}"] = chosen
        tiles = [[render(dataset, n, [], size=tile, from_gt=True, style=style),
                  render(dataset, n, gt[n], size=tile, from_gt=True, style=style),
                  render(dataset, n, released[n], size=tile, style=style),
                  render(dataset, n, config[n], size=tile, style=style),
                  render(dataset, n, ralf[n], size=tile, style=style),
                  render(dataset, n, ours[n], size=tile, style=style)] for n in chosen]
        shown = classes_shown([gt[n] for n in chosen] + [src[n] for src in (released, config, ralf, ours) for n in chosen])
        comparison_grid(tiles, headers, out_dir / f"qual_image_only_{dataset}{args.suffix}.pdf", bold=(5,),
                        sep_after=1, row_ids=[f"#{n.split('.')[0]}" for n in chosen],
                        classes=shown, width=TEXT_WIDTH)

    from qual_prompt_figure import draw_grid, element_list_line, prompt_pair_figure
    # Same printed tile size as the image-only grids above (full width, six columns, row IDs, one wider gap).
    tile_w = (TEXT_WIDTH - 0.17 - 0.10 - GUTTER * 5) / 6
    prompts, letter, columns, drawn = [], 0, [], []
    for dataset in ("pku", "cgl"):
        free = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}.csv").set_index("poster_path").text_prompt
        structured = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}_structured.csv").set_index("poster_path").text_prompt
        raw = load(FIGS / f"rev_prompt_{dataset}_vit_{args.text_model}_freeform_trainseed1_inferseed1_test_output.pt")
        listed = load(FIGS / f"rev_prompt_{dataset}_vit_{args.text_model}_structured_trainseed1_inferseed1_test_output.pt")
        postero = load(ROOT / f"other_baselines/standardized/postero_prompted_{dataset}_freeform_subset.pt")
        names = sorted(set(raw) & set(listed) & set(postero))
        examples = []
        for name in pick(names, "prompt_control", dataset):
            label = f"({chr(ord('a') + letter)})"
            letter += 1
            prompts.append({"label": f"{label} {dataset.upper()}", "image": name, "prompt": free[name],
                            "element_list": structured[name]})
            tiles = [draw_grid(render(dataset, name, layouts[name], size=None, style=style)).resize(tile, Image.LANCZOS)
                     for layouts in (raw, listed, postero)]
            drawn += [layouts[name] for layouts in (raw, listed, postero)]
            examples.append({"label": label, "prompt": free[name], "element_list": element_list_line(structured[name]),
                             "tiles": tiles})
        columns.append(({"pku": "PKU PosterLayout", "cgl": "CGL"}[dataset], examples))
    prompt_pair_figure(columns, out_dir / f"qual_prompt_control{args.suffix}.pdf", COLORS, EDGE_COLORS,
                       tile_w=tile_w, gutter=GUTTER, width=TEXT_WIDTH, classes=classes_shown(drawn))
    record["prompt_examples"] = prompts
    return record


def paper_figures(args, pick, out_dir: Path) -> dict:
    """Image-only comparison as one figure per dataset, then the prompt-control figure."""
    tile = (340, 497)  # 513 x 750 poster aspect; about 300 dpi at the printed tile width
    labels = ["Input", "Designer\nlayout", "LayoutDiT\n(released)", "LayoutDiT\nconfig.", "RALF\n(released)", "IntentDiT"]
    record = {}
    for dataset in ("pku", "cgl"):
        gt = ground_truth(dataset)
        released = load(ROOT / f"other_baselines/layoutidit/{dataset}_anno_uncond_test_output.pt")
        config = load(FIGS / f"rev_{dataset}_vit_saliency_linear_trainseed1_inferseed1_test_output.pt")
        ralf = load(ROOT / f"other_baselines/ralf/ralf_{dataset}_test_seed0.pt")
        ours = load(FIGS / f"rev_{dataset}_vit_both_linear_trainseed1_inferseed1_test_output.pt")
        names = sorted(set(gt) & set(released) & set(config) & set(ralf) & set(ours))
        chosen = pick(names, "image_only", dataset)
        record[f"image_only_{dataset}"] = chosen
        tiles = [[render(dataset, n, [], size=tile, from_gt=True, style=args.style),
                  render(dataset, n, gt[n], size=tile, from_gt=True, style=args.style),
                  render(dataset, n, released[n], size=tile, style=args.style),
                  render(dataset, n, config[n], size=tile, style=args.style),
                  render(dataset, n, ralf[n], size=tile, style=args.style),
                  render(dataset, n, ours[n], size=tile, style=args.style)] for n in chosen]
        # A wider gap separates the two reference columns (input, designer layout) from the generated layouts.
        panel_figure(tiles, labels, out_dir / f"qual_image_only_{dataset}{args.suffix}.pdf",
                     row_labels=[f"Image {n.split('.')[0]}" for n in chosen], highlight=5,
                     groups=[("", 0, 1), ("", 2, 5)],
                     classes=(1, 2, 3) if dataset == "pku" else (1, 2, 3, 4))

    rows, prompts, groups = [[], [], []], [], []
    for dataset in ("pku", "cgl"):
        free = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}.csv").set_index("poster_path").text_prompt
        structured = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}_structured.csv").set_index("poster_path").text_prompt
        raw = load(FIGS / f"rev_prompt_{dataset}_vit_{args.text_model}_freeform_trainseed1_inferseed1_test_output.pt")
        listed = load(FIGS / f"rev_prompt_{dataset}_vit_{args.text_model}_structured_trainseed1_inferseed1_test_output.pt")
        postero = load(ROOT / f"other_baselines/standardized/postero_prompted_{dataset}_freeform_subset.pt")
        names = sorted(set(raw) & set(listed) & set(postero))
        chosen = pick(names, "prompt_control", dataset)
        first = len(prompts)
        for name in chosen:
            label = f"({chr(ord('a') + len(prompts))})"
            prompts.append({"label": f"{label} {dataset.upper()}", "image": name, "prompt": free[name],
                            "element_list": structured[name]})
            rows[0].append(render(dataset, name, raw[name], size=tile, style=args.style))
            rows[1].append(render(dataset, name, listed[name], size=tile, style=args.style))
            rows[2].append(render(dataset, name, postero[name], size=tile, style=args.style))
        groups.append(({"pku": "PKU PosterLayout", "cgl": "CGL"}[dataset], first, len(prompts) - 1))
    panel_figure(rows, [p["label"].split()[0] for p in prompts], out_dir / f"qual_prompt_control{args.suffix}.pdf",
                 row_labels=["IntentDiT\ninstruction", "IntentDiT\n+ list", "PosterO\n+ list"], groups=groups)
    record["prompt_examples"] = prompts
    return record


def export_tiles(args, pick, out_dir: Path) -> dict:
    """Every tile of the qualitative figures as a separate PNG at the image's native resolution.

    Folders are named <rank>_<image id>; a "_in_paper" suffix marks the examples used in the
    paper's main figures (configs/qualitative_examples.json).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {}
    in_paper = json.loads((ROOT / "code/configs/qualitative_examples.json").read_text())

    def folder_for(kind, dataset, rank, name):
        stem = name.split(".")[0]
        tag = "_in_paper" if name in in_paper[kind][dataset] else ""
        return out_dir / kind / dataset / f"{rank:02d}_{stem}{tag}"
    methods = [("01_input", None), ("02_ground_truth", "gt"), ("03_layoutdit_released", "released"),
               ("04_layoutdit_config", "config"), ("05_ralf_released", "ralf"), ("06_intentdit", "ours")]
    for dataset in ("pku", "cgl"):
        gt = ground_truth(dataset)
        sources = {
            "released": load(ROOT / f"other_baselines/layoutidit/{dataset}_anno_uncond_test_output.pt"),
            "config": load(FIGS / f"rev_{dataset}_vit_saliency_linear_trainseed1_inferseed1_test_output.pt"),
            "ralf": load(ROOT / f"other_baselines/ralf/ralf_{dataset}_test_seed0.pt"),
            "ours": load(FIGS / f"rev_{dataset}_vit_both_linear_trainseed1_inferseed1_test_output.pt"),
        }
        names = sorted(set(gt).intersection(*sources.values()))
        chosen = pick(names, "image_only", dataset)
        record[f"image_only_{dataset}"] = chosen
        for rank, name in enumerate(chosen, 1):
            folder = folder_for("image_only", dataset, rank, name)
            folder.mkdir(parents=True, exist_ok=True)
            for file_name, key in methods:
                if key is None:
                    tile = render(dataset, name, [], size=None, from_gt=True, style=args.style)
                elif key == "gt":
                    tile = render(dataset, name, gt[name], size=None, from_gt=True, style=args.style)
                else:
                    tile = render(dataset, name, sources[key][name], size=None, style=args.style)
                tile.save(folder / f"{file_name}.png")

    record["prompt_examples"] = []
    for dataset in ("pku", "cgl"):
        free = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}.csv").set_index("poster_path").text_prompt
        structured = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}_structured.csv").set_index("poster_path").text_prompt
        sources = {
            "02_intentdit_instruction": load(FIGS / f"rev_prompt_{dataset}_vit_{args.text_model}_freeform_trainseed1_inferseed1_test_output.pt"),
            "03_intentdit_with_list": load(FIGS / f"rev_prompt_{dataset}_vit_{args.text_model}_structured_trainseed1_inferseed1_test_output.pt"),
            "04_postero_with_list": load(ROOT / f"other_baselines/standardized/postero_prompted_{dataset}_freeform_subset.pt"),
        }
        names = sorted(set.intersection(*(set(v) for v in sources.values())))
        for rank, name in enumerate(pick(names, "prompt_control", dataset), 1):
            folder = folder_for("prompt_control", dataset, rank, name)
            folder.mkdir(parents=True, exist_ok=True)
            render(dataset, name, [], size=None, from_gt=True, style=args.style).save(folder / "01_input.png")
            for file_name, layouts in sources.items():
                render(dataset, name, layouts[name], size=None, style=args.style).save(folder / f"{file_name}.png")
            (folder / "prompt.txt").write_text(f"Free-form prompt:\n{free[name]}\n\nElement list given to "
                                              f"IntentDiT + list and PosterO + list:\n{structured[name]}\n")
            record["prompt_examples"].append({"dataset": dataset, "image": name})

    # Class legend swatches.
    legend = out_dir / "legend"
    legend.mkdir(exist_ok=True)
    for k, label in NAMES.items():
        Image.new("RGB", (120, 60), COLORS[k]).save(legend / f"{k}_{label.lower()}_{COLORS[k][1:]}.png")
    return record


def image_only_strip(args, pick, out_dir: Path) -> dict:
    labels = ["Input", "Designer\nlayout", "LayoutDiT\n(released)", "LayoutDiT\nconfig.", "RALF\n(released)", "IntentDiT"]
    panels, record = [], {}
    for dataset in ("pku", "cgl"):
        gt = ground_truth(dataset)
        released = load(ROOT / f"other_baselines/layoutidit/{dataset}_anno_uncond_test_output.pt")
        config = load(FIGS / f"rev_{dataset}_vit_saliency_linear_trainseed1_inferseed1_test_output.pt")
        ralf = load(ROOT / f"other_baselines/ralf/ralf_{dataset}_test_seed0.pt")
        ours = load(FIGS / f"rev_{dataset}_vit_both_linear_trainseed1_inferseed1_test_output.pt")
        names = sorted(set(gt) & set(released) & set(config) & set(ralf) & set(ours))
        chosen = pick(names, "image_only", dataset)
        record[f"image_only_{dataset}"] = chosen
        rows = []
        for name in chosen:
            rows.append([render(dataset, name, [], from_gt=True, style=args.style),
                         render(dataset, name, gt[name], from_gt=True, style=args.style),
                         render(dataset, name, released[name], style=args.style),
                         render(dataset, name, config[name], style=args.style),
                         render(dataset, name, ralf[name], style=args.style),
                         render(dataset, name, ours[name], style=args.style)])
        panels.append(({"pku": "PKU PosterLayout", "cgl": "CGL"}[dataset], rows))
    strip(panels, labels, out_dir / f"fig05_qualitative_image_only_strip{args.suffix}.pdf")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per-dataset", type=int, default=3)
    parser.add_argument("--selection", choices=("random", "curated"), default="random")
    parser.add_argument("--examples-json", default=str(ROOT / "code/configs/qualitative_examples.json"))
    parser.add_argument("--suffix", default="", help="Appended to the output file names")
    parser.add_argument("--style", choices=("solid", "outline"), default="solid")
    parser.add_argument("--layout", choices=("paper", "paper_v1", "tiles", "grid", "strip"), default="paper",
                        help="paper: print-ready figures (image-only per dataset + prompt control); "
                             "grid: methods as rows; strip: image-only strip with both datasets side by side")
    parser.add_argument("--text-model", default="both_text_gated_scaled",
                        help="Slug of the text-conditioned IntentDiT runs shown in the prompt-control figure")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)
    record = {"seed": args.seed, "selection": args.selection}
    curated = json.loads(Path(args.examples_json).read_text()) if args.selection == "curated" else None

    def pick(names, kind, dataset):
        if curated is not None:
            missing = [n for n in curated[kind][dataset] if n not in names]
            if missing:
                raise ValueError(f"{kind}/{dataset}: {missing} not available")
            return list(curated[kind][dataset])
        return list(rng.choice(names, size=args.per_dataset, replace=False))

    def draw(dataset, name, layout, from_gt=False):
        return render(dataset, name, layout, from_gt=from_gt, style=args.style)

    if args.layout == "tiles":
        record.update(export_tiles(args, pick, out_dir))
        (out_dir / "selection.json").write_text(json.dumps(record, indent=1))
        print(json.dumps({k: v for k, v in record.items() if k != "prompt_examples"}, indent=1))
        return

    if args.layout in ("paper", "paper_v1"):
        record.update((paper_figures_v2 if args.layout == "paper" else paper_figures)(args, pick, out_dir))
        (out_dir / f"qualitative_selection_paper{args.suffix}.json").write_text(json.dumps(record, indent=1))
        print(json.dumps({k: v for k, v in record.items() if k != "prompt_examples"}, indent=1))
        return

    if args.layout == "strip":
        record.update(image_only_strip(args, pick, out_dir))
        (out_dir / f"qualitative_selection_strip{args.suffix}.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(record, indent=1))
        return

    # ---------------- image-only comparison (sampler S_L, training seed 1)
    columns, rows = [], [[], [], [], []]
    for dataset in ("pku", "cgl"):
        gt = ground_truth(dataset)
        released = load(ROOT / f"other_baselines/layoutidit/{dataset}_anno_uncond_test_output.pt")
        config = load(FIGS / f"rev_{dataset}_vit_saliency_linear_trainseed1_inferseed1_test_output.pt")
        ours = load(FIGS / f"rev_{dataset}_vit_both_linear_trainseed1_inferseed1_test_output.pt")
        names = sorted(set(gt) & set(released) & set(config) & set(ours))
        chosen = pick(names, "image_only", dataset)
        record[f"image_only_{dataset}"] = chosen
        for name in chosen:
            columns.append(f"{dataset.upper()} {name.split('.')[0]}")
            rows[0].append(draw(dataset, name, gt[name], from_gt=True))
            rows[1].append(draw(dataset, name, released[name]))
            rows[2].append(draw(dataset, name, config[name]))
            rows[3].append(draw(dataset, name, ours[name]))
    grid(rows, ["Designer layout", "LayoutDiT (released)", "LayoutDiT config.", "IntentDiT"], columns,
         out_dir / f"fig05_qualitative_image_only{args.suffix}.pdf")

    # ---------------- prompt-controlled layouts (sampler S_C, training seed 1)
    columns, rows, prompts = [], [[], [], []], []
    for dataset in ("pku", "cgl"):
        free = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}.csv").set_index("poster_path").text_prompt
        structured = pd.read_csv(ROOT / f"data/prompts/free_form_{dataset}_structured.csv").set_index("poster_path").text_prompt
        raw = load(FIGS / f"rev_prompt_{dataset}_vit_{args.text_model}_freeform_trainseed1_inferseed1_test_output.pt")
        listed = load(FIGS / f"rev_prompt_{dataset}_vit_{args.text_model}_structured_trainseed1_inferseed1_test_output.pt")
        postero = load(ROOT / f"other_baselines/standardized/postero_prompted_{dataset}_freeform_subset.pt")
        names = sorted(set(raw) & set(listed) & set(postero))
        chosen = pick(names, "prompt_control", dataset)
        for name in chosen:
            label = f"({chr(ord('a') + len(prompts))}) {dataset.upper()}"
            columns.append(label)
            prompts.append({"label": label, "image": name, "prompt": free[name], "element_list": structured[name]})
            rows[0].append(draw(dataset, name, raw[name]))
            rows[1].append(draw(dataset, name, listed[name]))
            rows[2].append(draw(dataset, name, postero[name]))
    grid(rows, ["IntentDiT, instruction", "IntentDiT, + list", "PosterO, + list"], columns,
         out_dir / f"fig03_prompt_control{args.suffix}.pdf")
    record["prompt_examples"] = prompts
    (out_dir / f"qualitative_selection{args.suffix}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
