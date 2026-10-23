"""Prompt-control figure: one example per row, prompt banner above, request tile and outputs below.

Each example shows
  * a banner with the free-form prompt (positions in bold),
  * a request tile: the poster with a 3x3 grid and the requested cells of the element list shaded
    in the class colour and labelled, so the target is visible at a glance,
  * the outputs (IntentDiT on the raw instruction, IntentDiT and PosterO given the element list), each with
    a faint 3x3 grid because spatial adherence (SPLA) is scored by grid cell.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "code"))
from cgbdm.text_spatial import parse_positions_from_prompt  # noqa: E402
from utils.metric import _parse_prompt_counts  # noqa: E402

FONT_DIR = Path("/usr/share/fonts/opentype/urw-base35")
FONT_FILES = {"regular": "NimbusSans-Regular.otf", "italic": "NimbusSans-Italic.otf",
              "bold": "NimbusSans-Bold.otf", "bolditalic": "NimbusSans-BoldItalic.otf"}
INK, INK_2, BANNER, RULE, GRID = "#111111", "#5a5955", "#f1f0ec", "#8c8b86", (128, 128, 128)
CLASS_ID = {"Text": 1, "Logo": 2, "Underlay": 3, "Embellishment": 4}
ROWS, COLS = ("top", "middle", "bottom"), ("left", "center", "right")
CELL = re.compile(r"(top|middle|bottom)-(left|center|right)")


def _font(kind: str, size: int):
    return ImageFont.truetype(str(FONT_DIR / FONT_FILES[kind]), size)


def draw_grid(image: Image.Image, alpha=0.42) -> Image.Image:
    """Faint dashed 3x3 grid."""
    out = image.convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h = out.size
    dash, gap, lw = max(6, w // 60), max(4, w // 90), max(1, w // 300)
    for k in (1, 2):
        x = round(w * k / 3)
        for y0 in range(0, h, dash + gap):
            d.line([(x, y0), (x, min(h, y0 + dash))], fill=GRID + (int(255 * alpha),), width=lw)
        y = round(h * k / 3)
        for x0 in range(0, w, dash + gap):
            d.line([(x0, y), (min(w, x0 + dash), y)], fill=GRID + (int(255 * alpha),), width=lw)
    return Image.alpha_composite(out, layer).convert("RGB")


def request_tile(image: Image.Image, structured_prompt: str, colors: dict, edges: dict) -> tuple[Image.Image, str]:
    """Poster washed out, 3x3 grid, requested cells shaded and labelled. Returns the tile and a note on
    requested elements without a position."""
    counts = {k: v for k, v in _parse_prompt_counts(structured_prompt).items() if v}
    positions = parse_positions_from_prompt(structured_prompt)
    base = Image.blend(image.convert("RGB"), Image.new("RGB", image.size, "white"), 0.42)
    base = draw_grid(base, alpha=0.75).convert("RGBA")
    w, h = base.size
    cw, ch = w / 3, h / 3
    by_cell: dict[str, Counter] = {}
    for cls, cells in positions.items():
        for cell in cells[: counts.get(cls, len(cells))]:
            by_cell.setdefault(cell, Counter())[cls] += 1
    font = _font("bold", max(14, w // 17))
    for cell, classes in by_cell.items():
        r, c = ROWS.index(cell.split("-")[0]), COLS.index(cell.split("-")[1])
        x0, y0 = c * cw, r * ch
        parts = sorted(classes.items(), key=lambda kv: CLASS_ID[kv[0]])
        band = ch / len(parts)
        for k, (cls, n) in enumerate(parts):
            k_id = CLASS_ID[cls]
            inset = max(3, w // 90)
            box = (x0 + inset, y0 + k * band + inset, x0 + cw - inset, y0 + (k + 1) * band - inset)
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            fill = tuple(int(colors[k_id][i : i + 2], 16) for i in (1, 3, 5))
            edge = tuple(int(edges[k_id][i : i + 2], 16) for i in (1, 3, 5))
            d.rectangle(box, fill=fill + (150,), outline=edge + (255,), width=max(2, w // 200))
            base = Image.alpha_composite(base, layer)
            label = cls if n == 1 else f"{cls} ×{n}"
            d = ImageDraw.Draw(base)
            tw = d.textlength(label, font=font)
            d.text(((box[0] + box[2]) / 2 - tw / 2, (box[1] + box[3]) / 2), label, fill=(17, 17, 17, 255),
                   font=font, anchor="lm")
    placed = Counter()
    for cls_counts in by_cell.values():
        placed.update(cls_counts)
    missing = [f"{n - placed[cls]} {cls.lower()}{'s' if n - placed[cls] > 1 else ''}"
               for cls, n in counts.items() if n > placed[cls]]
    note = ("+ " + ", ".join(missing) + " without position") if missing else ""
    return base.convert("RGB"), note


def _layout_words(text: str, width_pt: float, size: float) -> list[list[tuple[str, str, float]]]:
    """Word wrap with per-word style (positions bold); returns lines of (word, style, x offset in pt)."""
    metrics = {k: _font(k, 100) for k in ("italic", "bolditalic")}
    scale = size / 100 * 72 / 72  # font units at size 100 -> pt at `size`
    space = metrics["italic"].getlength(" ") * scale
    lines, line, x = [], [], 0.0
    for word in text.split():
        # a position word is set bold; trailing punctuation stays regular
        m = CELL.search(word)
        pieces = [(word, "italic")] if not m else [p for p in ((word[: m.start()], "italic"),
                                                                 (word[m.start(): m.end()], "bolditalic"),
                                                                 (word[m.end():], "italic")) if p[0]]
        wlen = sum(metrics[st].getlength(t) for t, st in pieces) * scale
        if line and x + wlen > width_pt:
            lines.append(line)
            line, x = [], 0.0
        for t, st in pieces:
            line.append((t, st, x))
            x += metrics[st].getlength(t) * scale
        x += space
    return lines + ([line] if line else [])


def prompt_figure(examples, output: Path, colors: dict, edges: dict, classes=(1, 2, 3), width=5.47) -> None:
    """examples: list of dicts with label, prompt, tiles=[request, raw, list, postero], note."""
    plt.rcParams.update({"font.family": "Nimbus Sans", "pdf.fonttype": 42})
    gutter, n_cols = 0.045, 4
    tile_w = (width - gutter * (n_cols - 1)) / n_cols
    tile_h = tile_w * examples[0]["tiles"][0].height / examples[0]["tiles"][0].width
    size, lead = 6.9, 1.28
    text_w_pt = (width - 0.42) * 72
    layouts = [_layout_words(f"“{e['prompt']}”", text_w_pt, size) for e in examples]
    banner_hs = [0.10 + len(lines) * size * lead / 72 for lines in layouts]
    head1, head2, row_gap, legend_h = 0.17, 0.17, 0.16, 0.30
    height = head1 + head2 + sum(b + 0.05 + tile_h for b in banner_hs) + row_gap * (len(examples) - 1) + legend_h
    fig = plt.figure(figsize=(width, height))
    canvas = fig.add_axes([0, 0, 1, 1]); canvas.axis("off")
    canvas.set_xlim(0, width); canvas.set_ylim(0, height)
    xs = [k * (tile_w + gutter) for k in range(n_cols)]

    # Two-tier header.
    top = height
    tier1 = [("Instruction only", 1, 1), ("Instruction + element list", 2, 3)]
    for label, a, b in tier1:
        x0, x1 = xs[a], xs[b] + tile_w
        canvas.text((x0 + x1) / 2, top - 0.02, label, ha="center", va="top", fontsize=7.2, color=INK_2,
                    fontstyle="italic")
        canvas.plot([x0 + 0.03, x1 - 0.03], [top - head1 + 0.015] * 2, color=RULE, lw=0.6)
    names = [("Requested cells", False), ("IntentDiT", True), ("IntentDiT", True), ("PosterO", False)]
    for k, (name, bold) in enumerate(names):
        canvas.text(xs[k] + tile_w / 2, top - head1 - 0.03, name, ha="center", va="top", fontsize=7.8,
                    color=INK, fontweight="bold" if bold else "normal")

    y = top - head1 - head2
    for e, lines, bh in zip(examples, layouts, banner_hs):
        # Prompt banner.
        canvas.add_patch(FancyBboxPatch((0.0, y - bh), width, bh, boxstyle="round,pad=0,rounding_size=0.045",
                                        facecolor=BANNER, edgecolor="none"))
        canvas.text(0.07, y - 0.05 - size * lead / 72 / 2, e["label"], ha="left", va="center", fontsize=size + 0.4,
                    color=INK, fontweight="bold")
        for li, line in enumerate(lines):
            ly = y - 0.05 - (li + 0.5) * size * lead / 72
            for word, style, off in line:
                canvas.text(0.36 + off / 72, ly, word, ha="left", va="center", fontsize=size, color=INK,
                            fontstyle="italic", fontweight="bold" if style == "bolditalic" else "normal")
        y -= bh + 0.05
        for k, tile in enumerate(e["tiles"]):
            ax = fig.add_axes([xs[k] / width, (y - tile_h) / height, tile_w / width, tile_h / height])
            ax.imshow(tile, interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#cfcec9"); spine.set_linewidth(0.4)
        if e.get("note"):
            # figure-level text so that it is drawn above the image axes
            fig.text((xs[0] + 0.04) / width, (y - tile_h + 0.05) / height, e["note"], ha="left", va="bottom", fontsize=5.8,
                        color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
        y -= tile_h + row_gap

    # Legend: classes, requested cell, grid.
    from PIL import ImageFont as _IF  # noqa: F401
    items = [(NAME, c) for c, NAME in ((k, {1: "Text", 2: "Logo", 3: "Underlay", 4: "Embellishment"}[k]) for k in classes)]
    reg = _font("regular", 100)
    x, ly = 0.0, 0.10
    entries = [("box", c, n) for n, c in items] + [("grid", None, "3×3 grid (cells scored by SPLA)")]
    widths = [0.17 + 0.05 + reg.getlength(t) * 7 / 100 / 72 for _, _, t in entries]
    x = (width - (sum(widths) + 0.16 * (len(entries) - 1))) / 2
    for (kind, c, t), wdt in zip(entries, widths):
        if kind == "box":
            canvas.add_patch(Rectangle((x, ly - 0.035), 0.17, 0.075, facecolor=colors[c], alpha=0.8, edgecolor="none"))
            canvas.add_patch(Rectangle((x, ly - 0.035), 0.17, 0.075, facecolor="none", edgecolor=edges[c], lw=0.6))
        elif kind == "cell":
            canvas.add_patch(Rectangle((x, ly - 0.035), 0.17, 0.075, facecolor="#dcdbd6", edgecolor=INK_2, lw=0.6))
        else:
            for dx in (0.057, 0.113):
                canvas.plot([x + dx] * 2, [ly - 0.04, ly + 0.045], color="#808080", lw=0.5, ls=(0, (2, 1.5)))
            for dy in (-0.012, 0.018):
                canvas.plot([x, x + 0.17], [ly + dy] * 2, color="#808080", lw=0.5, ls=(0, (2, 1.5)))
        canvas.text(x + 0.22, ly, t, ha="left", va="center", fontsize=7, color=INK)
        x += wdt + 0.16
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)


def element_list_line(structured_prompt: str) -> str:
    """Compact element list, e.g. 'Logo top-left · Text top-center · Text middle-center'."""
    counts = {k: v for k, v in _parse_prompt_counts(structured_prompt).items() if v}
    positions = parse_positions_from_prompt(structured_prompt)
    items = []
    for cls in sorted(counts, key=lambda c: CLASS_ID[c]):
        cells = positions.get(cls, [])[: counts[cls]]
        items += [f"{cls} {cell}" for cell in cells]
        items += [f"{cls} (no position)"] * (counts[cls] - len(cells))
    return "  ·  ".join(items)


def prompt_pair_figure(columns, output: Path, colors: dict, edges: dict, tile_w: float, gutter: float,
                       width: float = 7.27, classes=(1, 2, 3, 4)) -> None:
    """Full-width figure with two blocks side by side (columns = [(dataset label, examples), ...]).

    Each example: label, prompt, element_list, tiles (raw, + list, PosterO). Tiles have the same printed
    size as in the image-only figures (tile_w, gutter)."""
    plt.rcParams.update({"font.family": "Nimbus Sans", "pdf.fonttype": 42})
    block_w = 3 * tile_w + 2 * gutter
    block_gap = width - 2 * block_w
    sample = columns[0][1][0]["tiles"][0]
    tile_h = tile_w * sample.height / sample.width
    size, lead, note_size = 6.3, 1.25, 5.8
    text_x, text_w_pt = 0.07, (block_w - 0.07 - 0.06) * 72
    reg = _font("regular", 100)

    def note_lines(text):
        lines, line = [], ""
        for word in text.split(" "):
            trial = f"{line} {word}" if line else word
            if line and reg.getlength(trial) * note_size / 100 > text_w_pt:
                lines.append(line.strip()); line = word
            else:
                line = trial
        return lines + ([line.strip()] if line else [])

    layouts = [[(_layout_words(f"“{e['prompt']}”", text_w_pt, size), note_lines(f"List: {e['element_list']}"))
                for e in examples] for _, examples in columns]
    n_rows = max(len(examples) for _, examples in columns)

    def banner_h(item):
        lines, notes = item
        return 0.09 + len(lines) * size * lead / 72 + 0.025 + len(notes) * note_size * lead / 72

    row_banner = [max(banner_h(col[r]) for col in layouts if r < len(col)) for r in range(n_rows)]
    head_ds, head_tier, head_name, row_gap, legend_h = 0.20, 0.15, 0.16, 0.13, 0.27
    height = head_ds + head_tier + head_name + sum(b + 0.035 + tile_h for b in row_banner) + row_gap * (n_rows - 1) + legend_h
    fig = plt.figure(figsize=(width, height))
    canvas = fig.add_axes([0, 0, 1, 1]); canvas.axis("off")
    canvas.set_xlim(0, width); canvas.set_ylim(0, height)

    for b, (ds_label, examples) in enumerate(columns):
        bx = b * (block_w + block_gap)
        xs = [bx + k * (tile_w + gutter) for k in range(3)]
        top = height
        canvas.text(bx + block_w / 2, top - 0.02, ds_label, ha="center", va="top", fontsize=7.8, color=INK,
                    fontweight="bold")
        canvas.plot([bx, bx + block_w], [top - head_ds + 0.025] * 2, color=INK_2, lw=0.7)
        for label, a, c in (("Instruction only", 0, 0), ("Instruction + element list", 1, 2)):
            x0, x1 = xs[a], xs[c] + tile_w
            canvas.text((x0 + x1) / 2, top - head_ds - 0.005, label, ha="center", va="top", fontsize=6.3,
                        color=INK_2, fontstyle="italic")
            canvas.plot([x0 + 0.04, x1 - 0.04], [top - head_ds - head_tier + 0.02] * 2, color=RULE, lw=0.5)
        for k, (name, bold) in enumerate((("IntentDiT", True), ("IntentDiT", True), ("PosterO", False))):
            canvas.text(xs[k] + tile_w / 2, top - head_ds - head_tier - 0.012, name, ha="center", va="top",
                        fontsize=7, color=INK, fontweight="bold" if bold else "normal")
        y = top - head_ds - head_tier - head_name
        for r, (e, (lines, notes)) in enumerate(zip(examples, layouts[b])):
            bh = row_banner[r]
            canvas.add_patch(FancyBboxPatch((bx, y - bh), block_w, bh, boxstyle="round,pad=0,rounding_size=0.04",
                                            facecolor=BANNER, edgecolor="none"))
            first = y - 0.045 - 0.5 * size * lead / 72
            for li, line in enumerate(lines):
                ly = first - li * size * lead / 72
                for word, style, off in line:
                    canvas.text(bx + text_x + off / 72, ly, word, ha="left", va="center", fontsize=size, color=INK,
                                fontstyle="italic", fontweight="bold" if style == "bolditalic" else "normal")
            ny = first - len(lines) * size * lead / 72 - 0.025
            for ni, note in enumerate(notes):
                canvas.text(bx + text_x, ny - ni * note_size * lead / 72, note, ha="left", va="center",
                            fontsize=note_size, color=INK_2)
            y -= bh + 0.035
            for k, tile in enumerate(e["tiles"]):
                ax = fig.add_axes([xs[k] / width, (y - tile_h) / height, tile_w / width, tile_h / height])
                ax.imshow(tile, interpolation="lanczos")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cfcec9"); spine.set_linewidth(0.4)
            y -= tile_h + row_gap

    names = {1: "Text", 2: "Logo", 3: "Underlay", 4: "Embellishment"}
    entries = [("box", c, names[c]) for c in classes] + [("grid", None, "3×3 grid (cells scored by SPLA)")]
    widths = [0.17 + 0.05 + reg.getlength(t) * 6.6 / 100 / 72 for _, _, t in entries]
    x, ly = (width - (sum(widths) + 0.12 * (len(entries) - 1))) / 2, 0.10
    for (kind, c, t), wdt in zip(entries, widths):
        if kind == "box":
            canvas.add_patch(Rectangle((x, ly - 0.035), 0.17, 0.075, facecolor=colors[c], alpha=0.8, edgecolor="none"))
            canvas.add_patch(Rectangle((x, ly - 0.035), 0.17, 0.075, facecolor="none", edgecolor=edges[c], lw=0.6))
        else:
            for dx in (0.057, 0.113):
                canvas.plot([x + dx] * 2, [ly - 0.04, ly + 0.045], color="#808080", lw=0.5, ls=(0, (2, 1.5)))
            for dy in (-0.012, 0.018):
                canvas.plot([x, x + 0.17], [ly + dy] * 2, color="#808080", lw=0.5, ls=(0, (2, 1.5)))
        canvas.text(x + 0.22, ly, t, ha="left", va="center", fontsize=6.6, color=INK)
        x += wdt + 0.12
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)
