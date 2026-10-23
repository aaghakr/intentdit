"""Overview figure (Figure 1): inputs -> IntentDiT -> generated layout and rendered design.

Drawn at the MDPI full width (7.27 in) with vector labels and arrows; the four image
tiles (background image, conditioning map, generated layout, rendered design) are
given as files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

INK, INK_2, FRAME, FILL = "#0b0b0b", "#52514e", "#bdbcb7", "#f3f3f1"
WIDTH, HEIGHT = 7.27, 2.42
PROMPT = "A layout with one logo at the\ntop-center and three text\nboxes below."


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("input", "map", "layout", "rendered"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--prompt", default=PROMPT, help="Prompt text; use \\n for the line break")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "Nimbus Sans", "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, WIDTH)
    canvas.set_ylim(0, HEIGHT)
    canvas.axis("off")

    tile_w, tile_h, gap = 1.0, 1.46, 0.08
    top = HEIGHT - 0.28  # top edge of the image tiles
    bottom = top - tile_h
    middle = (top + bottom) / 2

    def tile(path, x, label):
        ax = fig.add_axes([x / WIDTH, bottom / HEIGHT, tile_w / WIDTH, tile_h / HEIGHT])
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(FRAME)
            spine.set_linewidth(0.6)
        canvas.text(x + tile_w / 2, bottom - 0.06, label, ha="center", va="top", fontsize=7, color=INK_2)

    def header(x0, x1, text):
        canvas.text((x0 + x1) / 2, top + 0.08, text, ha="center", va="bottom", fontsize=8.5,
                    color=INK, fontweight="bold")

    def arrow(x0, x1, y):
        canvas.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9,
                                         color=INK, lw=1.0, shrinkA=0, shrinkB=0))

    # Inputs: background image, conditioning map and prompt.
    total = 2 * (2 * tile_w + gap) + 0.42 + 1.88 + 0.42  # inputs, arrow, model, arrow, output
    x_in = (WIDTH - total) / 2
    tile(args.input, x_in, "Background image")
    tile(args.map, x_in + tile_w + gap, "Conditioning map")
    in_right = x_in + 2 * tile_w + gap
    header(x_in, in_right, "Inputs")
    prompt_y, prompt_h = 0.08, 0.40
    canvas.add_patch(FancyBboxPatch((x_in, prompt_y), in_right - x_in, prompt_h,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="white", edgecolor=INK_2, lw=0.8))
    canvas.text(x_in + 0.07, prompt_y + prompt_h / 2, "Prompt", ha="left", va="center", fontsize=7.5,
                color=INK, fontweight="bold")
    prompt = args.prompt.replace("\\n", "\n")
    canvas.text(x_in + 0.55, prompt_y + prompt_h / 2, f"“{prompt}”", ha="left", va="center",
                fontsize=6.8, color=INK, linespacing=1.15)

    # Model.
    m_x0, m_x1 = in_right + 0.42, in_right + 2.30
    m_y0, m_y1 = prompt_y, top
    canvas.add_patch(FancyBboxPatch((m_x0, m_y0), m_x1 - m_x0, m_y1 - m_y0,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=FILL, edgecolor=INK, lw=1.0))
    cx = (m_x0 + m_x1) / 2
    canvas.text(cx, m_y1 - 0.30, "IntentDiT", ha="center", va="center", fontsize=14, color=INK,
                fontweight="bold")
    canvas.text(cx, m_y1 - 0.54, "diffusion transformer", ha="center", va="center", fontsize=8, color=INK_2)
    canvas.add_artist(plt.Line2D([m_x0 + 0.25, m_x1 - 0.25], [m_y1 - 0.72] * 2, color=FRAME, lw=0.7))
    blocks = [("Implicit intent", "placement-suitability map\nand saliency cues"),
              ("Explicit intent", "prompt tokens and\nparsed grid positions")]
    for k, (title, body) in enumerate(blocks):
        y = m_y1 - 0.92 - k * 0.56
        canvas.text(cx, y, title, ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold")
        canvas.text(cx, y - 0.22, body, ha="center", va="center", fontsize=7, color=INK_2, linespacing=1.15)
    arrow(in_right + 0.04, m_x0, middle)
    arrow(in_right + 0.04, m_x0, prompt_y + prompt_h / 2)

    # Output: generated layout and rendered design.
    out_x = m_x1 + 0.42
    tile(args.layout, out_x, "Generated layout")
    tile(args.rendered, out_x + tile_w + gap, "Rendered design")
    header(out_x, out_x + 2 * tile_w + gap, "Output")
    arrow(m_x1, out_x - 0.04, middle)

    fig.savefig(args.output, dpi=300)
    fig.savefig(Path(args.output).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
