"""Occlusion against LayoutFID for the image-only models under sampler S_C.

Values are the means and standard deviations over three training seeds reported
in Tables 3 and 5 of the paper (metrics in experiments/paper_figures, LayoutFID in
experiments/paper_results/revision/layoutfid). The arrow marks the effect of the
placement loss (lambda_2 = 0 -> 0.05) with all inputs fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# name: (colour, marker)
STYLE = {
    "LayoutDiT config.": ("#2a78d6", "o"),
    "Placement map only": ("#1baf7a", "s"),
    r"IntentDiT, $\lambda_2=0$": ("#4a3aa7", "D"),
    r"IntentDiT, $\lambda_2=0.05$": ("#eb6834", "^"),
    "Mean-density prior": ("#eda100", "v"),
    "Inverse-saliency prior": ("#e87ba4", "P"),
}
# name: (FID mean, FID std, Occ mean, Occ std, n)
DATA = {
    "PKU": {
        "LayoutDiT config.": (1.99, 0.11, 0.165, 0.002, 4.42),
        "Placement map only": (1.91, 0.13, 0.188, 0.012, 4.53),
        r"IntentDiT, $\lambda_2=0$": (1.66, 0.25, 0.154, 0.006, 4.41),
        r"IntentDiT, $\lambda_2=0.05$": (14.36, 0.20, 0.153, 0.003, 4.98),
        "Mean-density prior": (23.22, 0.58, 0.194, 0.001, 5.06),
        "Inverse-saliency prior": (35.97, 2.18, 0.212, 0.005, 5.18),
    },
    "CGL": {
        "LayoutDiT config.": (1.92, 0.07, 0.174, 0.004, 4.90),
        "Placement map only": (3.55, 0.71, 0.190, 0.003, 5.28),
        r"IntentDiT, $\lambda_2=0$": (2.29, 0.10, 0.175, 0.011, 4.92),
        r"IntentDiT, $\lambda_2=0.05$": (26.77, 0.67, 0.158, 0.003, 5.37),
    },
}
DESIGNER_OCC = {"PKU": 0.127, "CGL": 0.141}
# label offsets in points (dx, dy, ha) per dataset
OFFSETS = {
    "PKU": {
        "LayoutDiT config.": (7, 3, "left"),
        "Placement map only": (7, 0, "left"),
        r"IntentDiT, $\lambda_2=0$": (0, -15, "center"),
        r"IntentDiT, $\lambda_2=0.05$": (0, 14, "center"),
        "Mean-density prior": (-7, 4, "right"),
        "Inverse-saliency prior": (-7, -2, "right"),
    },
    "CGL": {
        "LayoutDiT config.": (-7, 4, "right"),
        "Placement map only": (7, 0, "left"),
        r"IntentDiT, $\lambda_2=0$": (6, -15, "left"),
        r"IntentDiT, $\lambda_2=0.05$": (0, -15, "center"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, (dataset, rows) in zip(axes, DATA.items()):
        ax.axhline(DESIGNER_OCC[dataset], color=INK_2, lw=0.8, ls=(0, (4, 3)))
        ax.text(1.05, DESIGNER_OCC[dataset] + 0.0015, "designer layouts", color=INK_2, fontsize=6.5, va="bottom")
        for name, (fid, fid_sd, occ, occ_sd, n) in rows.items():
            colour, marker = STYLE[name]
            ax.errorbar(fid, occ, xerr=fid_sd, yerr=occ_sd, fmt="none", ecolor=colour, elinewidth=1, capsize=0, zorder=2)
            ax.scatter(fid, occ, s=42, marker=marker, color=colour, edgecolors="white", linewidths=1.5, zorder=3)
            dx, dy, ha = OFFSETS[dataset][name]
            ax.annotate(f"{name}\n$n$ = {n:.2f}", (fid, occ), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=6.3, color=INK, linespacing=1.0)
        start, end = rows[r"IntentDiT, $\lambda_2=0$"], rows[r"IntentDiT, $\lambda_2=0.05$"]
        ax.annotate("", xy=(end[0] * 0.93, end[2]), xytext=(start[0] * 1.12, start[2]),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.9, shrinkA=2, shrinkB=2))
        mid = (start[0] * end[0]) ** 0.5
        ax.text(mid, (start[2] + end[2]) / 2 + (0.002 if dataset == "PKU" else 0.006), "placement loss", ha="center", va="bottom", fontsize=6.3, color=INK_2)
        ax.set_xscale("log")
        ax.set_xlim(1.0, 70)
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
        ax.set_title(dataset, fontsize=8, color=INK)
        ax.set_xlabel(r"LayoutFID (log scale) $\downarrow$", color=INK)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=INK_2, width=0.6)
    axes[0].set_ylabel(r"Occ $\downarrow$", color=INK)
    axes[0].set_ylim(0.120, 0.222)
    fig.tight_layout(w_pad=1.2)
    out = Path(args.output)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)


if __name__ == "__main__":
    main()
