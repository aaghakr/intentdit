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
# name: (colour, marker).  Colours follow Figure 5: IntentDiT blue, LayoutDiT configuration orange.
STYLE = {
    "LayoutDiT config.": ("#eb6834", "o"),
    "Placement map only": ("#1baf7a", "s"),
    r"IntentDiT, $\lambda_2=0$": ("#2a78d6", "D"),
    r"IntentDiT, $\lambda_2=0.05$": ("#2a78d6", "o"),
    "Mean-density prior": ("#8c8b86", "v"),
    "Inverse-saliency prior": ("#8c8b86", "P"),
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
        "LayoutDiT config.": (8, 4, "left"),
        "Placement map only": (8, 2, "left"),
        r"IntentDiT, $\lambda_2=0$": (0, -16, "center"),
        r"IntentDiT, $\lambda_2=0.05$": (0, 15, "center"),
        "Mean-density prior": (-8, 4, "right"),
        "Inverse-saliency prior": (-8, -2, "right"),
    },
    # ("at", x, y, ha): label placed at data coordinates (x, y) with a thin leader line to the marker,
    # for markers that sit too close together for offset labels.
    "CGL": {
        "LayoutDiT config.": ("at", 1.09, 0.203, "left"),
        "Placement map only": (8, 2, "left"),
        r"IntentDiT, $\lambda_2=0$": ("at", 2.05, 0.151, "center"),
        r"IntentDiT, $\lambda_2=0.05$": (0, -16, "center"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Drawn at the MDPI full width (7.27 in) so that the placed figure is not rescaled.
    plt.rcParams.update({"font.family": "Nimbus Sans", "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold", "font.size": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.27, 3.0), sharey=True)
    for ax, (dataset, rows) in zip(axes, DATA.items()):
        ax.axhline(DESIGNER_OCC[dataset], color=INK_2, lw=0.8, ls=(0, (4, 3)))
        ax.text(1.08, DESIGNER_OCC[dataset] - 0.0015, "designer layouts", color=INK_2, fontsize=7, va="top")
        for name, (fid, fid_sd, occ, occ_sd, n) in rows.items():
            colour, marker = STYLE[name]
            ax.errorbar(fid, occ, xerr=fid_sd, yerr=occ_sd, fmt="none", ecolor=colour, elinewidth=1, capsize=0, zorder=2)
            hollow = name == r"IntentDiT, $\lambda_2=0$"
            ax.scatter(fid, occ, s=42, marker=marker, facecolor="white" if hollow else colour, edgecolors=colour,
                       linewidths=1.6 if hollow else 1.0, zorder=3)
            place = OFFSETS[dataset][name]
            if place[0] == "at":
                _, lx, ly, ha = place
                ax.annotate(f"{name}\n$n$ = {n:.2f}", (fid, occ), xytext=(lx, ly), textcoords="data",
                            ha=ha, va="center", fontsize=7, color=INK, linespacing=1.05,
                            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.5, shrinkA=1, shrinkB=5))
            else:
                dx, dy, ha = place
                ax.annotate(f"{name}\n$n$ = {n:.2f}", (fid, occ), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va="center", fontsize=7, color=INK, linespacing=1.05)
        start, end = rows[r"IntentDiT, $\lambda_2=0$"], rows[r"IntentDiT, $\lambda_2=0.05$"]
        ax.annotate("", xy=(end[0] * 0.93, end[2]), xytext=(start[0] * 1.12, start[2]),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.9, shrinkA=2, shrinkB=2))
        mid = (start[0] * end[0]) ** 0.5
        ax.text(mid, (start[2] + end[2]) / 2 + (0.002 if dataset == "PKU" else 0.0045), "placement loss",
                ha="center", va="bottom", fontsize=7, color=INK_2)
        ax.set_xscale("log")
        ax.set_xlim(1.0, 70)
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
        ax.set_title(dataset, fontsize=8.5, color=INK, loc="left", fontweight="bold")
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
