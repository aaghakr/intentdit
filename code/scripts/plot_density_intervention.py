"""Plot occlusion against layout density: designer layouts thinned at random vs. models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "experiments" / "paper_results"

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#8c8b86"


def seed_mean(summary: dict, run: str, metric: str) -> float:
    return summary[run]["metrics"][metric]["mean"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    thinning = json.loads((RESULTS / "gt_drop" / "gt_drop_summary.json").read_text())
    linear = json.loads((RESULTS / "primary_training_seed_summary_linear.json").read_text())
    cosine = json.loads((RESULTS / "primary_training_seed_summary_cosine.json").read_text())
    # Revision runs: PKU models retrained on corrected placement maps (rev_*); the
    # CGL image-only models are unchanged and re-sampled under S_L as rev_*_linear.
    revision = json.loads((RESULTS / "revision" / "rev_seed_summary.json").read_text())

    def model_point(dataset: str, slug: str, sampler: str, metric: str) -> float:
        key = f"rev_{dataset}_vit_{slug}_{sampler}"
        if key in revision:
            return seed_mean(revision, key, metric)
        source = linear if sampler == "linear" else cosine
        return seed_mean(source, f"ivc_{dataset}_vit_{slug}", metric)

    # Drawn at the MDPI full width (7.27 in) so that the placed figure is not rescaled.
    plt.rcParams.update({"font.family": "Nimbus Sans", "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold", "font.size": 8, "axes.edgecolor": INK_2,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.27, 2.85), sharey=False)
    for ax, dataset, title in zip(axes, ("pku", "cgl"), ("PKU", "CGL")):
        rates = sorted(thinning[dataset], key=int)
        n = [thinning[dataset][r]["n_pred"][0] for r in rates]
        occ = [thinning[dataset][r]["occ"][0] for r in rates]
        ax.plot(n, occ, color=NEUTRAL, lw=1.5, marker="o", ms=4, zorder=1)
        for r, x, y in zip(rates, n, occ):
            ax.annotate(f"{int(r)}%", (x, y), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=7, color=INK_2, zorder=4,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))

        released = json.loads((PROJECT_ROOT / "experiments" / "paper_figures"
                               / f"baseline_layoutdit_{dataset}_image_only_metrics.json").read_text())
        ax.scatter(released["n_pred"], released["occ"], s=34, marker="s", facecolor=ORANGE, edgecolor="white",
                   linewidth=1.2, zorder=3)
        for slug, color in (("saliency", ORANGE), ("both", BLUE)):
            pts = {sampler: (model_point(dataset, slug, sampler, "n_pred"), model_point(dataset, slug, sampler, "occ"))
                   for sampler in ("linear", "cosine")}
            # the same trained weights under the two samplers
            ax.plot([pts["linear"][0], pts["cosine"][0]], [pts["linear"][1], pts["cosine"][1]],
                    color=color, lw=0.8, ls=(0, (2, 2)), zorder=2)
            for sampler, filled in (("linear", True), ("cosine", False)):
                x, y = pts[sampler]
                ax.scatter(x, y, s=34, marker="o", facecolor=color if filled else "white",
                           edgecolor=color if not filled else "white", linewidth=1.6 if not filled else 1.2,
                           zorder=3)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("elements per layout (mean)")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0.5, 5.8)
    axes[0].set_ylabel("Occ (lower is better)")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=NEUTRAL, marker="o", ms=4, lw=1.5, label="Designer layouts, thinned at random"),
        Line2D([], [], color="none", marker="s", ms=6, markerfacecolor=ORANGE, markeredgecolor="white",
               label="LayoutDiT, released outputs"),
        Line2D([], [], color="none", marker="o", ms=6, markerfacecolor=ORANGE, markeredgecolor="white",
               label="LayoutDiT config., retrained"),
        Line2D([], [], color="none", marker="o", ms=6, markerfacecolor=BLUE, markeredgecolor="white",
               label="IntentDiT"),
        Line2D([], [], color="none", marker="o", ms=6, markerfacecolor=NEUTRAL, markeredgecolor="white",
               label=r"filled: sampler $S_{\mathrm{L}}$"),
        Line2D([], [], color=NEUTRAL, lw=0.8, ls=(0, (2, 2)), marker="o", ms=6, markerfacecolor="white",
               markeredgecolor=NEUTRAL, markeredgewidth=1.4, label=r"hollow: sampler $S_{\mathrm{C}}$ (same weights)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0), columnspacing=2.0, handletextpad=0.6)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.30, wspace=0.18)
    fig.savefig(args.output)
    fig.savefig(str(Path(args.output).with_suffix(".png")), dpi=200)


if __name__ == "__main__":
    main()
