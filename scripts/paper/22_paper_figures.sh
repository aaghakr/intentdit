#!/usr/bin/env bash
# Print-ready figures for the MDPI manuscript, numbered in order of appearance.
# Every figure is drawn at the full MDPI width (7.27 in) with embedded TrueType fonts;
# image tiles are recompressed as JPEG (quality 92) while text and lines stay vector.
#
# Usage: OUT=<dir> TEASER_TILES=<dir with input/map/layout/rendered.png> ARCH_TILES=<dir with input/saliency/placement/output.png> \
#        bash scripts/paper/22_paper_figures.sh
# Figure 2 (architecture) is a TikZ drawing (code/scripts/plot_architecture_v3.tex) compiled with pdflatex around the
# four ARCH_TILES thumbnails; all other figures are drawn with matplotlib (Nimbus Sans, embedded).
set -Eeuo pipefail
export PATH_PROFILE=${PATH_PROFILE:-local}
source "$(dirname "$0")/00_common.sh"
OUT=${OUT:?set OUT to the output directory}
TEASER_TILES=${TEASER_TILES:?set TEASER_TILES to the directory with the Figure 1 tiles}
ARCH_TILES=${ARCH_TILES:?set ARCH_TILES to the directory with the Figure 2 thumbnails}
WORK=$(mktemp -d)
mkdir -p "$OUT"

run_command "$PYTHON_BIN" "$CODE_DIR/scripts/plot_teaser.py" --input "$TEASER_TILES/input.png" \
    --map "$TEASER_TILES/map.png" --layout "$TEASER_TILES/layout.png" --rendered "$TEASER_TILES/rendered.png" \
    --output "$WORK/fig01.pdf"
mkdir -p "$WORK/arch" && cp "$ARCH_TILES"/{input,saliency,placement,output}.png "$WORK/arch/"
cp "$CODE_DIR/scripts/plot_architecture_v3.tex" "$WORK/fig02.tex"
(cd "$WORK" && run_command pdflatex -interaction=nonstopmode fig02.tex > /dev/null)
run_command "$PYTHON_BIN" "$CODE_DIR/scripts/plot_qualitative.py" --out-dir "$WORK" --selection curated
run_command "$PYTHON_BIN" "$CODE_DIR/scripts/plot_qualitative.py" --out-dir "$WORK" --selection random --suffix _random
run_command "$PYTHON_BIN" "$CODE_DIR/scripts/plot_density_intervention.py" --output "$WORK/fig05.pdf"
run_command "$PYTHON_BIN" "$CODE_DIR/scripts/plot_fidelity_tradeoff.py" --output "$WORK/fig07.pdf"

declare -A NAMES=(
    [fig01]=fig01 [fig02]=fig02 [fig03]=qual_image_only_pku [fig04]=qual_image_only_cgl [fig05]=fig05
    [fig06]=qual_prompt_control [fig07]=fig07
    [figA1]=qual_image_only_pku_random [figA2]=qual_image_only_cgl_random [figA3]=qual_prompt_control_random
)
for name in "${!NAMES[@]}"; do
    gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 \
        -dAutoFilterColorImages=false -dColorImageFilter=/DCTEncode -dJPEGQ=92 \
        -dDownsampleColorImages=false -dEmbedAllFonts=true -dSubsetFonts=true \
        -o "$OUT/$name.pdf" "$WORK/${NAMES[$name]}.pdf"
done
cp "$WORK"/qualitative_selection_paper*.json "$OUT/"
rm -rf "$WORK"
log "Figures written to $OUT"
