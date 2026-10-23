#!/usr/bin/env bash
# Instruction-conditioned PosterO given the gold element list (manual reference counts
# and positions, appended to the instruction), for comparison with IntentDiT on the
# gold structured prompts (data/prompts/free_form_<dataset>_gold.csv). Requests are
# built from the manual references; predictor boxes and features come from the
# postero phase of 20_revision_runs.sh.
#
# Usage: GPU_IDS=0 bash scripts/paper/21_postero_gold.sh
set -Eeuo pipefail
export PATH_PROFILE=${PATH_PROFILE:-local}
source "$(dirname "$0")/00_common.sh"
mkdir_outputs
POSTERO_ROOT=${POSTERO_ROOT:-/home/viplab/Aagha/Archive/layout/PosterO-CVPR2025}
POSTERO_PYTHON=${POSTERO_PYTHON:-/home/viplab/anaconda3/envs/VLM_SETUP/bin/python}
POSTERO_PYDEPS=${POSTERO_PYDEPS:-$POSTERO_ROOT/.pydeps}
POSTERO_GPU_MEMORY_UTILIZATION=${POSTERO_GPU_MEMORY_UTILIZATION:-0.6}
gpu=${GPU_IDS%%,*}

for dataset in ${DATASETS:-pku cgl}; do
    epoch=$([[ "$dataset" == pku ]] && echo 100 || echo 35)
    bbox_dir="design_intent_detect/${dataset}_128_1e-06_none/result/design_intent_${dataset}_epoch${epoch}"
    out="$ACTIVE_ROOT/other_baselines/postero_prompted/postero_prompted_gold_${dataset}.pt"
    args=(
        "$POSTERO_PYTHON" scripts/run_prompted_infer.py
        --requests-json "$ACTIVE_ROOT/other_baselines/postero_prompted/requests_gold_${dataset}.json"
        --out-path "$out"
        --dataset_name "$dataset" --structure hierarchical --injection top
        --design_intent_bbox_dir "$bbox_dir" --annotation_dir "dataset/$dataset/"
        --model_dir models/Meta-Llama-3.1-8B --N 10 --num_return 10
        --pool_strategy metric_filter --rank_strategy rank_by_feature
        --metric_path "sample_select/metric_train_${dataset}.pt"
        --filter_dict "sample_select/filter_metric_${dataset}.json"
        --feature_dir "$bbox_dir/${dataset}_features" --sample_size 10 --batch_size 1
    )
    [[ "$dataset" == cgl ]] && args+=(--name-map-csv "$ACTIVE_ROOT/data/dataset/cgl/split/csv/test.csv")
    log "POSTERO-PROMPTED-GOLD $dataset on GPU $gpu"
    [[ -f "$out" && "${FORCE:-0}" != "1" ]] || (cd "$POSTERO_ROOT" && env CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONPATH="$POSTERO_PYDEPS" POSTERO_GPU_MEMORY_UTILIZATION="$POSTERO_GPU_MEMORY_UTILIZATION" "${args[@]}") \
        2>&1 | tee "$LOG_DIR/rev_postero_prompted_gold_${dataset}.log"
    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/import_postero_predictions.py" \
        --input "$out" --output "$ACTIVE_ROOT/other_baselines/standardized/postero_prompted_gold_${dataset}_freeform_subset.pt" \
        --dataset "$dataset" --split valid --selection official --postero-code-root "$POSTERO_ROOT" \
        --subset-csv "$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv" \
        $([[ "$dataset" == cgl ]] && echo --name-map-csv "$ACTIVE_ROOT/data/dataset/cgl/split/csv/test.csv")
done
log "PosterO gold runs complete"
