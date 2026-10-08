#!/usr/bin/env bash
# Revision experiments for the MDPI resubmission (see submissions/MDPI/IntentDiT/REVISION_NOTES.md).
#
# What changed relative to the IVC experiments, and why:
#   * PKU placement-suitability maps are regenerated with the cited PKU predictor
#     (data/*/intent_map_v2, csv/*_intent_mbbox_v2.csv); the maps used before did not
#     come from that checkpoint (map correlation 0.07 -> 0.53).
#   * The count reader behind L_cnt is fixed (utils/metric.py), so the count-loss
#     targets are now correct; the main configuration becomes lambda1=0, lambda2=0.05.
#   * Simple spatial priors (mean train density, inverse saliency) are trained inside
#     the generator as controls for the learned map (reviewer R1.7).
#   * Every image-only model is evaluated under both samplers, S_L (linear, as in the
#     released LayoutDiT) and S_C (cosine), keeping per-image outputs for paired bootstrap.
#
# All outputs use the prefix rev_ and never overwrite the IVC evidence (ivc_*).
#
# Usage (from the repository root; GPUs as a comma list, several jobs per GPU are fine):
#   GPU_IDS=0,1 JOBS_PER_GPU=4 bash scripts/paper/20_revision_runs.sh train      # all training
#   GPU_IDS=0,1 JOBS_PER_GPU=4 bash scripts/paper/20_revision_runs.sh eval       # all evaluation
#   GPU_IDS=0,1 JOBS_PER_GPU=4 bash scripts/paper/20_revision_runs.sh reeval     # S_L per-image for existing models
#   GPU_IDS=0     bash scripts/paper/20_revision_runs.sh postero                 # prompt-conditioned PosterO
#   bash scripts/paper/20_revision_runs.sh aggregate                             # summaries (CPU)
# Add DRY_RUN=1 to print the commands without running them.

set -Eeuo pipefail
export PATH_PROFILE=${PATH_PROFILE:-local}
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

PHASE=${1:-}
[[ -n "$PHASE" ]] || die "usage: $0 {train|eval|reeval|postero|aggregate}"
REV_SUMMARY_DIR="$SUMMARY_DIR/revision"
POSTERO_ROOT=${POSTERO_ROOT:-/home/viplab/Aagha/Archive/layout/PosterO-CVPR2025}
# LLM run: the env with vLLM; PosterO's pandas dependency comes from a private folder.
POSTERO_PYTHON=${POSTERO_PYTHON:-/home/viplab/anaconda3/envs/VLM_SETUP/bin/python}
POSTERO_PYDEPS=${POSTERO_PYDEPS:-$POSTERO_ROOT/.pydeps}
# Predictor steps (U-Net maps, features, boxes): the env with segmentation_models_pytorch.
PREP_PYTHON=${PREP_PYTHON:-/home/viplab/anaconda3/envs/intentdit/bin/python}
POSTERO_GPU_MEMORY_UTILIZATION=${POSTERO_GPU_MEMORY_UTILIZATION:-0.6}

# dataset|guidance|text|slug|train_config|test_config|lambda1|lambda2|text_mode|disable_boxes
# Ordered by priority: the new main models first.
REV_VARIANTS=(
    "pku|2|1|both_text|pku_v2|pku_v2_anno_test|0|0.05|token|0"
    "pku|2|0|both|pku_v2|pku_v2_anno_test|0|0.05|token|0"
    "cgl|2|1|both_text|cgl|cgl_anno_test|0|0.05|token|0"
    "pku|1|0|intent|pku_v2|pku_v2_anno_test|0|0.05|token|0"
    "pku|2|0|prior_density|pku_prior_density|pku_prior_density_anno_test|0|0.05|token|0"
    "pku|2|0|prior_invsal|pku_prior_invsal|pku_prior_invsal_anno_test|0|0.05|token|0"
    "pku|2|1|both_text_cnt|pku_v2|pku_v2_anno_test|0.1|0.05|token|0"
    "pku|2|1|both_text_noaux|pku_v2|pku_v2_anno_test|0|0|token|0"
    "pku|0|1|saliency_text|pku|pku_anno_test|0|0|token|0"
    "pku|1|1|intent_text|pku_v2|pku_v2_anno_test|0|0.05|token|0"
    "pku|2|1|pooled_text|pku_v2|pku_v2_anno_test|0|0.05|pooled|0"
    "pku|2|1|pixel_map_only_text|pku_v2|pku_v2_anno_test|0|0.05|token|1"
    "pku|3|1|intent_boxes_only_text|pku_v2|pku_v2_anno_test|0|0.05|token|0"
    "cgl|0|1|saliency_text|cgl|cgl_anno_test|0|0|token|0"
    # Image-only IntentDiT without the placement loss (same inputs as "both", lambda2=0):
    # tests the placement-loss claim in the matched image-only setting.
    "pku|2|0|both_noaux|pku_v2|pku_v2_anno_test|0|0|token|0"
    "cgl|2|0|both_noaux|cgl|cgl_anno_test|0|0|token|0"
)
# REV_ONLY="pku:both_noaux cgl:both_noaux" restricts train/eval to the listed variants.
if [[ -n "${REV_ONLY:-}" ]]; then
    selected=()
    for entry in "${REV_VARIANTS[@]}"; do
        IFS='|' read -r dataset _ _ slug _ <<< "$entry"
        [[ " $REV_ONLY " == *" ${dataset}:${slug} "* ]] && selected+=("$entry")
    done
    (( ${#selected[@]} )) || die "REV_ONLY matched no variant: $REV_ONLY"
    REV_VARIANTS=("${selected[@]}")
fi
PROMPT_STYLES=${PROMPT_STYLES:-"basic enhanced advanced spatial rich freeform stress"}

rev_experiment() { printf 'rev_%s_vit_%s_trainseed%s' "$1" "$2" "$3"; }

run_logged() {  # name gpu args...
    local name=$1 gpu=$2; shift 2
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "$@"
    else
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "$@") 2>&1 | tee "$LOG_DIR/${name}.log"
    fi
}

done_for() {  # output_name schedule
    [[ "${FORCE:-0}" != "1" ]] && DDIM_SCHEDULE=$2 result_complete_for "$1"
}

# ----------------------------------------------------------------------------- train
train_task() {
    local task_index=$1 gpu=$2 seed_array
    read -r -a seed_array <<< "$SEEDS"
    local entry=${REV_VARIANTS[$((task_index % ${#REV_VARIANTS[@]}))]}
    local seed=${seed_array[$((task_index / ${#REV_VARIANTS[@]}))]}
    IFS='|' read -r dataset guidance text slug train_cfg _ lambda1 lambda2 text_mode disable_boxes <<< "$entry"
    local experiment
    experiment=$(rev_experiment "$dataset" "$slug" "$seed")
    if [[ "$DRY_RUN" != "1" ]] && has_checkpoint "$dataset" "$experiment"; then
        log "SKIP $experiment (Epoch${FINAL_EPOCH} exists)"; return
    fi
    local args=(
        "$PYTHON_BIN" scripts/train.py --dataset "$dataset" --config "configs/${train_cfg}.yaml"
        --task uncond --v_encoder vit --spatial_guidance "$guidance" --seed "$seed"
        --experiment_name "$experiment" --gpuid 0 --path-profile "$PATH_PROFILE"
        --lambda1 "$lambda1" --lambda2 "$lambda2"
    )
    [[ "$text" == "1" ]] && args+=(--text_control --text-conditioning-mode "$text_mode")
    [[ "$disable_boxes" == "1" ]] && args+=(--disable-spatial-boxes)
    append_training_runtime_args args "$dataset"
    log "TRAIN $experiment on GPU $gpu"
    run_logged "$experiment" "$gpu" "${args[@]}"
}

# ----------------------------------------------------------------------------- eval helpers
eval_image_only() {  # dataset guidance slug seed test_cfg schedule gpu [text_mode disable_boxes]
    local dataset=$1 guidance=$2 slug=$3 seed=$4 test_cfg=$5 schedule=$6 gpu=$7
    local experiment checkpoint name
    experiment=$(rev_experiment "$dataset" "$slug" "$seed")
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || { log "MISSING $experiment"; return; }
    name="rev_${dataset}_vit_${slug}_${schedule}_trainseed${seed}_inferseed${INFERENCE_SEED}"
    done_for "$name" "$schedule" && { log "SKIP $name"; return; }
    local args=(
        "$PYTHON_BIN" scripts/test.py --dataset "$dataset" --anno anno
        --config "configs/${test_cfg}.yaml" --task uncond --check_path "$checkpoint" --v_encoder vit
        --spatial_guidance "$guidance" --seed "$INFERENCE_SEED" --ddim_num_steps "$DDIM_STEPS"
        --ddim_schedule "$schedule" --experiment_name "$name" --gpuid 0 --path-profile "$PATH_PROFILE"
        --save-test-output auto --protocol image_only --no-render
    )
    run_logged "$name" "$gpu" "${args[@]}"
}

prompt_csv_for() {
    case "$2" in
        basic|enhanced|advanced|spatial) printf '%s/data/dataset/%s/split/csv/test_with_prompts_%s.csv' "$ACTIVE_ROOT" "$1" "$2" ;;
        rich) printf '%s/data/dataset/%s/split/csv/test_with_rich_prompts.csv' "$ACTIVE_ROOT" "$1" ;;
        freeform) printf '%s/data/prompts/free_form_%s.csv' "$ACTIVE_ROOT" "$1" ;;
        stress) printf '%s/data/prompts/stress_%s.csv' "$ACTIVE_ROOT" "$1" ;;
        edit_base) printf '%s/data/prompts/edit_base_%s.csv' "$ACTIVE_ROOT" "$1" ;;
        edit_changed) printf '%s/data/prompts/edit_changed_%s.csv' "$ACTIVE_ROOT" "$1" ;;
    esac
}

eval_prompt() {  # dataset guidance slug seed test_cfg text_mode disable_boxes style gpu [tag extra-args...]
    local dataset=$1 guidance=$2 slug=$3 seed=$4 test_cfg=$5 text_mode=$6 disable_boxes=$7 style=$8 gpu=$9
    local tag=${10:-} ; shift 10 || shift $#
    local experiment checkpoint name protocol=oracle_prompt
    experiment=$(rev_experiment "$dataset" "$slug" "$seed")
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || { log "MISSING $experiment"; return; }
    name="rev_prompt_${dataset}_vit_${slug}_${style}${tag}_trainseed${seed}_inferseed${INFERENCE_SEED}"
    done_for "$name" "$DDIM_SCHEDULE" && { log "SKIP $name"; return; }
    [[ "$style" == "freeform" ]] && protocol=freeform_prompt
    local args=(
        "$PYTHON_BIN" scripts/test.py --dataset "$dataset" --anno anno --config "configs/${test_cfg}.yaml"
        --task uncond --check_path "$checkpoint" --v_encoder vit --spatial_guidance "$guidance"
        --text_control --text-conditioning-mode "$text_mode" --prompts-csv "$(prompt_csv_for "$dataset" "$style")"
        --seed "$INFERENCE_SEED" --ddim_num_steps "$DDIM_STEPS" --ddim_schedule "$DDIM_SCHEDULE"
        --experiment_name "$name" --save-test-output auto --gpuid 0 --path-profile "$PATH_PROFILE"
        --protocol "$protocol" --no-render "$@"
    )
    [[ "$disable_boxes" == "1" ]] && args+=(--disable-spatial-boxes)
    [[ "$style" == "spatial" ]] && args+=(--spatial-metrics)
    [[ "$style" == freeform || "$style" == stress || "$style" == edit_* ]] && args+=(--prompt-subset-only --spatial-metrics)
    run_logged "$name" "$gpu" "${args[@]}"
}

# ----------------------------------------------------------------------------- eval
eval_task() {
    local task_index=$1 gpu=$2 seed_array
    read -r -a seed_array <<< "$SEEDS"
    local entry=${REV_VARIANTS[$((task_index % ${#REV_VARIANTS[@]}))]}
    local seed=${seed_array[$((task_index / ${#REV_VARIANTS[@]}))]}
    IFS='|' read -r dataset guidance text slug _ test_cfg _ _ text_mode disable_boxes <<< "$entry"
    if [[ "$text" == "0" ]]; then
        eval_image_only "$dataset" "$guidance" "$slug" "$seed" "$test_cfg" linear "$gpu"
        eval_image_only "$dataset" "$guidance" "$slug" "$seed" "$test_cfg" cosine "$gpu"
        return
    fi
    local styles="basic"
    [[ "$slug" == "both_text" ]] && styles=$PROMPT_STYLES
    [[ "$slug" == "saliency_text" || "$slug" == "both_text_cnt" || "$slug" == "both_text_noaux" ]] && styles="basic spatial freeform"
    for style in $styles; do
        eval_prompt "$dataset" "$guidance" "$slug" "$seed" "$test_cfg" "$text_mode" "$disable_boxes" "$style" "$gpu"
    done
    if [[ "$slug" == "both_text" ]]; then
        # Inference-time decomposition of the two prompt pathways (free-form prompts).
        eval_prompt "$dataset" "$guidance" "$slug" "$seed" "$test_cfg" token 0 freeform "$gpu" _noparser --disable-text-spatial-parser
        eval_prompt "$dataset" "$guidance" "$slug" "$seed" "$test_cfg" token 0 freeform "$gpu" _keysonly --text-guidance-scale 0
        # Text-guidance sweep on Basic prompts.
        for scale in 0 0.5 1.5 2; do
            eval_prompt "$dataset" "$guidance" "$slug" "$seed" "$test_cfg" token 0 basic "$gpu" "_scale${scale/./p}" --text-guidance-scale "$scale"
        done
        if [[ "$seed" == "1" ]]; then
            for version in edit_base edit_changed; do
                eval_prompt "$dataset" "$guidance" "$slug" 1 "$test_cfg" token 0 "$version" "$gpu"
            done
            for inference_seed in 2 3 4 5; do  # diversity: 5 samples per free-form condition
                INFERENCE_SEED=$inference_seed eval_prompt "$dataset" "$guidance" "$slug" 1 "$test_cfg" token 0 freeform "$gpu"
            done
        fi
    fi
}

# ----------------------------------------------------------------------------- reeval
# Existing IVC checkpoints whose conditioning data were correct, re-sampled with S_L
# so that per-image outputs exist for paired bootstrap against IntentDiT.
REEVAL=(
    "pku|0|saliency|pku_anno_test"
    "cgl|0|saliency|cgl_anno_test"
    "cgl|2|both|cgl_anno_test"
    "cgl|1|intent|cgl_anno_test"
)
reeval_task() {
    local task_index=$1 gpu=$2 seed_array
    read -r -a seed_array <<< "$SEEDS"
    local entry=${REEVAL[$((task_index % ${#REEVAL[@]}))]}
    local seed=${seed_array[$((task_index / ${#REEVAL[@]}))]}
    IFS='|' read -r dataset guidance slug test_cfg <<< "$entry"
    local experiment checkpoint name
    experiment=$(experiment_name "$dataset" vit "$slug" "$seed")
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || { log "MISSING $experiment"; return; }
    name="rev_${dataset}_vit_${slug}_linear_trainseed${seed}_inferseed${INFERENCE_SEED}"
    done_for "$name" linear && { log "SKIP $name"; return; }
    local args=(
        "$PYTHON_BIN" scripts/test.py --dataset "$dataset" --anno anno
        --config "configs/${test_cfg}.yaml" --task uncond --check_path "$checkpoint" --v_encoder vit
        --spatial_guidance "$guidance" --seed "$INFERENCE_SEED" --ddim_num_steps "$DDIM_STEPS"
        --ddim_schedule linear --experiment_name "$name" --gpuid 0 --path-profile "$PATH_PROFILE"
        --save-test-output auto --protocol image_only --no-render
    )
    run_logged "$name" "$gpu" "${args[@]}"
}

# ----------------------------------------------------------------------------- postero
# PosterO's intermediate files (predictor maps -> boxes, retrieval features) are not on
# disk any more; regenerate them with PosterO's own design_intent_detect pipeline
# (same as its test.sh, one GPU) before the LLM run.
postero_prep() {
    local gpu=$1 dataset=$2 epoch=$3
    local ckpt="${dataset}_128_1e-06_none/ckpt/design_intent_${dataset}_epoch${epoch}.pth"
    local result="$POSTERO_ROOT/design_intent_detect/${dataset}_128_1e-06_none/result/design_intent_${dataset}_epoch${epoch}"
    [[ -f "$result/design_intent_bbox_valid.pt" && -d "$result/${dataset}_features/train" ]] && return
    local torchrun=${PREP_PYTHON%/python}/torchrun
    local common=(--dataset_root "$POSTERO_ROOT/prepared_dataset" --dataset "$dataset" --infer --infer_ckpt "$ckpt")
    # PosterO's unannotated CGL test split is not available locally and is not needed here.
    local steps=("--infer_csv train" "--extract --extract_split valid --infer_csv train" "--extract --extract_split train --infer_csv train")
    [[ "$dataset" == pku ]] && steps+=("--infer_csv test" "--extract --extract_split test --infer_csv test")
    for step in "${steps[@]}"; do
        run_command bash -c "cd '$POSTERO_ROOT/design_intent_detect' && CUDA_VISIBLE_DEVICES=$gpu '$torchrun' --standalone --nnodes=1 --nproc-per-node=1 main.py ${common[*]} $step"
    done
    run_command bash -c "cd '$POSTERO_ROOT/design_intent_detect' && '$PREP_PYTHON' - <<'PY'
import os, torch
from map2box import getDesignIntentBox
root = '$POSTERO_ROOT/prepared_dataset/$dataset/annotation'
dm_root = '${dataset}_128_1e-06_none/result/design_intent_${dataset}_epoch${epoch}'
for subsplit in ('valid', 'train'):
    getDesignIntentBox(dm_root, root, split='train', subsplit=subsplit, kernel_n=37, preview=0)
if os.path.isdir(os.path.join(dm_root, 'test')):
    getDesignIntentBox(dm_root, root, split='test', kernel_n=37, preview=0)
else:
    torch.save([], os.path.join(dm_root, 'design_intent_bbox_test.pt'))
PY"
}

run_postero() {
    local gpu=${GPU_IDS%%,*} dataset epoch bbox_dir
    for dataset in pku cgl; do
        epoch=$([[ "$dataset" == pku ]] && echo 100 || echo 35)
        postero_prep "$gpu" "$dataset" "$epoch"
        bbox_dir="design_intent_detect/${dataset}_128_1e-06_none/result/design_intent_${dataset}_epoch${epoch}"
        local out="$ACTIVE_ROOT/other_baselines/postero_prompted/postero_prompted_${dataset}.pt"
        local args=(
            "$POSTERO_PYTHON" scripts/run_prompted_infer.py
            --requests-json "$ACTIVE_ROOT/other_baselines/postero_prompted/requests_${dataset}.json"
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
        log "POSTERO-PROMPTED $dataset on GPU $gpu"
        if [[ "$DRY_RUN" == "1" ]]; then
            print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
            continue
        fi
        [[ -f "$out" && "${FORCE:-0}" != "1" ]] || (cd "$POSTERO_ROOT" && env CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$POSTERO_PYDEPS" POSTERO_GPU_MEMORY_UTILIZATION="$POSTERO_GPU_MEMORY_UTILIZATION" "${args[@]}") \
            2>&1 | tee "$LOG_DIR/rev_postero_prompted_${dataset}.log"
        run_command "$PYTHON_BIN" "$CODE_DIR/scripts/import_postero_predictions.py" \
            --input "$out" --output "$ACTIVE_ROOT/other_baselines/standardized/postero_prompted_${dataset}_freeform_subset.pt" \
            --dataset "$dataset" --split valid --selection official --postero-code-root "$POSTERO_ROOT" \
            --subset-csv "$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv" \
            $([[ "$dataset" == cgl ]] && echo --name-map-csv "$ACTIVE_ROOT/data/dataset/cgl/split/csv/test.csv")
    done
}

# ----------------------------------------------------------------------------- aggregate
run_aggregate() {
    mkdir -p "$REV_SUMMARY_DIR"
    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/aggregate_training_seeds.py" --input-dir "$METRIC_DIR" \
        --inference-seed "$INFERENCE_SEED" --include-prefix rev_ --output "$REV_SUMMARY_DIR/rev_seed_summary.json"
    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/aggregate_training_seeds.py" --input-dir "$METRIC_DIR" \
        --inference-seed "$INFERENCE_SEED" --include-prefix rev_prompt_ --output "$REV_SUMMARY_DIR/rev_prompt_seed_summary.json"
    local dataset baseline tag
    for dataset in pku cgl; do
        # Score both external references on the free-form subset with the shared evaluator.
        for baseline in postero postero_prompted; do
            local tensor="$ACTIVE_ROOT/other_baselines/standardized/${baseline}_${dataset}_freeform_subset.pt"
            [[ "$DRY_RUN" == "1" || -f "$tensor" ]] || { log "MISSING $tensor"; continue; }
            run_command "$PYTHON_BIN" "$CODE_DIR/scripts/evaluate_saved_predictions.py" \
                --predictions "$tensor" --dataset "$dataset" --anno anno \
                --experiment-name "rev_${baseline}_${dataset}_freeform" --protocol text_baseline \
                --text-control --spatial-metrics --prompts-csv "$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv" \
                --path-profile "$PATH_PROFILE" --output-dir "$METRIC_DIR"
            # Manual-reference scores for the new main model against this reference.
            local out="$REV_SUMMARY_DIR/manual_${baseline}_${dataset}"
            mkdir -p "$out"
            run_command "$PYTHON_BIN" "$CODE_DIR/scripts/evaluate_manual_freeform_reference.py" \
                --dataset "$dataset" --audit-csv "$SUMMARY_DIR/fixed_parser_audit_${dataset}.csv" \
                --metric-dir "$METRIC_DIR" --baseline-predictions "$tensor" \
                --baseline-per-image "$METRIC_DIR/rev_${baseline}_${dataset}_freeform_per_image.csv" \
                --prediction-prefix "rev_prompt_{dataset}_vit_both_text_freeform" \
                --seeds $SEEDS --inference-seed "$INFERENCE_SEED" --output-dir "$out"
            local seed_paths=() seed
            for seed in $SEEDS; do
                seed_paths+=("$out/manual_freeform_${dataset}_intentdit_seed${seed}_all_per_image.csv")
            done
            run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_seed_image_bootstrap.py" \
                --method-a "${seed_paths[@]}" \
                --method-b "$out/manual_freeform_${dataset}_external_text_baseline_all_per_image.csv" \
                --name-a intentdit --name-b "$baseline" --iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
                --output "$REV_SUMMARY_DIR/paired_manual_freeform_${dataset}_vs_${baseline}.json"
        done
        # Matched image-only comparison under S_L: IntentDiT vs retrained LayoutDiT configuration.
        local a=() b=() seed
        for seed in $SEEDS; do
            a+=("$METRIC_DIR/rev_${dataset}_vit_both_linear_trainseed${seed}_inferseed${INFERENCE_SEED}_per_image.csv")
            if [[ "$dataset" == pku ]]; then
                b+=("$METRIC_DIR/rev_pku_vit_saliency_linear_trainseed${seed}_inferseed${INFERENCE_SEED}_per_image.csv")
            else
                b+=("$METRIC_DIR/rev_cgl_vit_saliency_linear_trainseed${seed}_inferseed${INFERENCE_SEED}_per_image.csv")
            fi
        done
        for seed_index in "${!a[@]}"; do
            run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_bootstrap.py" \
                --method-a "${a[$seed_index]}" --method-b "${b[$seed_index]}" \
                --name-a intentdit --name-b layoutdit_config --iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
                --output "$REV_SUMMARY_DIR/paired_linear_${dataset}_both_vs_saliency_seed$((seed_index + 1)).json"
        done
        # Placement-loss ablation in the image-only setting: IntentDiT vs the same model with lambda2=0.
        for schedule in linear cosine; do
            for seed in $SEEDS; do
                local noaux="$METRIC_DIR/rev_${dataset}_vit_both_noaux_${schedule}_trainseed${seed}_inferseed${INFERENCE_SEED}_per_image.csv"
                [[ "$DRY_RUN" == "1" || -f "$noaux" ]] || { log "MISSING $noaux"; continue; }
                run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_bootstrap.py" \
                    --method-a "$METRIC_DIR/rev_${dataset}_vit_both_${schedule}_trainseed${seed}_inferseed${INFERENCE_SEED}_per_image.csv" \
                    --method-b "$noaux" --name-a intentdit --name-b intentdit_lambda2_0 \
                    --iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
                    --output "$REV_SUMMARY_DIR/paired_${schedule}_${dataset}_both_vs_noaux_seed${seed}.json"
            done
        done
    done
}

read -r -a seed_array <<< "$SEEDS"
case "$PHASE" in
    train) run_task_workers train_task "$(( ${#REV_VARIANTS[@]} * ${#seed_array[@]} ))" ;;
    eval) run_task_workers eval_task "$(( ${#REV_VARIANTS[@]} * ${#seed_array[@]} ))" ;;
    reeval) run_task_workers reeval_task "$(( ${#REEVAL[@]} * ${#seed_array[@]} ))" ;;
    postero) run_postero ;;
    aggregate) run_aggregate ;;
    *) die "unknown phase: $PHASE" ;;
esac
log "Phase $PHASE complete"
