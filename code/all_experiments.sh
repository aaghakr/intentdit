#!/bin/bash

# Intent-Aware Layout Generation - Complete Experimental Suite
# Total: 24 experiments
# GPU 0: PKU ViT experiments (6) + PKU Swin experiments (6) = 12 experiments
# GPU 1: CGL ViT experiments (6)
# GPU 2: CGL Swin experiments (6)

echo "Starting Intent-Aware Layout Generation Experiments"
echo "Total experiments: 24"
echo "GPU 0: PKU experiments (12)"
echo "GPU 1: CGL ViT experiments (6)" 
echo "GPU 2: CGL Swin experiments (6)"
echo "=========================================="

# Function to run experiment with error handling
run_experiment() {
    local cmd="$1"
    local exp_name="$2"
    local gpu="$3"
    
    echo "Starting experiment: $exp_name on GPU $gpu"
    echo "Command: $cmd"
    echo "----------------------------------------"
    
    # Run the experiment
    $cmd
    
    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: $exp_name completed"
    else
        echo "❌ FAILED: $exp_name failed"
    fi
    echo "----------------------------------------"
}

# GPU 0: PKU Dataset Experiments (12 total)
echo "🚀 Starting PKU experiments on GPU 0..."

# PKU ViT Experiments (6)
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 0 --gpuid 0 --experiment_name 'pku_vit_saliency'" "pku_vit_saliency" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 1 --gpuid 0 --experiment_name 'pku_vit_intent'" "pku_vit_intent" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 2 --gpuid 0 --experiment_name 'pku_vit_both'" "pku_vit_both" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 0 --text_control --gpuid 0 --experiment_name 'pku_vit_saliency_text'" "pku_vit_saliency_text" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 1 --text_control --gpuid 0 --experiment_name 'pku_vit_intent_text'" "pku_vit_intent_text" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 2 --text_control --gpuid 0 --experiment_name 'pku_vit_both_text'" "pku_vit_both_text" 0

# PKU Swin Experiments (6)
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 0 --gpuid 0 --experiment_name 'pku_swin_saliency'" "pku_swin_saliency" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 1 --gpuid 0 --experiment_name 'pku_swin_intent'" "pku_swin_intent" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 2 --gpuid 0 --experiment_name 'pku_swin_both'" "pku_swin_both" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 0 --text_control --gpuid 0 --experiment_name 'pku_swin_saliency_text'" "pku_swin_saliency_text" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 1 --text_control --gpuid 0 --experiment_name 'pku_swin_intent_text'" "pku_swin_intent_text" 0
run_experiment "python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 2 --text_control --gpuid 0 --experiment_name 'pku_swin_both_text'" "pku_swin_both_text" 0

echo "✅ PKU experiments completed on GPU 0"

# GPU 1: CGL ViT Experiments (6)
echo "🚀 Starting CGL ViT experiments on GPU 1..."

run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 0 --gpuid 1 --experiment_name 'cgl_vit_saliency'" "cgl_vit_saliency" 1
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 1 --gpuid 1 --experiment_name 'cgl_vit_intent'" "cgl_vit_intent" 1
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 2 --gpuid 1 --experiment_name 'cgl_vit_both'" "cgl_vit_both" 1
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 0 --text_control --gpuid 1 --experiment_name 'cgl_vit_saliency_text'" "cgl_vit_saliency_text" 1
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 1 --text_control --gpuid 1 --experiment_name 'cgl_vit_intent_text'" "cgl_vit_intent_text" 1
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 2 --text_control --gpuid 1 --experiment_name 'cgl_vit_both_text'" "cgl_vit_both_text" 1

echo "✅ CGL ViT experiments completed on GPU 1"

# GPU 2: CGL Swin Experiments (6)
echo "🚀 Starting CGL Swin experiments on GPU 2..."

run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 0 --gpuid 2 --experiment_name 'cgl_swin_saliency'" "cgl_swin_saliency" 2
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 1 --gpuid 2 --experiment_name 'cgl_swin_intent'" "cgl_swin_intent" 2
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 2 --gpuid 2 --experiment_name 'cgl_swin_both'" "cgl_swin_both" 2
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 0 --text_control --gpuid 2 --experiment_name 'cgl_swin_saliency_text'" "cgl_swin_saliency_text" 2
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 1 --text_control --gpuid 2 --experiment_name 'cgl_swin_intent_text'" "cgl_swin_intent_text" 2
run_experiment "python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 2 --text_control --gpuid 2 --experiment_name 'cgl_swin_both_text'" "cgl_swin_both_text" 2

echo "✅ CGL Swin experiments completed on GPU 2"

echo "=========================================="
echo "🎉 ALL EXPERIMENTS COMPLETED!"
echo "Total: 24 experiments"
echo "GPU 0: 12 PKU experiments"
echo "GPU 1: 6 CGL ViT experiments"
echo "GPU 2: 6 CGL Swin experiments"
echo "=========================================="

# Optional: Generate summary report
echo "📊 Generating experiment summary..."
echo "Checkpoint directories:"
echo "- PKU: data/checkpoints/pku/"
echo "- CGL: data/checkpoints/cgl/"
echo ""
echo "TensorBoard logs: runs/"
echo "Generated images: data/output/image/"