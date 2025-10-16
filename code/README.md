# Intent-Aware Layout Generation - Experiment Commands

This document contains all the training and testing commands for the Intent-Aware Layout Generation project.

## Table of Contents
- [Training Commands](#training-commands)
- [Testing Commands](#testing-commands)
- [Single Image Generation](#single-image-generation)
- [Intent Detection Commands](#intent-detection-commands)
- [Demo Interface](#demo-interface)
- [Data Preprocessing](#data-preprocessing)

## Training Commands

### Basic Training
```bash
# Train PKU dataset - Unconditional
python scripts/train.py --dataset pku --task uncond --gpuid 0 --experiment_name "pku_baseline"

# Train PKU dataset - Conditional
python scripts/train.py --dataset pku --task c --gpuid 0 --experiment_name "pku_conditional"

# Train PKU dataset - Conditional with Height
python scripts/train.py --dataset pku --task cwh --gpuid 0 --experiment_name "pku_cwh"

# Train PKU dataset - Complete Conditional
python scripts/train.py --dataset pku --task complete --gpuid 0 --experiment_name "pku_complete"

# Train CGL dataset - Unconditional
python scripts/train.py --dataset cgl --task uncond --gpuid 0 --experiment_name "cgl_baseline"

# Train CGL dataset - Conditional
python scripts/train.py --dataset cgl --task c --gpuid 0 --experiment_name "cgl_conditional"

# Train CGL dataset - Complete Conditional
python scripts/train.py --dataset cgl --task complete --gpuid 0 --experiment_name "cgl_complete"
```

### Training with Different Visual Encoders
```bash
# Using ViT encoder (default)
python scripts/train.py --dataset pku --task uncond --v_encoder vit --gpuid 0 --experiment_name "pku_vit"

# Using Swin encoder
python scripts/train.py --dataset pku --task uncond --v_encoder swin --gpuid 0 --experiment_name "pku_swin"

# Using ConvNext encoder
python scripts/train.py --dataset pku --task uncond --v_encoder convnext --gpuid 0 --experiment_name "pku_convnext"
```

### Training with Spatial Guidance and Text Control

#### PKU Dataset - ViT Encoder
```bash
# Original saliency only
python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 0 --gpuid 0 --experiment_name "pku_vit_saliency"

# Intent only
python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 1 --gpuid 0 --experiment_name "pku_vit_intent"

# Sal + Intent
python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 2 --gpuid 0 --experiment_name "pku_vit_both"

# Sal + Text
python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 0 --text_control --gpuid 0 --experiment_name "pku_vit_saliency_text"

# Intent + Text
python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 1 --text_control --gpuid 0 --experiment_name "pku_vit_intent_text"

# Sal + Intent + Text
python scripts/train.py --dataset pku --task uncond --v_encoder vit --spatial_guidance 2 --text_control --gpuid 0 --experiment_name "pku_vit_both_text"
```

#### PKU Dataset - Swin Encoder
```bash
# Original saliency only
python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 0 --gpuid 0 --experiment_name "pku_swin_saliency"

# Intent only
python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 1 --gpuid 0 --experiment_name "pku_swin_intent"

# Sal + Intent
python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 2 --gpuid 0 --experiment_name "pku_swin_both"

# Sal + Text
python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 0 --text_control --gpuid 0 --experiment_name "pku_swin_saliency_text"

# Intent + Text
python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 1 --text_control --gpuid 0 --experiment_name "pku_swin_intent_text"

# Sal + Intent + Text
python scripts/train.py --dataset pku --task uncond --v_encoder swin --spatial_guidance 2 --text_control --gpuid 0 --experiment_name "pku_swin_both_text"
```

#### CGL Dataset - ViT Encoder
```bash
# Original saliency only
python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 0 --gpuid 0 --experiment_name "cgl_vit_saliency"

# Intent only
python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 1 --gpuid 0 --experiment_name "cgl_vit_intent"

# Sal + Intent
python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 2 --gpuid 0 --experiment_name "cgl_vit_both"

# Sal + Text
python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 0 --text_control --gpuid 0 --experiment_name "cgl_vit_saliency_text"

# Intent + Text
python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 1 --text_control --gpuid 0 --experiment_name "cgl_vit_intent_text"

# Sal + Intent + Text
python scripts/train.py --dataset cgl --task uncond --v_encoder vit --spatial_guidance 2 --text_control --gpuid 0 --experiment_name "cgl_vit_both_text"
```

#### CGL Dataset - Swin Encoder
```bash
# Original saliency only
python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 0 --gpuid 0 --experiment_name "cgl_swin_saliency"

# Intent only
python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 1 --gpuid 0 --experiment_name "cgl_swin_intent"

# Sal + Intent
python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 2 --gpuid 0 --experiment_name "cgl_swin_both"

# Sal + Text
python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 0 --text_control --gpuid 0 --experiment_name "cgl_swin_saliency_text"

# Intent + Text
python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 1 --text_control --gpuid 0 --experiment_name "cgl_swin_intent_text"

# Sal + Intent + Text
python scripts/train.py --dataset cgl --task uncond --v_encoder swin --spatial_guidance 2 --text_control --gpuid 0 --experiment_name "cgl_swin_both_text"
```

## Testing Commands

### Basic Testing
```bash
# Test PKU dataset - Unconditional (unannotated)
python scripts/test.py --dataset pku --anno unanno --task uncond --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test PKU dataset - Unconditional (annotated)
python scripts/test.py --dataset pku --anno anno --task uncond --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test PKU dataset - Conditional
python scripts/test.py --dataset pku --anno anno --task c --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test PKU dataset - Conditional with Height
python scripts/test.py --dataset pku --anno anno --task cwh --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test PKU dataset - Complete Conditional
python scripts/test.py --dataset pku --anno anno --task complete --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test PKU dataset - Refinement
python scripts/test.py --dataset pku --anno anno --task refine --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test CGL dataset - Unconditional (unannotated)
python scripts/test.py --dataset cgl --anno unanno --task uncond --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test CGL dataset - Unconditional (annotated)
python scripts/test.py --dataset cgl --anno anno --task uncond --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test CGL dataset - Conditional
python scripts/test.py --dataset cgl --anno anno --task c --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test CGL dataset - Complete Conditional
python scripts/test.py --dataset cgl --anno anno --task complete --gpuid 0 --check_path "path/to/checkpoint.pth"

# Test CGL dataset - Refinement
python scripts/test.py --dataset cgl --anno anno --task refine --gpuid 0 --check_path "path/to/checkpoint.pth"
```

### Testing with Different Visual Encoders
```bash
# Using ViT encoder
python scripts/test.py --dataset pku --anno unanno --task uncond --v_encoder vit --gpuid 0 --check_path "path/to/checkpoint.pth"

# Using Swin encoder
python scripts/test.py --dataset pku --anno unanno --task uncond --v_encoder swin --gpuid 0 --check_path "path/to/checkpoint.pth"

# Using ConvNext encoder
python scripts/test.py --dataset pku --anno unanno --task uncond --v_encoder convnext --gpuid 0 --check_path "path/to/checkpoint.pth"
```

### Testing with Spatial Guidance
```bash
# Saliency guidance only
python scripts/test.py --dataset pku --anno unanno --task uncond --spatial_guidance 0 --gpuid 0 --check_path "path/to/checkpoint.pth"

# Intent map guidance only
python scripts/test.py --dataset pku --anno unanno --task uncond --spatial_guidance 1 --gpuid 0 --check_path "path/to/checkpoint.pth"

# Both saliency and intent map guidance
python scripts/test.py --dataset pku --anno unanno --task uncond --spatial_guidance 2 --gpuid 0 --check_path "path/to/checkpoint.pth"
```

## Single Image Generation

### Generate Layout for Single Image
```bash
# Generate layout for PKU style
python scripts/run_single_image.py --render_style pku --image_path "path/to/image.jpg" --check_path "path/to/checkpoint.pth" --gpuid 0 --seed 1

# Generate layout for CGL style
python scripts/run_single_image.py --render_style cgl --image_path "path/to/image.jpg" --check_path "path/to/checkpoint.pth" --gpuid 0 --seed 1
```


## Configuration Files

The project uses YAML configuration files located in `configs/`:

- `pku.yaml` - PKU dataset training configuration
- `cgl.yaml` - CGL dataset training configuration
- `pku_anno_test.yaml` - PKU annotated test configuration
- `pku_unanno_test.yaml` - PKU unannotated test configuration
- `cgl_anno_test.yaml` - CGL annotated test configuration
- `cgl_unanno_test.yaml` - CGL unannotated test configuration

## Output Directories

### Training Outputs
- **Checkpoints**: `data/checkpoints/{dataset}/{experiment_name}/{datetime}/`
- **TensorBoard Logs**: `runs/`
- **Image Name Orders**: `data/output/ptfile/image_name_order/`

### Testing Outputs
- **Generated Images**: `data/output/image/{dataset}_{annotation_type}_test/`
- **Metrics**: Computed and logged during testing

## Example Experiment Workflow

```bash
# 1. Train a model
python scripts/train.py --dataset pku --task uncond --experiment_name "pku_baseline_v1" --gpuid 0

# 2. Test the model
python scripts/test.py --dataset pku --anno unanno --task uncond --check_path "data/checkpoints/pku/pku_baseline_v1/12_25_1430/Epoch400_cgbdm_weights.pth" --gpuid 0

# 3. Generate single image
python scripts/run_single_image.py --render_style pku --image_path "path/to/test_image.jpg" --check_path "data/checkpoints/pku/pku_baseline_v1/12_25_1430/Epoch400_cgbdm_weights.pth" --gpuid 0

# 4. Launch demo
cd demo && python app.py
```

## Experimental Design Summary

### **Total Experiments: 24**

Our experimental design systematically evaluates the contribution of each component:

#### **Spatial Guidance Combinations (6):**
1. **sal** - Original saliency only
2. **intent** - Intent detection only  
3. **sal + intent** - Both saliency and intent
4. **sal + text** - Saliency + text control
5. **intent + text** - Intent + text control
6. **sal + intent + text** - All three components

#### **Visual Encoders (2):**
- **ViT** - Original Vision Transformer
- **Swin** - Swin Transformer

#### **Datasets (2):**
- **PKU** - 4 element classes (Text, Logo, Underlay, Embellishment)
- **CGL** - 5 element classes (Text, Logo, Underlay, Embellishment, Decoration)

### **Experiment Naming Convention:**
`{dataset}_{encoder}_{spatial_guidance}_{text_control}`

Examples:
- `pku_vit_saliency` - PKU dataset, ViT encoder, saliency only
- `cgl_swin_both_text` - CGL dataset, Swin encoder, saliency + intent + text

### **Testing Commands:**
Each training experiment requires corresponding testing commands with the same parameters:

```bash
# Example: Test the PKU ViT saliency experiment
python scripts/test.py --dataset pku --anno unanno --task uncond --v_encoder vit --spatial_guidance 0 --gpuid 0 --check_path "data/checkpoints/pku/pku_vit_saliency/12_25_1430/Epoch400_cgbdm_weights.pth"
```

## Notes

- All commands should be run from the `code/` directory
- GPU ID can be changed using the `--gpuid` parameter
- Experiment names help organize different training runs
- Checkpoints are saved every 5 epochs after epoch 400
- TensorBoard logs are saved in the `runs/` directory
- Generated images are saved in `data/output/image/`
- **Text Control Parameter**: You'll need to add a `--text_control` flag to your training script if it doesn't exist yet
- **Resource Planning**: With 24 experiments, consider running them in parallel if you have multiple GPUs