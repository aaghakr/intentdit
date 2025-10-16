# Demo Configuration System

This directory contains the configuration system for the Gradio demo app, which manages checkpoint paths and model weights for all 24 experiments.

## Files

- `demo_config.yaml` - Main configuration file defining all experiment paths
- `config_loader.py` - Python module for loading and managing configuration
- `test_config.py` - Test script to validate configuration
- `app.py` - Updated Gradio demo app with configuration integration

## Configuration Structure

### Base Paths
```yaml
base_checkpoint_dir: "/path/to/checkpoints"
base_model_weights_dir: "/path/to/model_weights"
```

### Model Weights
```yaml
model_weights:
  saliency:
    basnet: "/path/to/basnet.pth"
    isnet: "/path/to/isnet.pth"
    u2net: "/path/to/u2net.pth"
  intent:
    pku: "/path/to/design_intent_pku_epoch100.pth"
    cgl: "/path/to/design_intent_cgl_epoch35.pth"
```

### Experiment Configuration
Each experiment is defined with:
- `checkpoint`: Specific checkpoint path (null = auto-discover latest)
- `encoder`: "vit" or "swin"
- `spatial_guidance`: "saliency", "intent", or "both"
- `text_control`: true or false

## Usage

### 1. Test Configuration
```bash
cd demo
python test_config.py
```

### 2. Run Demo
```bash
cd demo
python app.py
```

### 3. Check Experiment Status
The demo will show which experiments are available:
- ✅ Available (checkpoint found)
- ❌ Not available (checkpoint missing)

## Checkpoint Discovery

The system automatically discovers checkpoints using this pattern:
```
{base_checkpoint_dir}/{dataset}/{experiment_name}/**/Epoch*_cgbdm_weights.pth
```

For example:
- `data/checkpoints/pku/pku_vit_saliency/12_25_1430/Epoch50_cgbdm_weights.pth`
- `data/checkpoints/cgl/cgl_swin_intent_text/12_26_0900/Epoch100_cgbdm_weights.pth`

## Experiment Naming Convention

Experiments follow this naming pattern:
```
{dataset}_{encoder}_{spatial_guidance}[_text]
```

Examples:
- `pku_vit_saliency` - PKU dataset, ViT encoder, saliency guidance
- `cgl_swin_intent_text` - CGL dataset, Swin encoder, intent guidance with text control

## Configuration Features

### Auto-Discovery
- If `checkpoint: null`, system finds latest checkpoint automatically
- Sorts by modification time to get most recent checkpoint
- Logs warnings for missing experiments

### Fallback Behavior
- Shows error message if experiment not available
- Provides helpful debugging information
- Lists available experiments in demo interface

### Model Weight Management
- Centralized model weight paths
- Dataset-specific intent detection models
- Saliency detection model selection

## Troubleshooting

### Common Issues

1. **No experiments available**
   - Check `base_checkpoint_dir` path
   - Ensure experiments are trained
   - Verify naming convention

2. **Model weights not found**
   - Check `base_model_weights_dir` path
   - Verify model weight files exist
   - Check file permissions

3. **Configuration errors**
   - Run `python test_config.py` to validate
   - Check YAML syntax
   - Verify file paths

### Debug Information

The demo shows detailed information for each experiment:
- Checkpoint path
- Model weight paths
- Experiment configuration
- Availability status

## Adding New Experiments

1. Update `demo_config.yaml` with new experiment
2. Follow naming convention
3. Set `checkpoint: null` for auto-discovery
4. Run `test_config.py` to validate

## Integration with Training

The configuration system integrates with your training pipeline:
- Checkpoints are saved to configured directories
- Experiment names match training scripts
- Auto-discovery finds latest checkpoints
- Status updates in real-time
