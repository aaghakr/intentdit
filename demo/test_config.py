#!/usr/bin/env python3
"""
Test script for demo configuration
Validates checkpoint paths and model weights
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from config_loader import (
    validate_config,
    list_available_experiments,
    get_experiment_info,
    get_model_weights
)

def main():
    print("🧪 Demo Configuration Test")
    print("=" * 50)
    
    # Validate configuration
    print("\n1. Validating Configuration...")
    is_valid, errors = validate_config()
    
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
    
    # List available experiments
    print("\n2. Available Experiments:")
    print("-" * 30)
    experiments = list_available_experiments()
    
    pku_available = 0
    cgl_available = 0
    
    for exp_name, exp_info in experiments.items():
        status = "✅" if exp_info["checkpoint_available"] else "❌"
        dataset = exp_info["dataset"].upper()
        encoder = exp_info["encoder"].upper()
        guidance = exp_info["spatial_guidance"]
        text_control = "Text" if exp_info["text_control"] else "No Text"
        
        print(f"{status} {exp_name}: {dataset} | {encoder} | {guidance} | {text_control}")
        
        if exp_info["checkpoint_available"]:
            if exp_info["dataset"] == "pku":
                pku_available += 1
            else:
                cgl_available += 1
    
    print(f"\n📊 Summary:")
    print(f"  PKU Experiments Available: {pku_available}/12")
    print(f"  CGL Experiments Available: {cgl_available}/12")
    print(f"  Total Available: {pku_available + cgl_available}/24")
    
    # Test model weights
    print("\n3. Model Weights:")
    print("-" * 30)
    
    saliency_weights = get_model_weights("saliency")
    intent_weights_pku = get_model_weights("intent", "pku")
    intent_weights_cgl = get_model_weights("intent", "cgl")
    
    print(f"Saliency Detection: {saliency_weights}")
    print(f"Intent Detection (PKU): {intent_weights_pku}")
    print(f"Intent Detection (CGL): {intent_weights_cgl}")
    
    # Test specific experiment
    print("\n4. Testing Specific Experiment:")
    print("-" * 30)
    
    test_experiment = "pku_vit_saliency"
    exp_info = get_experiment_info(test_experiment, "pku")
    
    print(f"Experiment: {test_experiment}")
    print(f"Checkpoint: {exp_info['checkpoint']}")
    print(f"Encoder: {exp_info['encoder']}")
    print(f"Spatial Guidance: {exp_info['spatial_guidance']}")
    print(f"Text Control: {exp_info['text_control']}")
    
    # Recommendations
    print("\n5. Recommendations:")
    print("-" * 30)
    
    if pku_available == 0 and cgl_available == 0:
        print("❌ No experiments are available. Please:")
        print("  1. Train some models first")
        print("  2. Check checkpoint directory paths in demo_config.yaml")
        print("  3. Ensure experiments follow the naming convention")
    elif pku_available + cgl_available < 24:
        print("⚠️  Some experiments are missing. Consider:")
        print("  1. Training missing experiments")
        print("  2. Checking experiment naming in config")
        print("  3. Verifying checkpoint directory structure")
    else:
        print("✅ All experiments are available!")
    
    if not saliency_weights:
        print("❌ Saliency model weights not found")
    if not intent_weights_pku:
        print("❌ PKU intent model weights not found")
    if not intent_weights_cgl:
        print("❌ CGL intent model weights not found")

if __name__ == "__main__":
    main()
