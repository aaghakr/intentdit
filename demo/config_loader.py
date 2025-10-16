"""
Configuration loader for demo app
Handles checkpoint discovery and fallback logic
"""

import os
import yaml
import glob
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoConfigLoader:
    """Loads and manages demo configuration with checkpoint discovery"""
    
    def __init__(self, config_path: str = "demo_config.yaml"):
        """Initialize the config loader"""
        self.config_path = config_path
        self.config = self._load_config()
        self.base_checkpoint_dir = self.config.get("base_checkpoint_dir", "")
        self.base_model_weights_dir = self.config.get("base_model_weights_dir", "")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file {self.config_path} not found")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            return {}
    
    def get_model_weights(self, model_type: str, dataset: str = None) -> Optional[str]:
        """Get model weights path for saliency or intent detection"""
        model_weights = self.config.get("model_weights", {})
        
        if model_type == "saliency":
            # Default to ISNet for saliency detection
            return model_weights.get("saliency", {}).get("isnet")
        elif model_type == "intent":
            if dataset:
                return model_weights.get("intent", {}).get(dataset.lower())
            else:
                # Return PKU as default
                return model_weights.get("intent", {}).get("pku")
        
        return None
    
    def find_latest_checkpoint(self, experiment_name: str, dataset: str) -> Optional[str]:
        """Find the latest checkpoint for an experiment"""
        if not self.base_checkpoint_dir:
            return None
            
        # Construct experiment directory path
        exp_dir = os.path.join(self.base_checkpoint_dir, dataset, experiment_name)
        
        if not os.path.exists(exp_dir):
            logger.warning(f"Experiment directory not found: {exp_dir}")
            return None
        
        # Look for checkpoint files with pattern: Epoch*_cgbdm_weights.pth
        checkpoint_pattern = os.path.join(exp_dir, "**", "Epoch*_cgbdm_weights.pth")
        checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
        
        if not checkpoint_files:
            logger.warning(f"No checkpoint files found in {exp_dir}")
            return None
        
        # Sort by modification time and return the latest
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    def get_experiment_config(self, experiment_name: str, dataset: str) -> Dict:
        """Get configuration for a specific experiment"""
        # Get experiment config from YAML
        if dataset.lower() == "pku":
            exp_config = self.config.get("pku_experiments", {}).get(experiment_name, {})
        elif dataset.lower() == "cgl":
            exp_config = self.config.get("cgl_experiments", {}).get(experiment_name, {})
        else:
            logger.error(f"Unknown dataset: {dataset}")
            return {}
        
        # If no specific checkpoint is defined, find the latest one
        if not exp_config.get("checkpoint"):
            latest_checkpoint = self.find_latest_checkpoint(experiment_name, dataset)
            if latest_checkpoint:
                exp_config["checkpoint"] = latest_checkpoint
            else:
                logger.warning(f"No checkpoint found for {experiment_name}")
                exp_config["checkpoint"] = None
        
        return exp_config
    
    def get_experiment_checkpoint(self, experiment_name: str, dataset: str) -> Optional[str]:
        """Get checkpoint path for an experiment"""
        exp_config = self.get_experiment_config(experiment_name, dataset)
        return exp_config.get("checkpoint")
    
    def get_experiment_info(self, experiment_name: str, dataset: str) -> Dict:
        """Get complete experiment information"""
        exp_config = self.get_experiment_config(experiment_name, dataset)
        
        return {
            "experiment_name": experiment_name,
            "dataset": dataset,
            "checkpoint": exp_config.get("checkpoint"),
            "encoder": exp_config.get("encoder", "vit"),
            "spatial_guidance": exp_config.get("spatial_guidance", "saliency"),
            "text_control": exp_config.get("text_control", False),
            "model_weights": {
                "saliency": self.get_model_weights("saliency"),
                "intent": self.get_model_weights("intent", dataset)
            }
        }
    
    def list_available_experiments(self, dataset: str = None) -> Dict[str, Dict]:
        """List all available experiments with their status"""
        available_experiments = {}
        
        datasets = [dataset] if dataset else ["pku", "cgl"]
        
        for ds in datasets:
            if ds.lower() == "pku":
                experiments = self.config.get("pku_experiments", {})
            elif ds.lower() == "cgl":
                experiments = self.config.get("cgl_experiments", {})
            else:
                continue
            
            for exp_name, exp_config in experiments.items():
                checkpoint_path = self.get_experiment_checkpoint(exp_name, ds)
                available_experiments[exp_name] = {
                    "dataset": ds,
                    "checkpoint_available": checkpoint_path is not None,
                    "checkpoint_path": checkpoint_path,
                    "encoder": exp_config.get("encoder"),
                    "spatial_guidance": exp_config.get("spatial_guidance"),
                    "text_control": exp_config.get("text_control")
                }
        
        return available_experiments
    
    def validate_config(self) -> Tuple[bool, list]:
        """Validate configuration and return status"""
        errors = []
        
        # Check if base directories exist
        if not os.path.exists(self.base_checkpoint_dir):
            errors.append(f"Base checkpoint directory not found: {self.base_checkpoint_dir}")
        
        if not os.path.exists(self.base_model_weights_dir):
            errors.append(f"Base model weights directory not found: {self.base_model_weights_dir}")
        
        # Check model weights
        model_weights = self.config.get("model_weights", {})
        for model_type, weights in model_weights.items():
            for weight_name, weight_path in weights.items():
                if weight_path and not os.path.exists(weight_path):
                    errors.append(f"Model weight not found: {weight_path}")
        
        return len(errors) == 0, errors
    
    def get_defaults(self) -> Dict:
        """Get default configuration values"""
        return self.config.get("defaults", {})
    
    def get_fallback_config(self) -> Dict:
        """Get fallback configuration"""
        return self.config.get("fallback", {})

# Global config loader instance
config_loader = DemoConfigLoader()

def get_experiment_checkpoint(experiment_name: str, dataset: str) -> Optional[str]:
    """Convenience function to get experiment checkpoint"""
    return config_loader.get_experiment_checkpoint(experiment_name, dataset)

def get_experiment_info(experiment_name: str, dataset: str) -> Dict:
    """Convenience function to get experiment info"""
    return config_loader.get_experiment_info(experiment_name, dataset)

def get_model_weights(model_type: str, dataset: str = None) -> Optional[str]:
    """Convenience function to get model weights"""
    return config_loader.get_model_weights(model_type, dataset)

def list_available_experiments(dataset: str = None) -> Dict[str, Dict]:
    """Convenience function to list available experiments"""
    return config_loader.list_available_experiments(dataset)

def validate_config() -> Tuple[bool, list]:
    """Convenience function to validate configuration"""
    return config_loader.validate_config()

if __name__ == "__main__":
    # Test the configuration loader
    print("Testing Demo Configuration Loader")
    print("=" * 50)
    
    # Validate configuration
    is_valid, errors = validate_config()
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
    
    print("\nAvailable Experiments:")
    print("-" * 30)
    experiments = list_available_experiments()
    for exp_name, exp_info in experiments.items():
        status = "✅" if exp_info["checkpoint_available"] else "❌"
        print(f"{status} {exp_name}: {exp_info['dataset']} - {exp_info['encoder']} - {exp_info['spatial_guidance']}")
    
    print("\nModel Weights:")
    print("-" * 30)
    saliency_weights = get_model_weights("saliency")
    intent_weights_pku = get_model_weights("intent", "pku")
    intent_weights_cgl = get_model_weights("intent", "cgl")
    
    print(f"Saliency: {saliency_weights}")
    print(f"Intent (PKU): {intent_weights_pku}")
    print(f"Intent (CGL): {intent_weights_cgl}")
