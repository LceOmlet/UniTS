import yaml
import os
import json
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, config_path: str = None):
        """Initialize config manager"""
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 
            "../default_config/config_template.yaml"
        )
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load config from yaml file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model specific configuration"""
        return self.config['model_configs'].get(model_name, {})
    
    def get_optimizer_config(self, model_name: str, task: str = "pretraining") -> Dict[str, Any]:
        """Get merged optimizer configuration
        Args:
            model_name: Name of the model
            task: Task name for task-specific config
        Returns:
            Merged configuration dictionary
        """
        # 1. Start with base optimizer config
        base_config = self.config['optimizer'].copy()
        base_config['optimizer'] = base_config.pop('name', 'Adam')
        
        # 2. Update with preprocessing config
        preprocess_config = self.config.get('preprocessing', {})
        base_config.update(preprocess_config)
        
        # 3. Update with task specific config
        if task and self.config.get('task_configs', {}):
            task_config = self.config['task_configs']
            base_config.update(task_config)
            
        # 4. Update with model specific optimizer config
        if model_name in self.config['optimizer']:
            model_config = self.config['optimizer'][model_name]
            # Handle nested optimizer name if present
            if 'optimizer' in model_config:
                base_config['optimizer'] = model_config.pop('optimizer')
            base_config.update(model_config)
            
        # 5. Add training config
        training_config = self.config.get('training', {})
        base_config.update(training_config)
        
        return base_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['training']
    
    def update_config(self, updates: Dict[str, Any], model_name: str = None):
        """Update configuration with command line arguments"""
        if model_name:
            self.config['model_configs'][model_name].update(updates)
        else:
            self.config.update(updates)