"""Configuration loader and schema management."""

import os
import yaml
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Dictionary with parsed configuration parameters.
        
    Raises:
        FileNotFoundError: If configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}
