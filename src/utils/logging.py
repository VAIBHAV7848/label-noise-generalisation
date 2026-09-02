"""Structured JSON and console logging utilities."""

import os
import json
import logging
from typing import Any, Dict


def setup_logger(name: str = "label_noise", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a structured logger with console and optional file handlers.
    
    Args:
        name: Name of the logger.
        log_file: Optional path to output log file.
        level: Logging level (e.g. logging.INFO).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to a formatted JSON file.
    
    Args:
        data: Dictionary data.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
