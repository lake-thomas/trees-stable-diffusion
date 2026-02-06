"""
Utility functions for Trees SD
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_dataset_format(data_dir: str, dataset_type: str) -> bool:
    """
    Check if dataset directory has the expected format
    
    Args:
        data_dir: Directory containing the dataset
        dataset_type: Either 'inaturalist' or 'autoarborist'
        
    Returns:
        True if format is valid, False otherwise
    """
    from pathlib import Path
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return False
    
    # Check for metadata files
    if dataset_type == "inaturalist":
        metadata_file = data_path / "metadata.json"
        if not metadata_file.exists():
            print(f"Warning: metadata.json not found in {data_dir}")
            print("Will scan for images directly")
    elif dataset_type == "autoarborist":
        annotations_file = data_path / "annotations.json"
        if not annotations_file.exists():
            print(f"Warning: annotations.json not found in {data_dir}")
            print("Will scan for images directly")
    
    # Check for image files
    image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    if len(image_files) == 0:
        print(f"Error: No image files found in {data_dir}")
        return False
    
    print(f"Found {len(image_files)} image files")
    return True


def print_model_info(model_version: str):
    """Print information about the model version"""
    info = {
        "sd1.5": {
            "name": "Stable Diffusion 1.5",
            "default_model": "runwayml/stable-diffusion-v1-5",
            "description": "Original Stable Diffusion model, widely used and well-tested"
        },
        "sd3.5": {
            "name": "Stable Diffusion 3.5",
            "default_model": "stabilityai/stable-diffusion-3.5-large",
            "description": "Latest version with improved quality and composition"
        }
    }
    
    if model_version in info:
        model_info = info[model_version]
        print(f"\nModel: {model_info['name']}")
        print(f"Default: {model_info['default_model']}")
        print(f"Description: {model_info['description']}\n")


def estimate_training_time(
    num_images: int,
    max_train_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> str:
    """
    Estimate training time
    
    Args:
        num_images: Number of training images
        max_train_steps: Maximum training steps
        batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation steps
        
    Returns:
        Estimated time as string
    """
    # Rough estimate: 1-2 seconds per step on GPU
    total_seconds = max_train_steps * 1.5  # Average estimate
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    if hours > 0:
        return f"~{hours}h {minutes}m"
    else:
        return f"~{minutes}m"
