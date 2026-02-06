#!/usr/bin/env python3
"""
Example: Compare SD1.5 and SD3.5 on the same dataset
"""

from trees_sd import train_model
import os

def train_comparison(data_dir, dataset_type):
    """Train both SD1.5 and SD3.5 for comparison"""
    
    print("=" * 60)
    print("Training SD1.5...")
    print("=" * 60)
    
    # Train SD1.5
    trainer_15 = train_model(
        data_dir=data_dir,
        dataset_type=dataset_type,
        model_version="sd1.5",
        output_dir=f"./output/comparison/sd15_{dataset_type}",
        lora_rank=4,
        lora_alpha=32,
        learning_rate=1e-4,
        max_train_steps=1000,
        save_steps=500,
        mixed_precision="fp16",
    )
    
    print("\n" + "=" * 60)
    print("Training SD3.5...")
    print("=" * 60)
    
    # Train SD3.5
    trainer_35 = train_model(
        data_dir=data_dir,
        dataset_type=dataset_type,
        model_version="sd3.5",
        output_dir=f"./output/comparison/sd35_{dataset_type}",
        lora_rank=8,
        lora_alpha=64,
        learning_rate=5e-5,
        max_train_steps=1000,
        save_steps=500,
        mixed_precision="bf16",
    )
    
    print("\n" + "=" * 60)
    print("Comparison training completed!")
    print(f"SD1.5 output: ./output/comparison/sd15_{dataset_type}")
    print(f"SD3.5 output: ./output/comparison/sd35_{dataset_type}")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python compare_models.py <data_dir> <dataset_type>")
        print("Example: python compare_models.py ./data/trees inaturalist")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    dataset_type = sys.argv[2]
    
    if dataset_type not in ["inaturalist", "autoarborist"]:
        print(f"Error: dataset_type must be 'inaturalist' or 'autoarborist'")
        sys.exit(1)
    
    train_comparison(data_dir, dataset_type)
