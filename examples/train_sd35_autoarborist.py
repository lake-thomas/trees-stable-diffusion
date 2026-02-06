#!/usr/bin/env python3
"""
Example: Train SD3.5 on Autoarborist data
"""

from trees_sd import train_model

if __name__ == "__main__":
    # Train SD3.5 with optimized settings
    trainer = train_model(
        data_dir="./data/autoarborist",
        dataset_type="autoarborist",
        model_version="sd3.5",
        output_dir="./output/sd35_autoarborist",
        lora_rank=8,
        lora_alpha=64,
        learning_rate=5e-5,
        train_batch_size=1,
        gradient_accumulation_steps=8,
        max_train_steps=2000,
        save_steps=500,
        mixed_precision="bf16",
        seed=42,
        enable_xformers_memory_efficient_attention=True,
    )
    
    print("Training completed!")
