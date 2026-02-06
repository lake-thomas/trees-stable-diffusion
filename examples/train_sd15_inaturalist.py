#!/usr/bin/env python3
"""
Example: Train SD1.5 on iNaturalist data
"""

from trees_sd import train_model

if __name__ == "__main__":
    # Train SD1.5 with default settings
    trainer = train_model(
        data_dir="./data/inaturalist",
        dataset_type="inaturalist",
        model_version="sd1.5",
        output_dir="./output/sd15_inaturalist",
        lora_rank=4,
        lora_alpha=32,
        learning_rate=1e-4,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        max_train_steps=1000,
        save_steps=500,
        mixed_precision="fp16",
        seed=42,
    )
    
    print("Training completed!")
