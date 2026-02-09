"""
Example script for training SDXL Refiner on tree images

This script demonstrates how to use the SDXL Refiner model for fine-tuning
with LoRA on tree images from iNaturalist or Autoarborist datasets.
"""

import argparse
from pathlib import Path
from trees_sd.training import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train SDXL Refiner with LoRA on tree images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training data",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["inaturalist", "autoarborist"],
        default="inaturalist",
        help="Type of dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/sdxl_refiner",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--refiner_strength",
        type=float,
        default=0.3,
        help="Refiner strength (0.0-1.0)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Training SDXL Refiner on Tree Images")
    print("=" * 70)
    print(f"Dataset: {args.dataset_type}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max steps: {args.max_train_steps}")
    print(f"Refiner strength: {args.refiner_strength}")
    print("=" * 70)
    
    # Train the model
    trainer = train_model(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        model_version="sdxl-refiner",
        output_dir=args.output_dir,
        lora_rank=8,
        lora_alpha=64,
        learning_rate=5e-5,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        max_train_steps=args.max_train_steps,
        save_steps=250,
        mixed_precision="fp16",
        refiner_strength=args.refiner_strength,
        use_refiner=True,
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 70)
    print("\nTo generate images with the trained model:")
    print(f"trees-sd-generate \\")
    print(f"  --config {args.output_dir}/train_config.json \\")
    print(f"  --model_version sdxl-refiner \\")
    print(f"  --dataset_type {args.dataset_type} \\")
    print(f"  --num_images 12")


if __name__ == "__main__":
    main()
