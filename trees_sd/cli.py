"""
Command-line interface for Trees SD training
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Train Stable Diffusion with LoRA on tree images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset arguments
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
        help="Type of dataset to use",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_version",
        type=str,
        choices=["sd1.5", "sd3.5"],
        default="sd1.5",
        help="Stable Diffusion version to use",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Pretrained model path or HuggingFace model ID",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save outputs",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA adaptation",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Alpha parameter for LoRA",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout for LoRA layers",
    )
    
    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="fp16",
        help="Mixed precision training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers memory efficient attention",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        choices=["none", "wandb"],
        default="none",
        help="Experiment tracker backend",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases team/user entity",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="Weights & Biases API key (or set WANDB_API_KEY env var)",
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    
    args = parser.parse_args()
    
    # Convert args to dict, excluding None values
    kwargs = {k: v for k, v in vars(args).items() if v is not None and k not in ['data_dir', 'dataset_type', 'model_version', 'output_dir', 'config']}
    
    print("=" * 50)
    print("Trees SD - Stable Diffusion Fine-tuning with LoRA")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Model version: {args.model_version}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        # Import train_model only when actually running training
        from trees_sd.training import train_model
        
        # Train model
        trainer = train_model(
            data_dir=args.data_dir,
            dataset_type=args.dataset_type,
            model_version=args.model_version,
            output_dir=args.output_dir,
            config_file=args.config,
            **kwargs
        )
        
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print(f"Model saved to: {args.output_dir}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
