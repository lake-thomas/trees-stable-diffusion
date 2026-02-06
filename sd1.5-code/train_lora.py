#!/usr/bin/env python3
"""
Train LoRA models for multiple genera using HuggingFace's training script.
This script orchestrates training across multiple genera.
"""

import os
import json
import argparse
import subprocess
import torch
from peft import PeftModel

from precompute_latents import precompute_latents
from prompt_profiles import (
    get_generation_prompts,
    get_negative_prompt,
    normalize_dataset_type,
)


def train_genus(
    genus,
    train_data_dir,
    output_dir,
    model_id,
    resolution=512,
    train_batch_size=1,
    num_epochs=10,
    learning_rate=1e-04,
    mixed_precision="fp16",
    gradient_accumulation_steps=4,
    checkpointing_steps=500,
    validation_epochs=1,
    use_wandb=False,
    latents_dir=None,
    dataset_type="autoarborist"
):
    """
    Train LoRA for a single genus.

    Args:
        genus: Genus name
        train_data_dir: Directory containing training images and metadata.jsonl
        output_dir: Output directory for checkpoints
        model_id: Model ID or path
        resolution: Image resolution
        train_batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        mixed_precision: Mixed precision mode (fp16, bf16, or no)
        gradient_accumulation_steps: Gradient accumulation steps
        checkpointing_steps: Save checkpoint every N steps
        validation_epochs: Run validation every N epochs
        use_wandb: Whether to use wandb logging
        dataset_type: Dataset prompt profile (autoarborist or inaturalist)
    """
    print(f"\n{'='*60}")
    print(f"TRAINING LORA FOR: {genus}")
    print(f"{'='*60}\n")

    if not os.path.exists(train_data_dir):
        print(f"{genus}: training folder not found at {train_data_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    
    # Prompt inspired by: https://www.inaturalist.org/guide_taxa/355708
    # validation_prompt = f"A photo of a tree, genus {genus}, closeup of the leaves, and the whole plant if possible, fruit or flowers if there are any, for trees, close-up of bark is also helpful."

    dataset_type = normalize_dataset_type(dataset_type)

    # Build command - run training script directly with python
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_text_to_image_lora.py") # Ensure this script is sd1.5 version

    # run command with python
    cmd = [
        sys.executable,
        train_script,
        f"--pretrained_model_name_or_path={model_id}",
        f"--train_data_dir={train_data_dir}",
        "--dataloader_num_workers=0",
        f"--resolution={resolution}",
        "--center_crop",
        "--random_flip",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--num_train_epochs={num_epochs}",
        f"--learning_rate={learning_rate}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--seed=1337",
        f"--output_dir={output_dir}",
        f"--checkpointing_steps={checkpointing_steps}",
        "--checkpoints_total_limit=2",
        f"--mixed_precision={mixed_precision}",
        f"--dataset_type={dataset_type}",
    ]

    if use_wandb:
        cmd.append("--report_to=wandb")
    else:
        cmd.append("--report_to=none")

    # Add latents directory if provided (FAST path - skips VAE encoding)
    if latents_dir and os.path.exists(os.path.join(latents_dir, "latents.pt")):
        cmd.append(f"--latents_dir={latents_dir}")
        print(f"Using pre-computed latents from {latents_dir}")

    # Add memory optimization flags
    # cmd.append("--vae_cpu_offload")  # Disabled - H100 has plenty of memory (79GB)
    cmd.append("--enable_xformers")  # Enable xformers for faster training

    print("Launching training with command:")
    print(" ".join(cmd), "\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\nTraining completed for {genus}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed for {genus}: {e}")
        return False


def generate_images_for_genus(
    genus,
    lora_path,
    base_model_id,
    output_dir,
    dataset_type="autoarborist",
    num_images=12
):
    """Generate sample images after training"""

    print(f"\nGenerating sample images for {genus}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # generate stable diffusion 1.5 pipeline
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16
        )

        # Apply LoRA
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        pipe = pipe.to(device)

        dataset_type = normalize_dataset_type(dataset_type)
        prompts = get_generation_prompts(dataset_type, genus)
        negative_prompt = get_negative_prompt(dataset_type)

        for idx in range(num_images):
            prompt = prompts[idx % len(prompts)]
            with torch.no_grad():
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=7.0,
                    num_inference_steps=28,
                    height=512,
                    width=512
                ).images[0]

            save_path = os.path.join(output_dir, f"generated_tree{idx}.png")
            image.save(save_path)

        print(f"Generated {num_images} images for {genus}")
        del pipe
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"Image generation failed for {genus}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train LoRA models for multiple genera")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config JSON file")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip image generation after training")
    parser.add_argument("--num_images", type=int, default=12,
                       help="Number of images to generate per genus")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extract config
    train_base = config["train_data_dir"]
    output_base = config["output_path"]
    genera = config["selected_genera"]
    model_id = config.get("model_path", "runwayml/stable-diffusion-v1-5")
    dataset_type = normalize_dataset_type(config.get("dataset_type", "autoarborist"))

    # Training hyperparameters
    resolution = config.get("resolution", 512)
    train_batch_size = config.get("train_batch_size", 1)
    num_epochs = config.get("epochs", 10)
    learning_rate = config.get("learning_rate", 1e-4)
    mixed_precision = config.get("mixed_precision", "fp16")
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
    checkpointing_steps = config.get("checkpointing_steps", 500)
    validation_epochs = config.get("validation_epochs", 1)
    use_wandb = config.get("use_wandb", False)

    # Latent caching options
    do_precompute_latents = config.get("precompute_latents", False)
    latents_cache_dir = config.get("latents_cache_dir", None)
    model_cache = os.environ.get("MODEL_CACHE", r"C:\Users\talake2\Desktop\sd1.5\model_cache")

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Genera to train: {len(genera)}")
    print(f"Resolution: {resolution}")
    print(f"Batch size: {train_batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dataset type: {dataset_type}")
    print(f"Pre-compute latents: {do_precompute_latents}")
    print(f"{'='*60}\n")

    # Pre-compute latents if enabled
    if do_precompute_latents and latents_cache_dir:
        print(f"\n{'='*60}")
        print("PRE-COMPUTING LATENTS")
        print(f"{'='*60}\n")

        for genus in genera:
            genus_data_dir = os.path.join(train_base, genus, "lora")
            genus_latents_dir = os.path.join(latents_cache_dir, genus)

            # Skip if already computed
            if os.path.exists(os.path.join(genus_latents_dir, "latents.pt")):
                print(f"Latents already exist for {genus}, skipping...")
                continue

            if not os.path.exists(genus_data_dir):
                print(f"Data directory not found for {genus}, skipping...")
                continue

            print(f"\nPre-computing latents for {genus}...")
            precompute_latents(
                data_dir=genus_data_dir,
                output_dir=genus_latents_dir,
                model_cache=model_cache,
                resolution=resolution,
                batch_size=8,  # Can use larger batch for encoding
                mixed_precision=mixed_precision,
                center_crop=True,
            )

        print(f"\n{'='*60}")
        print("LATENT PRE-COMPUTATION COMPLETE")
        print(f"{'='*60}\n")

    # Train each genus
    results = {}
    for genus in genera:
        train_data_dir = os.path.join(train_base, genus)
        output_dir = os.path.join(output_base, f"lora-{genus}")

        # Check for pre-computed latents
        genus_latents_dir = None
        if do_precompute_latents and latents_cache_dir:
            potential_latents_dir = os.path.join(latents_cache_dir, genus)
            if os.path.exists(os.path.join(potential_latents_dir, "latents.pt")):
                genus_latents_dir = potential_latents_dir

        # Check if LoRA already exists — skip training and go straight to generation
        lora_path = output_dir
        success = False
        if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
            # Final trained model exists
            print(f"{genus}: Trained LoRA already exists at {output_dir}, skipping training...")
            success = True
        elif os.path.exists(output_dir):
            # Check for latest checkpoint from an interrupted run
            checkpoints = sorted(
                [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1])
            )
            if checkpoints:
                lora_path = os.path.join(output_dir, checkpoints[-1])
                print(f"{genus}: Found checkpoint {checkpoints[-1]}, skipping to generation...")
                success = True

        if not success:
            success = train_genus(
                genus=genus,
                train_data_dir=train_data_dir,
                output_dir=output_dir,
                model_id=model_id,
                resolution=resolution,
                train_batch_size=train_batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=gradient_accumulation_steps,
                checkpointing_steps=checkpointing_steps,
                validation_epochs=validation_epochs,
                use_wandb=use_wandb,
                latents_dir=genus_latents_dir,
                dataset_type=dataset_type
            )

        results[genus] = "success" if success else "failed"

        # Generate images after successful training (or from existing checkpoint)
        if success and not args.skip_generation:
            gen_success = generate_images_for_genus(
                genus=genus,
                lora_path=lora_path,
                base_model_id=model_id,
                output_dir=output_dir,
                dataset_type=dataset_type,
                num_images=args.num_images
            )
            results[genus] = "success + images" if gen_success else "success (no images)"

    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for genus, status in results.items():
        icon = "✓" if "success" in status else "✗"
        print(f"{icon} {genus}: {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
