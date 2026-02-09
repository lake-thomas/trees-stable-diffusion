#!/usr/bin/env python3
"""
Generate images using trained LoRA models.
"""

import os
import json
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel


def generate_for_genus(
    genus,
    lora_path,
    base_model_id,
    output_dir,
    num_images=12,
    guidance_scale=7.5,
    num_inference_steps=30,
    resolution=512,
    device="cuda"
):
    """
    Generate images for a single genus using its LoRA weights.

    Args:
        genus: Genus name
        lora_path: Path to LoRA weights
        base_model_id: Base model ID or path
        output_dir: Output directory for generated images
        num_images: Number of images to generate
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
        resolution: Image resolution
        device: Device to use
    """
    print(f"\n{'='*60}")
    print(f"Generating images for: {genus}")
    print(f"{'='*60}")

    if not os.path.exists(lora_path):
        print(f"⚠️  LoRA weights not found: {lora_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline
    print("Loading base model...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16
    )

    # Load LoRA weights
    print("Loading LoRA weights...")
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)

    pipe = pipe.to(device)

    # Generate prompts - Mix of detailed descriptive prompts, realistic contexts, and creative scenarios
    # prompts = [f"A real-world iNaturalist photograph of a tree, genus {genus}, taken outdoors in natural lighting. The image shows diagnostic features, such as leaves, branching structure, bark texture, and if present, flowers or fruit. Unstaged, field photograph, realistic perspective, ecological context."]
    prompts = [f"A real-world iNaturalist photograph of a tree, genus {genus}, taken outdoors in natural lighting. The image shows disease symptoms on leaves, like wilt, dieback, or leaf scorch. Spotted lanternfly insect has infested the tree."]
    # Generate images
    print(f"Generating {num_images} images...")
    prompt = prompts[0]

    for idx in range(num_images):
        print(f"  [{idx+1}/{num_images}] {prompt}")

        with torch.no_grad():
            image = pipe(
                prompt,
                negative_prompt="Illustration, drawing, painting, sketch, cartoon, anime, 3d render, cgi, artificial lighting, studio lighting, product photo, stock photo, extreme close-up, bokeh, fantasy, unreal, surreal, abstract, text, watermark, logo, frame",
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=resolution,
                width=resolution,
            ).images[0]

        save_path = os.path.join(output_dir, f"{genus}_{idx:03d}.png")
        image.save(save_path)
        print(f"    ✓ Saved: {save_path}")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    print(f"✓ Generation complete for {genus}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate images using trained LoRA models")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config JSON file")
    parser.add_argument("--num_images", type=int, default=12,
                       help="Number of images to generate per genus")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                       help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                       help="Number of inference steps")
    parser.add_argument("--resolution", type=int, default=1024,
                       help="Image resolution")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    base_model_id = config.get("model_path", "stabilityai/stable-diffusion-3-medium-diffusers")
    output_base = config["output_path"]
    genera = config["selected_genera"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"IMAGE GENERATION")
    print(f"{'='*60}")
    print(f"Base model: {base_model_id}")
    print(f"Device: {device}")
    print(f"Genera to process: {len(genera)}")
    print(f"{'='*60}\n")

    results = {}

    for genus in genera:
        # Load LoRA weights from where training saved them
        lora_path = os.path.join(output_base, f"lora-{genus}")
        # Save generated images to a separate tree_gen directory
        output_dir = os.path.join(output_base, "tree_gen", f"lora-{genus}", "generated_images")

        success = generate_for_genus(
            genus=genus,
            lora_path=lora_path,
            base_model_id=base_model_id,
            output_dir=output_dir,
            num_images=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            resolution=args.resolution,
            device=device
        )

        results[genus] = "success" if success else "failed"

    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    for genus, status in results.items():
        icon = "✓" if status == "success" else "✗"
        print(f"{icon} {genus}: {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
