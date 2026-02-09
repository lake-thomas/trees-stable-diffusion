"""Unified image generation across dataset domains and SD versions."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import torch

from trees_sd.generation.prompts import (
    get_generation_prompts,
    get_negative_prompt,
    normalize_dataset_type,
)


class ImageGenerator:
    """Generator that supports SD1.5 and SD3.5 LoRA adapters."""

    def __init__(
        self,
        model_version: Literal["sd1.5", "sd3.5", "sdxl-refiner"] = "sd1.5",
        base_model_id: Optional[str] = None,
        dataset_type: str = "autoarborist",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.model_version = model_version
        self.dataset_type = normalize_dataset_type(dataset_type)
        self.base_model_id = base_model_id or self._default_model_for_version(model_version)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    @staticmethod
    def _default_model_for_version(model_version: str) -> str:
        if model_version == "sd1.5":
            return "runwayml/stable-diffusion-v1-5"
        if model_version == "sd3.5":
            return "stabilityai/stable-diffusion-3.5-large"
        if model_version == "sdxl-refiner":
            return "stabilityai/stable-diffusion-xl-refiner-1.0"
        raise ValueError(f"Unknown model version: {model_version}")

    def _load_pipeline(self, lora_path: Path):
        if self.model_version == "sd1.5":
            from diffusers import StableDiffusionPipeline
            from peft import PeftModel

            pipe = StableDiffusionPipeline.from_pretrained(self.base_model_id, torch_dtype=self.dtype)
            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
            return pipe.to(self.device)

        if self.model_version == "sdxl-refiner":
            from diffusers import StableDiffusionXLImg2ImgPipeline
            from peft import PeftModel

            # SDXL refiner is an img2img pipeline for refining images
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.base_model_id, 
                torch_dtype=self.dtype
            )
            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
            return pipe.to(self.device)

        from diffusers import StableDiffusion3Pipeline
        from peft import PeftModel

        pipe = StableDiffusion3Pipeline.from_pretrained(self.base_model_id, torch_dtype=self.dtype)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(lora_path))
        return pipe.to(self.device)

    def generate_for_genus(
        self,
        genus: str,
        lora_path: str,
        output_dir: str,
        num_images: int = 12,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 28,
        resolution: int = 512,
        prompt_templates: Optional[List[str]] = None,
        refiner_strength: float = 0.3,
    ) -> bool:
        """Generate images for a genus from a LoRA checkpoint."""
        lora_dir = Path(lora_path)
        if not lora_dir.exists():
            print(f"⚠️  LoRA weights not found: {lora_dir}")
            return False

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        prompts = prompt_templates or get_generation_prompts(self.dataset_type, genus)
        negative_prompt = get_negative_prompt(self.dataset_type)
        pipe = self._load_pipeline(lora_dir)

        print(f"Generating {num_images} images for genus '{genus}'...")
        for idx in range(num_images):
            prompt = prompts[idx % len(prompts)]
            with torch.no_grad():
                if self.model_version == "sdxl-refiner":
                    # For SDXL refiner, we need to generate a base image first or use img2img
                    # For now, we'll use a simple noise-to-image approach
                    # In production, you'd typically use a base SDXL model first, then refine
                    from PIL import Image
                    import numpy as np
                    
                    # Create a simple base image (normally from a base SDXL model)
                    base_image = Image.fromarray(
                        np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
                    )
                    
                    image = pipe(
                        prompt,
                        image=base_image,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        strength=refiner_strength,
                    ).images[0]
                else:
                    image = pipe(
                        prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        height=resolution,
                        width=resolution,
                    ).images[0]
            save_path = out_dir / f"{genus}_{idx:03d}.png"
            image.save(save_path)

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True


def generate_images_from_config(
    config_path: str,
    num_images: int = 12,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 28,
    resolution: int = 512,
    model_version: Optional[Literal["sd1.5", "sd3.5", "sdxl-refiner"]] = None,
    dataset_type: Optional[str] = None,
    refiner_strength: float = 0.3,
) -> Dict[str, str]:
    """Generate images from a training config JSON with selected genera and output paths."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    selected_model_version = model_version or config.get("model_version", "sd1.5")
    selected_dataset_type = dataset_type or config.get("dataset_type", "autoarborist")
    output_base = Path(config["output_path"])

    generator = ImageGenerator(
        model_version=selected_model_version,
        base_model_id=config.get("model_path"),
        dataset_type=selected_dataset_type,
    )

    prompt_templates = config.get("prompts")
    if isinstance(prompt_templates, str):
        prompt_templates = [prompt_templates]

    genera: Iterable[str] = config.get("selected_genera", [])
    results: Dict[str, str] = {}

    for genus in genera:
        lora_path = output_base / f"lora-{genus}"
        output_dir = output_base / "tree_gen" / f"lora-{genus}" / "generated_images"

        success = generator.generate_for_genus(
            genus=genus,
            lora_path=str(lora_path),
            output_dir=str(output_dir),
            num_images=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            resolution=resolution,
            prompt_templates=prompt_templates,
            refiner_strength=refiner_strength,
        )
        results[genus] = "success" if success else "failed"

    return results
