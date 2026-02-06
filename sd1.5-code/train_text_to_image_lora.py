#!/usr/bin/env python3
"""
SD1.5 LoRA training script for text-to-image models.
Based on HuggingFace diffusers UNet LoRA training.
"""

import argparse
import logging
import math
import os
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from PIL import Image

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model

# Optional wandb (disabled by default)
try:
    import wandb
except ImportError:
    wandb = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train SD1.5 LoRA")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)

    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=2)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )

    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--vae_cpu_offload", action="store_true")

    parser.add_argument(
        "--latents_dir",
        type=str,
        default=None,
        help="Directory containing pre-computed latents.pt",
    )

    parser.add_argument("--report_to", type=str, default="none")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Image transforms (only used if NOT using precomputed latents)
# -----------------------------------------------------------------------------
def transform_images(examples, resolution=512, center_crop=False, random_flip=False):
    transform_list = []

    if center_crop:
        transform_list.append(transforms.Resize(resolution))
        transform_list.append(transforms.CenterCrop(resolution))
    else:
        transform_list.append(transforms.Resize((resolution, resolution)))

    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_fn = transforms.Compose(transform_list)

    pixel_values = []
    for img in examples["image"]:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        pixel_values.append(transform_fn(img).numpy())

    examples["pixel_values"] = pixel_values
    return examples

# -----------------------------------------------------------------------------
# Dataset for precomputed latents
# -----------------------------------------------------------------------------
class LatentDataset(TorchDataset):
    def __init__(self, latents_path):
        logger.info(f"Loading latents from {latents_path}")
        data = torch.load(latents_path, map_location="cpu", weights_only=True)
        self.latents = data["latents"]
        self.scaling_factor = data.get("scaling_factor", 0.18215)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {"latents": self.latents[idx]}

# -----------------------------------------------------------------------------
# LR scheduler
# -----------------------------------------------------------------------------
def cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Mixed precision dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # -------------------------------------------------------------------------
    # Load tokenizer & text encoder (SD1.5)
    # -------------------------------------------------------------------------
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    # -------------------------------------------------------------------------
    # Load VAE
    # -------------------------------------------------------------------------
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )
    vae.requires_grad_(False)

    # -------------------------------------------------------------------------
    # Load UNet
    # -------------------------------------------------------------------------
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
    )

    # -------------------------------------------------------------------------
    # Add LoRA to UNet
    # -------------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    # -------------------------------------------------------------------------
    # Scheduler
    # -------------------------------------------------------------------------
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    use_latents = args.latents_dir is not None

    if use_latents:
        latents_path = os.path.join(args.latents_dir, "latents.pt")
        dataset = LatentDataset(latents_path)

        def collate_fn(examples):
            return {
                "latents": torch.stack([ex["latents"] for ex in examples])
            }

    else:
        from datasets import load_dataset

        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            split="train",
        )

        dataset = dataset.map(
            lambda ex: transform_images(
                ex,
                args.resolution,
                args.center_crop,
                args.random_flip,
            ),
            batched=True,
            batch_size=100,
        )
        dataset = dataset.remove_columns(["image"])

        def collate_fn(examples):
            pixel_values = torch.stack([
                torch.from_numpy(ex["pixel_values"]) for ex in examples
            ])
            return {"pixel_values": pixel_values}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # Optimizer & LR scheduler
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
    )

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = steps_per_epoch * args.num_train_epochs

    if args.lr_scheduler == "constant":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        lr_scheduler = cosine_with_warmup(
            optimizer,
            args.lr_warmup_steps,
            max_train_steps,
        )

    # -------------------------------------------------------------------------
    # Prompt embedding (single fixed prompt per genus)
    # -------------------------------------------------------------------------
    genus = os.path.basename(args.train_data_dir.rstrip("/"))
    prompt = f"a street-level Google Street View photograph of a tree, genus {genus}"

    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(tokens)[0]

    # Free text encoder
    text_encoder.to("cpu")
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # Move models to GPU
    if not use_latents:
        if args.vae_cpu_offload:
            vae.to("cpu")
        else:
            vae.to(device)
    else:
        del vae
        gc.collect()

    unet.to(device)

    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"xFormers not available: {e}")

    unet.enable_gradient_checkpointing()

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(dataset)}")
    logger.info(f"Num epochs = {args.num_train_epochs}")
    logger.info(f"Batch size = {args.train_batch_size}")
    logger.info(f"Total steps = {max_train_steps}")

    global_step = 0
    optimizer.zero_grad()
    progress_bar = tqdm(range(max_train_steps))

    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(dataloader):

            if use_latents:
                latents = batch["latents"].to(device, dtype=weight_dtype)
            else:
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states.repeat(bsz, 1, 1),
            ).sample

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)

                if global_step % args.checkpointing_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unet.save_pretrained(ckpt_dir)
                    logger.info(f"Saved checkpoint to {ckpt_dir}")

    progress_bar.close()

    # Save final LoRA
    unet.save_pretrained(args.output_dir)
    logger.info(f"Training complete. LoRA saved to {args.output_dir}")

if __name__ == "__main__":
    main()
