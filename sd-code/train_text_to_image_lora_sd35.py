#!/usr/bin/env python3
"""
SD3 LoRA training script for text-to-image models.
Based on HuggingFace diffusers training approach, adapted for Stable Diffusion 3.
"""

import argparse
import logging
import math
import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, T5Tokenizer, T5EncoderModel
import numpy as np


# Disable wandb to avoid network issues on HPC
# os.environ["WANDB_MODE"] = "disabled"
try:
    import wandb
except ImportError:
    wandb = None

# set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple LoRA training script")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
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
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers", action="store_true", help="Enable xformers memory efficient attention")
    parser.add_argument("--vae_cpu_offload", action="store_true", help="Keep VAE on CPU to save GPU memory")
    parser.add_argument("--latents_dir", type=str, default=None,
                        help="Directory containing pre-computed latents (skips VAE encoding)")

    args = parser.parse_args()
    return args


def transform_images(examples, resolution=512, center_crop=False, random_flip=False):
    """Transform images to tensors normalized to [-1, 1]"""
    # Build transform list
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
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transform_fn = transforms.Compose(transform_list)

    pixel_values = []
    for img in examples["image"]:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        # Convert tensor to numpy to avoid PyArrow offset overflow
        pixel_values.append(transform_fn(img).numpy())
    examples["pixel_values"] = pixel_values
    return examples


class LatentDataset(TorchDataset):
    """Dataset that loads pre-computed latents directly."""

    def __init__(self, latents_path):
        logger.info(f"Loading pre-computed latents from {latents_path}")
        data = torch.load(latents_path, map_location="cpu", weights_only=True)
        self.latents = data["latents"]
        self.filenames = data.get("filenames", [])
        self.scaling_factor = data.get("scaling_factor", 0.18215)
        logger.info(f"Loaded {len(self.latents)} pre-computed latents")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {"latents": self.latents[idx]}


def learning_rate_generator(optimizer, num_warmup_steps, num_training_steps):
    """Simple cosine schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()

    # Must be set before any CUDA call — ignored if runtime is already initialized
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine data type for mixed precision
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # load models for SD3
    logger.info("Loading SD3 models from local cache")

    # Use local cache directory - models are stored in flat directory structure
    model_cache_base = os.environ.get("MODEL_CACHE", r"C:\Users\talake2\Desktop\sd3.5\model_cache")

    # SD3.5 base path - contains all tokenizers and models
    sd3_path = os.path.join(model_cache_base, "stabilityai_stable-diffusion-3.5-large")

    # Separate model paths for text encoders
    clip_l_path = os.path.join(model_cache_base, "openai_clip-vit-large-patch14")
    clip_g_path = os.path.join(model_cache_base, "laion_CLIP-ViT-bigG-14-laion2B-39B-b160k")
    t5_path = os.path.join(model_cache_base, "google_t5-v1_1-xxl")

    # SD3 Text Encoders (3 total) - load tokenizers from SD3.5 directory, models from separate dirs
    # CLIP-L: OpenAI CLIP ViT-Large
    tokenizer_l = CLIPTokenizer.from_pretrained(
        sd3_path,
        subfolder="tokenizer",
        local_files_only=True
    )
    text_encoder_l = CLIPTextModelWithProjection.from_pretrained(
        clip_l_path,
        torch_dtype=weight_dtype,
        local_files_only=True
    )

    # CLIP-G: OpenCLIP ViT-bigG
    tokenizer_g = CLIPTokenizer.from_pretrained(
        sd3_path,
        subfolder="tokenizer_2",
        local_files_only=True
    )
    text_encoder_g = CLIPTextModelWithProjection.from_pretrained(
        clip_g_path,
        torch_dtype=weight_dtype,
        local_files_only=True
    )

    # T5-XXL: Google T5 v1.1 XXL
    tokenizer_t5 = T5Tokenizer.from_pretrained(
        sd3_path,
        subfolder="tokenizer_3",
        local_files_only=True
    )
    text_encoder_t5 = T5EncoderModel.from_pretrained(
        t5_path,
        torch_dtype=weight_dtype,
        local_files_only=True
    )

    # VAE: SD3 uses 16-channel VAE
    vae = AutoencoderKL.from_pretrained(
        sd3_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        local_files_only=True
    )

    # Transformer: MMDiT (Multimodal Diffusion Transformer) - replaces UNet
    # Load in weight_dtype (bf16) to fit in 40 GB GPU
    transformer = SD3Transformer2DModel.from_pretrained(
        sd3_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True
    )

    # Noise Scheduler: continous flow matching
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        sd3_path,
        subfolder="scheduler",
        local_files_only=True
    )

    # Freeze vae and text encoders
    vae.requires_grad_(False)
    text_encoder_l.requires_grad_(False)
    text_encoder_g.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)

    # Add LoRA to Transformer
    logger.info("Adding LoRA layers...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none"
    )
    transformer = get_peft_model(transformer, lora_config)

    # Check GPU memory right before first allocation
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        free_memory = total_memory - allocated_memory
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"Total GPU memory: {total_memory:.2f} GB")
        logger.info(f"Allocated: {allocated_memory:.2f} GB")
        logger.info(f"Free: {free_memory:.2f} GB")

        if free_memory < 20:
            logger.warning(f"WARNING: Only {free_memory:.2f} GB free on GPU. SD3.5 needs ~20+ GB.")
            logger.warning("Consider killing other GPU processes or using a less busy GPU.")

    # Move only text encoders to device temporarily (for embedding computation)
    # Keep VAE and transformer off GPU for now to save memory
    text_encoder_l.to(device)
    text_encoder_g.to(device)
    text_encoder_t5.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate
    )

    # Dataset loading - use pre-computed latents if available
    use_precomputed_latents = args.latents_dir is not None

    if use_precomputed_latents:
        # Load pre-computed latents (FAST path)
        logger.info("Loading pre-computed latents (VAE encoding will be skipped)...")
        latents_path = os.path.join(args.latents_dir, "latents.pt")
        dataset = LatentDataset(latents_path)

        def collate_fn(examples):
            latents = torch.stack([ex["latents"] for ex in examples])
            return {"latents": latents}

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        logger.info(f"Dataset loaded: {len(dataset)} pre-computed latents")

    else:
        # Original path - load images and encode on the fly (SLOW)
        logger.info("Loading dataset (images will be VAE-encoded each batch - consider using --latents_dir)...")
        from datasets import load_dataset

        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, split="train")
        dataset = dataset.map(
            lambda examples: transform_images(examples, args.resolution, args.center_crop, args.random_flip),
            batched=True,
            batch_size=100,
            writer_batch_size=100
        )
        dataset = dataset.remove_columns(["image"])

        def collate_fn(examples):
            pixel_values = []
            for example in examples:
                pv = example["pixel_values"]
                if isinstance(pv, np.ndarray):
                    pixel_values.append(torch.from_numpy(pv))
                elif isinstance(pv, list):
                    pixel_values.append(torch.tensor(pv))
                else:
                    pixel_values.append(pv)
            pixel_values = torch.stack(pixel_values)
            return {"pixel_values": pixel_values}

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        logger.info(f"Dataset loaded: {len(dataset)} images")

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if args.lr_scheduler == "constant":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        lr_scheduler = learning_rate_generator(
            optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=max_train_steps
        )

    # Compute text embeddings 
    logger.info("Computing text embeddings...")
    
    # Extract genus name from train_data_dir
    genus = os.path.basename(args.train_data_dir.rstrip('/'))
    prompt = f"a street-level Google Street View photograph of a tree, genus {genus}, urban environment"

    # Tokenize with all three encoders
    tokens_l = tokenizer_l(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    tokens_g = tokenizer_g(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    tokens_t5 = tokenizer_t5(prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        # CLIP-L encoding
        outputs_l = text_encoder_l(tokens_l, output_hidden_states=True)
        hidden_states_l = outputs_l.hidden_states[-2]
        pooled_embeds_l = outputs_l.text_embeds

        # CLIP-G encoding
        outputs_g = text_encoder_g(tokens_g, output_hidden_states=True)
        hidden_states_g = outputs_g.hidden_states[-2]
        pooled_embeds = outputs_g.text_embeds

        # Concatenate pooled embeddings from CLIP-L and CLIP-G
        pooled_embeds = torch.cat([pooled_embeds_l, pooled_embeds], dim=-1)

        # T5 encoding
        hidden_states_t5 = text_encoder_t5(tokens_t5).last_hidden_state

        # Concatenate CLIP embeddings along feature dimension
        clip_embeds = torch.cat([hidden_states_l, hidden_states_g], dim=-1)
        # clip_embeds shape: [1, 77, 2048] (768 + 1280)

        # Pad CLIP embeddings to match T5's 4096 dimensions
        clip_embeds_padded = F.pad(clip_embeds, (0, 4096 - clip_embeds.shape[-1]))
        # clip_embeds_padded shape: [1, 77, 4096]

        # Concatenate CLIP and T5 along sequence dimension
        encoder_hidden_states = torch.cat([clip_embeds_padded, hidden_states_t5], dim=1)
        # encoder_hidden_states shape: [1, 333, 4096] (77 + 256)

    # Match transformer dtype
    encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
    pooled_embeds = pooled_embeds.to(weight_dtype)

    logger.info(f"Text embeddings computed for genus: {genus}")

    # Free text encoders from GPU (no longer needed)
    logger.info("Offloading text encoders from GPU to free memory...")
    text_encoder_l.to("cpu")
    text_encoder_g.to("cpu")
    text_encoder_t5.to("cpu")

    # Delete references to free memory
    del text_encoder_l, text_encoder_g, text_encoder_t5
    del outputs_l, outputs_g, hidden_states_l, hidden_states_g, hidden_states_t5
    del tokens_l, tokens_g, tokens_t5, clip_embeds, clip_embeds_padded

    # Aggressive garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if torch.cuda.is_available():
        logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved")

    # Now move VAE and transformer to GPU
    logger.info("Loading VAE and transformer to GPU...")
    if use_precomputed_latents:
        logger.info("VAE not needed (using pre-computed latents) - skipping GPU load")
        del vae  # Free memory since we don't need it
        import gc
        gc.collect()
    elif not args.vae_cpu_offload:
        vae.to(device)
    else:
        logger.info("VAE will remain on CPU (--vae_cpu_offload enabled)")
        vae.to("cpu")

    # Move transformer to GPU with error handling
    try:
        logger.info("Moving transformer to GPU...")
        transformer.to(device)
        logger.info(f"Transformer loaded successfully. GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"Failed to load transformer to GPU: {e}")
        logger.error("GPU is full. Try:")
        logger.error("  1. Kill other processes on the GPU")
        logger.error("  2. Use a GPU with more memory")
        logger.error("  3. Reduce batch size or resolution")
        logger.error("  4. Use gradient checkpointing (already enabled)")
        raise

    # Enable gradient checkpointing to reduce memory usage
    if hasattr(transformer, 'enable_gradient_checkpointing'):
        logger.info("Enabling gradient checkpointing...")
        transformer.enable_gradient_checkpointing()

    # Enable xformers if requested
    if args.enable_xformers:
        try:
            transformer.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")

    # Initialize wandb if requested - needs wifi
    use_wandb = args.report_to == "wandb" and wandb is not None
    # if use_wandb:
    #     wandb.init(
    #         project="Tree-Gen",
    #         entity="talake2-ncsu",
    #         name=f"lora-{genus}",
    #         config={
    #             "genus": genus,
    #             "resolution": args.resolution,
    #             "batch_size": args.train_batch_size,
    #             "gradient_accumulation_steps": args.gradient_accumulation_steps,
    #             "num_epochs": args.num_train_epochs,
    #             "learning_rate": args.learning_rate,
    #             "lora_rank": args.lora_rank,
    #             "lora_alpha": args.lora_alpha,
    #             "num_examples": len(dataset),
    #         }
    #     )

    # Training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Print GPU memory usage
    if torch.cuda.is_available():
        logger.info(f"  GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        logger.info(f"  GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

    global_step = 0
    progress_bar = tqdm(range(max_train_steps))

    # Initialize gradients
    optimizer.zero_grad()

    for epoch in range(args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            # Get latents - either pre-computed or encode on the fly
            if use_precomputed_latents:
                # FAST path: latents already computed
                latents = batch["latents"].to(device, dtype=weight_dtype)
            else:
                # SLOW path: encode images through VAE
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)

                with torch.no_grad():
                    if args.vae_cpu_offload:
                        latents = vae.encode(pixel_values.to("cpu")).latent_dist.sample()
                        latents = (latents * vae.config.scaling_factor).to(device)
                    else:
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                    latents = latents.to(weight_dtype)

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample timesteps (continuous time for flow matching)
            timesteps = torch.rand((bsz,), device=device, dtype=torch.float32)

            # Add noise using flow matching: x_t = (1-t)*x_0 + t*noise
            # Cast back to weight_dtype — fp32 timesteps promote the result to fp32
            noisy_latents = ((1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise).to(weight_dtype)

            # Expand encoder hidden states to match batch size
            encoder_batch = encoder_hidden_states.repeat(bsz, 1, 1)
            pooled_batch = pooled_embeds.repeat(bsz, 1)

            # Predict velocity (v = noise - x_0)
            # SD3 expects timesteps scaled to [0, 1000] range
            timesteps_scaled = timesteps * 1000.0

            # Call transformer - try different API versions
            try:
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps_scaled,
                    encoder_hidden_states=encoder_batch,
                    pooled_projections=pooled_batch,
                    return_dict=False
                )[0]
            except TypeError as e:
                # Fallback: try without hidden_states parameter name (positional)
                logger.warning(f"Transformer call failed, trying alternative API: {e}")
                model_pred = transformer(
                    noisy_latents,
                    timestep=timesteps_scaled,
                    encoder_hidden_states=encoder_batch,
                    pooled_projections=pooled_batch,
                    return_dict=False
                )[0]

            # Flow matching loss: target is velocity; cast to fp32 for stable loss computation
            target = noise - latents
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Backward with gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    transformer.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

                logs = {"loss": loss.detach().item() * args.gradient_accumulation_steps,
                       "lr": lr_scheduler.get_last_lr()[0],
                       "epoch": epoch,
                       "step": global_step}
                progress_bar.set_postfix(**logs)

                # Log to wandb
                if use_wandb:
                    wandb.log(logs)

    # Close progress bar
    progress_bar.close()

    # Save final model
    transformer.save_pretrained(args.output_dir)
    logger.info(f"Training complete! Model saved to {args.output_dir}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

