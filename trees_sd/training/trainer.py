"""
Unified trainer for Stable Diffusion with LoRA fine-tuning
Supports both SD1.5 and SD3.5
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import torch
from torch.utils.data import DataLoader
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
import yaml


class LoRATrainer:
    """Trainer for Stable Diffusion models with LoRA"""
    
    def __init__(
        self,
        model_version: Literal["sd1.5", "sd3.5"] = "sd1.5",
        pretrained_model_name_or_path: Optional[str] = None,
        output_dir: str = "./output",
        lora_rank: int = 4,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_train_steps: int = 1000,
        save_steps: int = 500,
        mixed_precision: str = "fp16",
        seed: int = 42,
        enable_xformers_memory_efficient_attention: bool = False,
        dataloader_num_workers: int = 2,
    ):
        """
        Initialize the LoRA trainer
        
        Args:
            model_version: SD version to use ("sd1.5" or "sd3.5")
            pretrained_model_name_or_path: Path or HuggingFace model ID
            output_dir: Directory to save outputs
            lora_rank: Rank of LoRA adaptation
            lora_alpha: Alpha parameter for LoRA
            lora_dropout: Dropout for LoRA layers
            learning_rate: Learning rate for training
            train_batch_size: Batch size for training
            gradient_accumulation_steps: Number of gradient accumulation steps
            max_train_steps: Maximum training steps
            save_steps: Save checkpoint every N steps
            mixed_precision: Mixed precision training ("no", "fp16", "bf16")
            seed: Random seed
            enable_xformers_memory_efficient_attention: Use xformers
            dataloader_num_workers: Number of workers for data loading
        """
        self.model_version = model_version
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataloader_num_workers = dataloader_num_workers
        
        # Set default model path if not provided
        if pretrained_model_name_or_path is None:
            if model_version == "sd1.5":
                pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
            elif model_version == "sd3.5":
                pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-large"
            else:
                raise ValueError(f"Unknown model version: {model_version}")
        
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        
        # Training hyperparameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_train_steps = max_train_steps
        self.save_steps = save_steps
        self.mixed_precision = mixed_precision
        self.seed = seed
        self.enable_xformers = enable_xformers_memory_efficient_attention
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        
        # Set seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            
    def load_models(self):
        """Load Stable Diffusion models"""
        print(f"Loading {self.model_version} model from {self.pretrained_model_name_or_path}...")
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer" if "/" in self.pretrained_model_name_or_path else None,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder" if "/" in self.pretrained_model_name_or_path else None,
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae" if "/" in self.pretrained_model_name_or_path else None,
        )
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="unet" if "/" in self.pretrained_model_name_or_path else None,
        )
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler" if "/" in self.pretrained_model_name_or_path else None,
        )
        
        # Freeze weights
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Apply LoRA to UNet
        self._apply_lora()
        
        # Enable xformers if requested
        if self.enable_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"Could not enable xformers: {e}")
                
    def _apply_lora(self):
        """Apply LoRA to UNet"""
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=self.lora_dropout,
        )
        
        # Apply LoRA
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
    def prepare_dataset(self, dataset, collate_fn=None):
        """Prepare dataset for training"""
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.dataloader_num_workers,
        )
        
    def train(self):
        """Main training loop"""
        # Prepare for training
        self.unet, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unet, 
            torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate),
            self.train_dataloader
        )
        
        # Move models to device
        self.vae.to(self.accelerator.device)
        self.text_encoder.to(self.accelerator.device)
        
        # Training loop
        global_step = 0
        progress_bar = tqdm(
            range(self.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        
        self.unet.train()
        
        while global_step < self.max_train_steps:
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.unet):
                    # Get latents
                    latents = self.vae.encode(
                        batch["pixel_values"].to(self.accelerator.device)
                    ).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    
                    # Sample noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    # Sample timesteps
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()
                    
                    # Add noise to latents
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )
                    
                    # Get text embeddings
                    encoder_hidden_states = self.text_encoder(
                        batch["input_ids"].to(self.accelerator.device)
                    )[0]
                    
                    # Predict noise
                    model_pred = self.unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample
                    
                    # Calculate loss (epsilon/noise prediction for both SD1.5 and SD3.5)
                    # Note: Both SD1.5 and SD3.5 use epsilon prediction by default
                    loss = torch.nn.functional.mse_loss(
                        model_pred.float(), noise.float(), reduction="mean"
                    )
                    
                    # Backprop
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                if self.accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)
                    
                    logs = {"loss": loss.detach().item()}
                    progress_bar.set_postfix(**logs)
                    
                    # Save checkpoint
                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(global_step)
                        
                if global_step >= self.max_train_steps:
                    break
                    
        # Save final checkpoint
        self.save_checkpoint(global_step)
        
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        save_path = self.output_dir / f"checkpoint-{step}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.accelerator.unwrap_model(self.unet).save_pretrained(save_path)
        
        print(f"Saved checkpoint to {save_path}")


def train_model(
    data_dir: str,
    dataset_type: str = "inaturalist",
    model_version: str = "sd1.5",
    output_dir: str = "./output",
    config_file: Optional[str] = None,
    **kwargs,
):
    """
    High-level function to train a Stable Diffusion model with LoRA
    
    Args:
        data_dir: Directory containing training data
        dataset_type: Type of dataset ("inaturalist" or "autoarborist")
        model_version: SD version ("sd1.5" or "sd3.5")
        output_dir: Directory to save outputs
        config_file: Optional YAML config file
        **kwargs: Additional training parameters
    """
    # Load config from file if provided
    config = {}
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    
    # Merge with kwargs
    config.update(kwargs)
    
    # Initialize trainer
    trainer = LoRATrainer(
        model_version=model_version,
        output_dir=output_dir,
        **config
    )
    
    # Load models
    trainer.load_models()
    
    # Create dataset
    from trees_sd.datasets import create_dataset
    dataset = create_dataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        tokenizer=trainer.tokenizer,
    )
    
    # Custom collate function
    def collate_fn(examples):
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        pixel_values = [transform(ex["image"]) for ex in examples]
        input_ids = [ex["input_ids"] for ex in examples]
        
        pixel_values = torch.stack(pixel_values)
        input_ids = torch.stack(input_ids)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }
    
    # Prepare dataset
    trainer.prepare_dataset(dataset, collate_fn=collate_fn)
    
    # Train
    trainer.train()
    
    return trainer
