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
    AutoencoderKL, # For both SD1.5 and SD3.5
    DDPMScheduler, # For SD1.5
    UNet2DConditionModel, # For SD1.5
    FlowMatchEulerDiscreteScheduler, # For SD3.5
    SD3Transformer2DModel, # For SD3.5
)

from transformers import (
    CLIPTextModel, 
    CLIPTokenizer, # CLIP L
    CLIPTextModelWithProjection, # CLIP G
    T5Tokenizer, 
    T5EncoderModel,
)

from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
import yaml

try:
    import wandb
except Exception:
    wandb = None


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
        dataloader_num_workers: int = 0,
        report_to: str = "none",
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
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
            report_to: Experiment tracker backend ("none" or "wandb")
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases user/team entity
            wandb_run_name: Optional run name for Weights & Biases
            wandb_api_key: Optional Weights & Biases API key
        """
        self.model_version = model_version
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataloader_num_workers = dataloader_num_workers # Set to 0 for ForkingPickler compatibility
        
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
        self.report_to = report_to
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_api_key = wandb_api_key

        self._configure_tracking_env()

        # Initialize accelerator
        if self.report_to == "wandb":
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=mixed_precision,
                log_with="wandb",
                project_dir=str(self.output_dir),
            )
        else:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=mixed_precision,
            )
        
        # Set seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            

    def _configure_tracking_env(self):
        """Configure experiment tracking backends."""
        if self.report_to not in {"none", "wandb"}:
            raise ValueError(f"Unsupported report_to value: {self.report_to}")

        if self.report_to != "wandb":
            return

        if wandb is None:
            raise ImportError("wandb is not installed. Please install wandb to enable tracking.")

        if self.wandb_api_key:
            os.environ["WANDB_API_KEY"] = self.wandb_api_key

        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project

        if self.wandb_entity:
            os.environ["WANDB_ENTITY"] = self.wandb_entity

        if self.wandb_run_name:
            os.environ["WANDB_NAME"] = self.wandb_run_name


    def load_models(self):
        """Load Stable Diffusion model versions 1.5 or 3.5 based on configuration"""
        print(f"Loading {self.model_version} model from {self.pretrained_model_name_or_path}...")

        # Load tokenizer and text encoders
        # SD 1.5 uses CLIP L, while SD 3.5 uses three encoders:CLIP L, CLIP G and T5

        # Text encoders and tokenizers
        if self.model_version == "sd3.5":
            self.tokenizer_l = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer")
            self.text_encoder_l = CLIPTextModelWithProjection.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder")

            self.tokenizer_g = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer_2")
            self.text_encoder_g = CLIPTextModelWithProjection.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder_2")

            self.tokenizer_t5 = T5Tokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer_3")
            self.text_encoder_t5 = T5EncoderModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder_3")

            self.tokenizer = self.tokenizer_l  # Use tokenizer_l as the main tokenizer
        
        else: # SD 1.5
            self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder")

        # Load VAE (for both SD1.5 and SD3.5)
        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae")

        # Load UNet or SD3 transformer based on model version
        if self.model_version == "sd3.5":
            self.unet = SD3Transformer2DModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="transformer")
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")

        else: # SD 1.5
            self.unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="unet")
            self.noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")
        
        # Freeze base model weights
        self.vae.requires_grad_(False)
        if self.model_version == "sd3.5":
            self.text_encoder_l.requires_grad_(False)
            self.text_encoder_g.requires_grad_(False)
            self.text_encoder_t5.requires_grad_(False)
        else:
            self.text_encoder.requires_grad_(False)
            
        # Freeze base model parameters too
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
        """Apply LoRA to UNet or Transformer based on model version"""
        if self.model_version == "sd3.5":
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        else: # SD 1.5
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
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

    def save_checkpoint(self, global_step: int):
        """Save LoRA checkpoint"""

        if not self.accelerator.is_main_process:
            return

        save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)

        # Unwrap model from accelerator
        unwrapped_unet = self.accelerator.unwrap_model(self.unet)

        # Save only LoRA adapters (PEFT)
        unwrapped_unet.save_pretrained(save_path)

        self.accelerator.print(f"Saved LoRA checkpoint to {save_path}")
        
    def train(self):
        """Main training loop for SD1.5 and SD3.5 with LoRA"""
        # Prepare models and optimizer
        self.unet, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unet,
            torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate),
            self.train_dataloader
        )

        # Move models to device
        self.vae.to(self.accelerator.device)
        if self.model_version == "sd3.5":
            self.text_encoder_l.to(self.accelerator.device)
            self.text_encoder_g.to(self.accelerator.device)
            self.text_encoder_t5.to(self.accelerator.device)
        else:
            self.text_encoder.to(self.accelerator.device)

        # Set training mode
        self.unet.train()

        # Setup progress bar
        global_step = 0
        progress_bar = tqdm(
            range(self.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )

        # Initialize W&B tracking if requested
        if self.report_to == "wandb":
            tracker_kwargs = {"wandb": {}}
            if self.wandb_entity:
                tracker_kwargs["wandb"]["entity"] = self.wandb_entity
            if self.wandb_run_name:
                tracker_kwargs["wandb"]["name"] = self.wandb_run_name

            run_name = self.wandb_run_name or f"trees-sd-{self.model_version}"
            self.accelerator.init_trackers(
                project_name=self.wandb_project or "trees-stable-diffusion",
                config={
                    "model_version": self.model_version,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "learning_rate": self.learning_rate,
                    "train_batch_size": self.train_batch_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "max_train_steps": self.max_train_steps,
                },
                init_kwargs=tracker_kwargs if tracker_kwargs["wandb"] else None,
            )
            self.accelerator.print(f"Initialized W&B tracking with run name: {run_name}")

        # Main training loop
        while global_step < self.max_train_steps:
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.unet):
                    # Encode images to latents
                    latents = self.vae.encode(batch["pixel_values"].to(self.accelerator.device)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    # Sample noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # --- BRANCH: NOISE SCHEDULING ---
                    if self.model_version == "sd3.5":
                        timesteps = torch.rand((bsz,), device=latents.device, dtype=torch.float32)

                        noisy_latents = ((1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise)

                        target = noise - latents
                        
                    else:
                        # Standard Diffusion Logic (SD1.5)
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=latents.device
                        ).long()

                        # Add noise
                        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                        
                        # Target is Noise
                        target = noise

                    # --- BRANCH: MODEL FORWARD PASS ---
                    if self.model_version == "sd3.5":

                        device = self.accelerator.device
                        weight_dtype = latents.dtype # torch.float32

                        # -----------------------------
                        # CLIP-L
                        # -----------------------------
                        out_l = self.text_encoder_l(
                            batch["input_ids_l"].to(device),
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        hidden_states_l = out_l.hidden_states[-2]
                        pooled_l = out_l.text_embeds

                        # -----------------------------
                        # CLIP-G
                        # -----------------------------
                        out_g = self.text_encoder_g(
                            batch["input_ids_g"].to(device),
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        hidden_states_g = out_g.hidden_states[-2]
                        pooled_g = out_g.text_embeds

                        # -----------------------------
                        # Combine CLIP token embeddings
                        # -----------------------------
                        clip_embeds = torch.cat([hidden_states_l, hidden_states_g], dim=-1)  # [B,77,2048]

                        # Pad to 4096
                        clip_embeds = torch.nn.functional.pad(
                            clip_embeds,
                            (0, 4096 - clip_embeds.shape[-1])
                        )  # [B,77,4096]

                        # -----------------------------
                        # T5 full sequence
                        # -----------------------------
                        t5_out = self.text_encoder_t5(
                            batch["t5_input_ids"].to(device),
                            return_dict=True,
                        )
                        hidden_states_t5 = t5_out.last_hidden_state  # [B,256,4096]

                        # -----------------------------
                        # Final encoder_hidden_states
                        # -----------------------------
                        encoder_hidden_states = torch.cat(
                            [clip_embeds, hidden_states_t5],
                            dim=1
                        )  # [B,333,4096]

                        # -----------------------------
                        # Pooled projections (CLIP ONLY)
                        # -----------------------------
                        pooled_projections = torch.cat(
                            [pooled_l, pooled_g],
                            dim=-1
                        )

                        # -----------------------------
                        # Timesteps (Flow matching)
                        # -----------------------------
                        timesteps_scaled = timesteps * 1000.0

                        model_pred = self.unet(
                            hidden_states=noisy_latents,
                            timestep=timesteps_scaled,
                            encoder_hidden_states=encoder_hidden_states,
                            pooled_projections=pooled_projections,
                            return_dict=False,
                        )[0]

                    else:
                        # SD1.5 UNet Forward Pass
                        encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.accelerator.device))[0]
                        model_pred = self.unet(
                            sample=noisy_latents,             # SD1.5 uses 'sample'
                            timestep=timesteps,
                            encoder_hidden_states=encoder_hidden_states
                        ).sample

                    # Compute loss
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Backprop
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Step logging and checkpointing
                if self.accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)
                    logs = {"loss": loss.detach().item(), "step": global_step}
                    progress_bar.set_postfix(loss=logs["loss"])
                    if self.report_to == "wandb":
                        self.accelerator.log(logs, step=global_step)

                    # Save checkpoint
                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(global_step)

                if global_step >= self.max_train_steps:
                    break

        # Save final checkpoint
        self.save_checkpoint(global_step)

        if self.report_to == "wandb":
            self.accelerator.end_training()

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
    def collate_fn(examples, trainer: LoRATrainer):
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        pixel_values = torch.stack([transform(ex["image"]) for ex in examples])
        captions = [ex["caption"] for ex in examples]  # or however text is stored

        batch = {
            "pixel_values": pixel_values,
        }

        if trainer.model_version == "sd1.5":
            tokens = trainer.tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=trainer.tokenizer.model_max_length,
                return_tensors="pt",
            )
            batch["input_ids"] = tokens.input_ids

        else:  # SD3.5
            tokens_l = trainer.tokenizer_l(
                captions,
                padding="max_length",
                truncation=True,
                max_length=trainer.tokenizer_l.model_max_length,
                return_tensors="pt",
            )
            tokens_g = trainer.tokenizer_g(
                captions,
                padding="max_length",
                truncation=True,
                max_length=trainer.tokenizer_g.model_max_length,
                return_tensors="pt",
            )
            tokens_t5 = trainer.tokenizer_t5(
                captions,
                padding="max_length",
                truncation=True,
                max_length=trainer.tokenizer_t5.model_max_length,
                return_tensors="pt",
            )

            batch.update({
                "input_ids_l": tokens_l.input_ids,
                "input_ids_g": tokens_g.input_ids,
                "t5_input_ids": tokens_t5.input_ids,
            })

        return batch

    
    # Prepare dataset
    trainer.prepare_dataset(
    dataset,
    collate_fn=lambda examples: collate_fn(
        examples,
        trainer=trainer,
    ))
    
    # Train
    trainer.train()
    
    return trainer
