# LoRA Training with Precomputed Latents

A modular LoRA fine-tuning pipeline for Stable Diffusion with precomputed latent caching for efficient training.

## Core Scripts

- **train_lora.py** - Main orchestrator for training multiple LoRA models
- **train_text_to_image_lora.py** - Core training script with latent caching
- **precompute_latents.py** - Precompute and cache image latents for faster training
- **generate_images.py** - Generate images using trained LoRA models
- **download_models.py** - Download Stable Diffusion models for offline use

## Quick Start

### 1. Precompute Latents
```bash
python precompute_latents.py \
    --model_path /path/to/model \
    --train_data_dir /path/to/dataset \
    --output_dir /path/to/latents \
    --resolution 512
```

### 2. Configure Training
Edit `modular_config.json` with your settings:
```json
{
    "model_path": "/path/to/model",
    "train_data_dir": "/path/to/dataset",
    "latent_cache_dir": "/path/to/latents",
    "output_dir": "/path/to/outputs",
    "epochs": 10,
    "learning_rate": 1e-4
}
```

### 3. Train
```bash
python train_lora.py --config modular_config.json
```

### 4. Generate
```bash
python generate_images.py \
    --model_path /path/to/model \
    --lora_path /path/to/trained/lora \
    --prompt "your prompt" \
    --num_images 4
```

## HPC Deployment

Use the provided SLURM template: `train_exclusive_gpu.slurm`

Set environment variables for offline model loading:
```bash
export HF_HOME=/path/to/model_cache
export TRANSFORMERS_CACHE=/path/to/model_cache
```

## Requirements

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install peft
pip install datasets pillow tqdm
```
