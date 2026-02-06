# Trees Stable Diffusion

A unified Python package for fine-tuning Stable Diffusion models with LoRA on tree images from iNaturalist and Autoarborist datasets.

## Features

- 🌲 **Unified Interface**: Single codebase supporting both SD1.5 and SD3.5
- 🎯 **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with LoRA
- 📊 **Multiple Datasets**: Built-in support for iNaturalist and Autoarborist formats
- ⚙️ **Flexible Configuration**: Command-line arguments or YAML config files
- 🚀 **Production Ready**: Accelerate integration for distributed training
- 💾 **Checkpointing**: Regular checkpoint saving during training

## Installation

```bash
# Clone the repository
git clone https://github.com/lake-thomas/trees-stable-diffusion.git
cd trees-stable-diffusion

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

Train SD1.5 on iNaturalist data:

```bash
trees-sd-train \
  --data_dir /path/to/inaturalist/data \
  --dataset_type inaturalist \
  --model_version sd1.5 \
  --output_dir ./output/sd15_inaturalist \
  --max_train_steps 1000
```

Train SD3.5 on Autoarborist data:

```bash
trees-sd-train \
  --data_dir /path/to/autoarborist/data \
  --dataset_type autoarborist \
  --model_version sd3.5 \
  --output_dir ./output/sd35_autoarborist \
  --max_train_steps 2000
```

### Using Configuration Files

Use pre-configured settings:

```bash
trees-sd-train \
  --data_dir /path/to/data \
  --dataset_type inaturalist \
  --config trees_sd/configs/sd15_inaturalist.yaml \
  --output_dir ./output
```

### Advanced Options

Full control over training hyperparameters:

### Weights & Biases (W&B) Tracking

To log training runs to your W&B account, either export your API key:

```bash
export WANDB_API_KEY=your_api_key
```

or pass it directly at runtime, then enable W&B reporting:

```bash
trees-sd-train \
  --data_dir /path/to/data \
  --dataset_type inaturalist \
  --model_version sd1.5 \
  --output_dir ./output \
  --report_to wandb \
  --wandb_project trees-sd \
  --wandb_entity your_wandb_username_or_team \
  --wandb_run_name sd15-inat-exp1
```

Optional: provide `--wandb_api_key` if you prefer not to export `WANDB_API_KEY`.

```bash
trees-sd-train \
  --data_dir /path/to/data \
  --dataset_type inaturalist \
  --model_version sd1.5 \
  --output_dir ./output \
  --lora_rank 8 \
  --lora_alpha 64 \
  --learning_rate 1e-4 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1500 \
  --save_steps 250 \
  --mixed_precision fp16 \
  --enable_xformers_memory_efficient_attention
```

### Evaluate Generated Image Quality

After generating images, compare them against the original iNaturalist/Autoarborist images using post-hoc metrics (including FID):

```bash
trees-sd-eval \
  --real_dir /path/to/real_images \
  --generated_dir /path/to/generated_images \
  --metrics fid basic \
  --device cuda \
  --output_json ./output/eval_metrics.json
```

- `fid` computes Frechet Inception Distance between real and generated sets.
- `basic` reports dataset-level sanity stats (count, average resolution, mean RGB).

## Dataset Formats

### iNaturalist Format

Expected directory structure:

```
data/
├── metadata.json          # Optional metadata file
├── image1.jpg
├── image2.jpg
└── ...
```

Example `metadata.json`:

```json
[
  {
    "image_path": "image1.jpg",
    "caption": "A photo of an oak tree",
    "species": "Quercus robur"
  },
  {
    "image_path": "image2.jpg",
    "caption": "A photo of a pine tree",
    "species": "Pinus sylvestris"
  }
]
```

### Autoarborist Format

Expected directory structure:

```
data/
├── annotations.json       # Optional annotations file
├── image1.jpg
├── image2.jpg
└── ...
```

Example `annotations.json`:

```json
[
  {
    "image_file": "image1.jpg",
    "caption": "A photo of a maple tree",
    "tree_info": {
      "species": "Acer saccharum",
      "height": "15m"
    }
  }
]
```

If metadata/annotations files are not present, the package will scan for images and generate basic captions.

## Python API

You can also use the package programmatically:

```python
from trees_sd import train_model

# Train a model
trainer = train_model(
    data_dir="/path/to/data",
    dataset_type="inaturalist",
    model_version="sd1.5",
    output_dir="./output",
    lora_rank=4,
    lora_alpha=32,
    learning_rate=1e-4,
    max_train_steps=1000,
)
```

## Model Versions

### SD1.5 (Stable Diffusion 1.5)
- **Default Model**: `runwayml/stable-diffusion-v1-5`
- **Best For**: General purpose, well-tested, broad compatibility
- **Recommended Settings**: 
  - LoRA rank: 4-8
  - Learning rate: 1e-4
  - Mixed precision: fp16

### SD3.5 (Stable Diffusion 3.5)
- **Default Model**: `stabilityai/stable-diffusion-3.5-large`
- **Best For**: Improved quality, better composition, latest features
- **Recommended Settings**:
  - LoRA rank: 8-16
  - Learning rate: 5e-5
  - Mixed precision: bf16

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_dir` | Directory containing training data | Required |
| `--dataset_type` | Dataset type (inaturalist/autoarborist) | inaturalist |
| `--model_version` | SD version (sd1.5/sd3.5) | sd1.5 |
| `--output_dir` | Output directory for checkpoints | ./output |
| `--lora_rank` | Rank of LoRA adaptation | 4 |
| `--lora_alpha` | Alpha parameter for LoRA | 32 |
| `--lora_dropout` | Dropout for LoRA layers | 0.1 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--train_batch_size` | Training batch size | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 4 |
| `--max_train_steps` | Maximum training steps | 1000 |
| `--save_steps` | Save checkpoint every N steps | 500 |
| `--mixed_precision` | Mixed precision (no/fp16/bf16) | fp16 |
| `--seed` | Random seed | 42 |
| `--report_to` | Experiment tracker backend (`none`/`wandb`) | none |
| `--wandb_project` | W&B project name | None |
| `--wandb_entity` | W&B user/team entity | None |
| `--wandb_run_name` | W&B run name | None |
| `--wandb_api_key` | W&B API key (or use `WANDB_API_KEY`) | None |

## Tips for Training

1. **Start Small**: Begin with 1000 steps to verify everything works
2. **Monitor Loss**: Loss should decrease over time; if not, adjust learning rate
3. **Use Config Files**: Easier to reproduce experiments
4. **Enable XFormers**: Use `--enable_xformers_memory_efficient_attention` for memory savings
5. **Batch Size**: Increase if you have GPU memory, or use gradient accumulation
6. **Save Often**: Regular checkpoints let you recover from interruptions

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM (with gradient checkpointing)
- **Recommended**: NVIDIA GPU with 16GB+ VRAM
- **CPU Training**: Possible but very slow, not recommended

## Troubleshooting

### Out of Memory Errors
- Reduce `train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `--enable_xformers_memory_efficient_attention`
- Use mixed precision training

### Slow Training
- Enable XFormers for memory efficient attention
- Use fp16 or bf16 mixed precision
- Increase batch size if you have GPU memory

## Examples

Example configurations are provided in `trees_sd/configs/`:
- `sd15_inaturalist.yaml` - SD1.5 with iNaturalist data
- `sd35_inaturalist.yaml` - SD3.5 with iNaturalist data
- `sd15_autoarborist.yaml` - SD1.5 with Autoarborist data

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
