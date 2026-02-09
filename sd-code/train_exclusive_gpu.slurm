#!/bin/bash
#SBATCH --job-name=sd3_lora_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclusive

# Request exclusive access to prevent GPU sharing

# Activate conda environment
source ~/.bashrc
conda activate pytorch

# Create logs directory
mkdir -p logs

# Set genus and paths
GENUS="fraxinus"
TRAIN_DATA_DIR="/path/to/your/data/${GENUS}"
OUTPUT_DIR="/path/to/output/${GENUS}_lora"

echo "Starting training for genus: ${GENUS}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Run training
python train_text_to_image_lora.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-3.5-large" \
  --train_data_dir "${TRAIN_DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --lr_scheduler "constant" \
  --checkpointing_steps 500 \
  --mixed_precision "fp16" \
  --vae_cpu_offload \
  --enable_xformers

echo "Training complete!"
