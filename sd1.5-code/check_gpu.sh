#!/bin/bash
# Check GPU memory usage on HPC

echo "=== GPU Status ==="
nvidia-smi

echo ""
echo "=== Processes using GPU ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "=== Your processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | grep $(whoami) || echo "No processes found for $(whoami)"

echo ""
echo "=== Recommended Actions ==="
echo "1. Request exclusive GPU: Use SLURM with --gres=gpu:1 --exclusive"
echo "2. Or request a full GPU node in your batch script"
echo "3. Or manually select a less busy GPU with CUDA_VISIBLE_DEVICES=X"
