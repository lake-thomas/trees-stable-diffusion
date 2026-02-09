#!/bin/bash
#BSUB -n 4
#BSUB -W 480
#BSUB -J generate_modular
#BSUB -o stdout_generate.%J
#BSUB -e stderr_generate.%J
#BSUB -q gpu
#BSUB -R span[hosts=1]
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R "select[ h100 || a100 || l40 || l40s ]"
#BSUB -R rusage[mem=64]

module load conda
source activate /usr/local/usrapps/rkmeente/btfarre2/conda_envs/pytorch

export HF_DATASETS_CACHE=/share/rkmeente/btfarre2/model/model_cache/datasets
export TRANSFORMERS_CACHE=/share/rkmeente/btfarre2/model/model_cache
export HF_HOME=/share/rkmeente/btfarre2/model/model_cache
export TMPDIR=/share/rkmeente/btfarre2/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PATH=/usr/local/usrapps/rkmeente/btfarre2/conda_envs/pytorch/bin:$PATH
export PYTHONPATH=$PYTHONPATH:/home/btfarre2/gsv_host_detector/tree_classification

cd /home/btfarre2/gsv_host_detector/tree_classification

python modular/generate_images.py --config "modular/modular_config.json" --num_images 12 --guidance_scale 7.5 --num_inference_steps 30 --resolution 512

conda deactivate
