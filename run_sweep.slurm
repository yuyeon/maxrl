#!/bin/bash

#SBATCH --job-name=smollm-rg
#SBATCH --account=bhqc-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=04:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

#SBATCH --output=slurm-smollm-rg-%j.out
#SBATCH --error=slurm-smollm-rg-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p rg_sweep_results/smollm2-360m-instruct

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

export MAMBA_ROOT_PREFIX=/u/ali35/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate /u/ali35/envs/rg-sweep

echo "Environment: $CONDA_PREFIX"
echo "Python: $(which python3)"

python3 -c '
import sys
import torch
import vllm

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("vLLM:", vllm.__version__)
'

python3 -u smollm/rg_sweep.py \
  --backend vllm \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --out rg_sweep_results/smollm2-360m-instruct \
  --all \
  --size 50 \
  --samples 32 \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-new-tokens 2048 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --examples-per-task 5

echo "Finished: $(date)"