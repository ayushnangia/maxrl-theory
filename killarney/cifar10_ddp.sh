#!/bin/bash
#SBATCH --job-name=cifar10-ddp
#SBATCH --account=aip-rgrosse
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/anangia/logs/cifar10_ddp_%j.out
#SBATCH --error=/scratch/anangia/logs/cifar10_ddp_%j.err

# Phase 4b: CIFAR-10 multi-GPU DDP experiments

source "$(dirname "$0")/common.sh"

echo "=== CIFAR-10 RL Experiments (4-GPU DDP) ==="
srun torchrun --nproc_per_node=4 \
    verl/cifar10_experiments/pytorch_cifar10_rl_experiments_multi_gpu.py \
    --epochs 200 \
    --wandb

echo "=== CIFAR-10 DDP experiments complete ==="
