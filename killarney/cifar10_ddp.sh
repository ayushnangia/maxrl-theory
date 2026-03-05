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

source /home/anangia/maxrl-theory/killarney/common.sh

CIFAR_DATA=/scratch/anangia/data/cifar10
CIFAR_CKPT=/scratch/anangia/checkpoints/cifar10_ddp
mkdir -p ${CIFAR_DATA} ${CIFAR_CKPT}

echo "=== CIFAR-10 RL Experiments (4-GPU DDP) ==="
torchrun --nproc_per_node=4 \
    verl/cifar10_experiments/pytorch_cifar10_rl_experiments_multi_gpu.py \
    --data-dir ${CIFAR_DATA} \
    --checkpoint-dir ${CIFAR_CKPT} \
    --epochs 200 \
    --no-amp \
    --wandb

echo "=== CIFAR-10 DDP experiments complete ==="
