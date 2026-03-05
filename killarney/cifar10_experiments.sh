#!/bin/bash
#SBATCH --job-name=cifar10-single
#SBATCH --account=aip-rgrosse
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/anangia/logs/cifar10_%j.out
#SBATCH --error=/scratch/anangia/logs/cifar10_%j.err

# Phase 4a: CIFAR-10 single-GPU experiments

source /home/anangia/maxrl-theory/killarney/common.sh

echo "=== CIFAR-10 RL Experiments (single GPU) ==="
python verl/cifar10_experiments/pytorch_cifar10_rl_experiments.py \
    --epochs 200 \
    --wandb

echo "=== CIFAR-10 Gradient Similarity Analysis ==="
python verl/cifar10_experiments/gradient_similarity_analysis.py

echo "=== CIFAR-10 experiments complete ==="
