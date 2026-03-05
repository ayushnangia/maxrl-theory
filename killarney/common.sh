#!/bin/bash
# common.sh — Shared environment setup for all Killarney jobs
# Source this file at the top of every Slurm script:
#   source /home/anangia/maxrl-theory/killarney/common.sh

set -eo pipefail

# ============ Modules ============
module load python/3.10.13 cuda/12.6 arrow/17.0.0 opencv/4.11.0

# ============ Virtual Environment ============
source ~/maxrl_env/bin/activate

# ============ HuggingFace Token ============
export HF_TOKEN=$(grep HF_TOKEN ~/.env 2>/dev/null | cut -d= -f2)

# ============ Working Directory ============
cd /home/anangia/maxrl-theory

# ============ Paths ============
export SCRATCH=/scratch/anangia
export MODEL_DIR=${SCRATCH}/models
export DATA_DIR=${SCRATCH}/data
export CKPT_DIR=${SCRATCH}/checkpoints
export LOG_DIR=${SCRATCH}/logs

# ============ NCCL / Communication ============
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IGNORE_CPU_AFFINITY=1

# ============ vLLM ============
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ============ Ray ============
export RAY_TMPDIR=/tmp/ray

# ============ WandB ============
export WANDB_MODE=offline

# ============ 7 Estimators ============
ESTIMATORS=(maxrl vr_cond grpo maclaurin maclaurin_baseline cross_fitted_maclaurin rloo)

echo "=== Killarney common.sh loaded ==="
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-N/A}"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
