#!/bin/bash
#SBATCH --job-name=maxrl-setup
#SBATCH --account=aip-rgrosse
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/anangia/logs/setup_%j.out
#SBATCH --error=/scratch/anangia/logs/setup_%j.err

# Phase 0: One-time setup — build vllm from source, download models and datasets
# IMPORTANT: This runs on the compute node (not login node!) via sbatch

set -euo pipefail

# ============ Environment ============
module load python/3.10.13 cuda/12.6 arrow/17.0.0 opencv/4.11.0
source ~/maxrl_env/bin/activate
cd /home/anangia/maxrl-theory

SCRATCH=/scratch/anangia

# ============ Create Directory Structure ============
echo "=== Creating directories ==="
mkdir -p ${SCRATCH}/{models,data,checkpoints,logs}
mkdir -p ${SCRATCH}/checkpoints/{smollm,maze,qwen3}

# ============ Build vLLM 0.8.4 from Source ============
echo "=== Building vLLM 0.8.4 from source (on compute node with CUDA 12.6) ==="
BUILD_DIR=$(mktemp -d /tmp/vllm_build.XXXXXX)
cd ${BUILD_DIR}

git clone --branch v0.8.4 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm

# Build with CUDA 12.6 on the H100
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 = sm_90
export MAX_JOBS=8
pip install -e .

cd /home/anangia/maxrl-theory
rm -rf ${BUILD_DIR}

echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"

# ============ Download Models ============
echo "=== Downloading SmolLM2-360M-Instruct ==="
huggingface-cli download HuggingFaceTB/SmolLM2-360M-Instruct \
    --local-dir ${SCRATCH}/models/SmolLM2-360M-Instruct

echo "=== Downloading Qwen3-1.7B-Base ==="
huggingface-cli download Qwen/Qwen3-1.7B-Base \
    --local-dir ${SCRATCH}/models/Qwen3-1.7B-Base

# ============ Preprocess Datasets ============
echo "=== Preprocessing GSM8K ==="
python examples/maxrl_data_preprocess/gsm8k.py \
    --local_dir ${SCRATCH}/data/gsm8k

echo "=== Downloading Maze 17x17 ==="
huggingface-cli download guanning-ai/maze_17x17_1m \
    --repo-type dataset \
    --local-dir ${SCRATCH}/data/maze_17x17

echo "=== Preprocessing Polaris ==="
python examples/maxrl_data_preprocess/polaris.py \
    --local_dir ${SCRATCH}/data/polaris

echo "=== Preprocessing AIME25 ==="
python examples/maxrl_data_preprocess/aime25.py \
    --local_dir ${SCRATCH}/data/aime25

echo "=== Preprocessing Math 500 ==="
python examples/maxrl_data_preprocess/math_500.py \
    --local_dir ${SCRATCH}/data/math500

# ============ Verify Downloads ============
echo ""
echo "=== Verification ==="
echo "--- vLLM ---"
python -c "import vllm; print(f'vllm {vllm.__version__}')"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

echo "--- Models ---"
for model in SmolLM2-360M-Instruct Qwen3-1.7B-Base; do
    if [ -f "${SCRATCH}/models/${model}/config.json" ]; then
        echo "OK: ${model}"
    else
        echo "MISSING: ${model}"
    fi
done

echo "--- Datasets ---"
for dset in gsm8k maze_17x17 polaris aime25 math500; do
    count=$(find ${SCRATCH}/data/${dset} -name "*.parquet" 2>/dev/null | wc -l)
    echo "${dset}: ${count} parquet file(s)"
done

echo ""
echo "=== Setup complete ==="
