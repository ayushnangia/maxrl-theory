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

# Phase 0: One-time setup — download models and datasets
# vLLM 0.8.4 + Ray 2.43.0 already installed in ~/maxrl_env

source /home/anangia/maxrl-theory/killarney/common.sh

SCRATCH=/scratch/anangia

# ============ Create Directory Structure ============
echo "=== Creating directories ==="
mkdir -p ${SCRATCH}/{models,data,checkpoints,logs}
mkdir -p ${SCRATCH}/checkpoints/{smollm,maze,qwen3}

# ============ Verify vLLM ============
echo "=== Checking vLLM ==="
python -c "import vllm; print(f'vllm {vllm.__version__}')"
python -c "import ray; print(f'ray {ray.__version__}')"

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
