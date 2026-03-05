#!/bin/bash
#SBATCH --job-name=qwen3-sweep
#SBATCH --account=aip-rgrosse
#SBATCH --partition=gpubase_h100_b4
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/anangia/logs/qwen3_%A_%a.out
#SBATCH --error=/scratch/anangia/logs/qwen3_%A_%a.err

# Phase 3: Qwen3-1.7B on Math — 7-estimator sweep (multi-node, 4x8 H100)

source /home/anangia/maxrl-theory/killarney/common.sh
source /home/anangia/maxrl-theory/killarney/ray_cluster.sh

# ============ Estimator Selection ============
ADVANTAGE_ESTIMATOR=${ESTIMATORS[$SLURM_ARRAY_TASK_ID]}
echo "=== Running estimator: ${ADVANTAGE_ESTIMATOR} (array index ${SLURM_ARRAY_TASK_ID}) ==="

# ============ Paths ============
MODEL_PATH=${MODEL_DIR}/Qwen3-1.7B-Base
MODEL_NAME=Qwen3-1.7B-Base

TRAIN_DATASET_PATH=${DATA_DIR}/polaris/train.parquet
TEST_DATASET_PATH="['${DATA_DIR}/aime25/test.parquet','${DATA_DIR}/math500/test.parquet']"

# ============ Hyperparameters ============
FULL_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=256
NUM_PER_PROMPT_ROLLOUTS=16
MAX_RESPONSE_LENGTH=4096
MAX_PROMPT_LENGTH=1024
LEARNING_RATE=1e-6
REWARD_MANAGER=multi_thread

PER_GPU_MINI_BATCH_SIZE=4
NUM_PER_PROMPT_ROLLOUTS_VALIDATION=32
MAX_MODEL_LEN=32000
MAX_NUM_BATCHED_TOKENS=32000
TENSOR_MODEL_PARALLEL_SIZE=1
PPO_EPOCHS=1

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2
GRAD_CLIP=0.3
KL_COEFF=0.0
TOTAL_EPOCHS=5
SEED=79

PROJECT_NAME=Qwen3_MaxRL_Experiments
EXPERIMENT_NAME=${ADVANTAGE_ESTIMATOR}_${MODEL_NAME}
CHECKPOINT_SAVE_PATH=${CKPT_DIR}/qwen3/${EXPERIMENT_NAME}

# ============ NCCL Tuning for InfiniBand ============
export NCCL_DEBUG=INFO
export NCCL_ALGO=RING
export NCCL_IB_AR_THRESHOLD=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_QPS_PER_CONNECTION=2
export UCX_IB_PCI_RELAXED_ORDERING=on
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=$(
    for dev in $(ls -d /sys/class/infiniband/mlx5_* 2>/dev/null | sort -V); do
        name=$(basename $dev)
        link=$(cat $dev/ports/1/link_layer 2>/dev/null)
        if [[ "$link" == "InfiniBand" ]]; then
            pcie=$(basename $(readlink -f $dev/device))
            echo "$pcie $name"
        fi
    done | sort | awk '{print $2":1"}' | paste -sd,
)
export UCX_NET_DEVICES=eth0
export SEED

# ============ Ray Cluster Setup (multi-node) ============
start_ray_cluster

# ============ Training ============
python3 -W ignore -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADVANTAGE_ESTIMATOR \
    data.train_files=$TRAIN_DATASET_PATH \
    data.val_files=$TEST_DATASET_PATH \
    data.train_batch_size=$FULL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEFF \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$NUM_PER_PROMPT_ROLLOUTS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=$NUM_PER_PROMPT_ROLLOUTS_VALIDATION \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=$KL_COEFF \
    reward_model.reward_manager=$REWARD_MANAGER \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.val_on_last_step=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CHECKPOINT_SAVE_PATH \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=400 \
    trainer.max_critic_ckpt_to_keep=400 \
    trainer.test_freq=50 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    ray_init.ray_dir=/tmp/ray

echo "=== Qwen3 ${ADVANTAGE_ESTIMATOR} complete ==="
