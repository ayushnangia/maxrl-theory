#!/bin/bash
#SBATCH --job-name=maze-sweep
#SBATCH --account=aip-rgrosse
#SBATCH --partition=gpubase_h100_b2
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/anangia/logs/maze_%A_%a.out
#SBATCH --error=/scratch/anangia/logs/maze_%A_%a.err

# Phase 2: Maze 17x17 — 7-estimator sweep (job array)
# Uses HF rollout backend (no vllm needed)

source /home/anangia/maxrl-theory/killarney/common.sh
source /home/anangia/maxrl-theory/killarney/ray_cluster.sh

# ============ Estimator Selection ============
ADVANTAGE_ESTIMATOR=${ESTIMATORS[$SLURM_ARRAY_TASK_ID]}
echo "=== Running estimator: ${ADVANTAGE_ESTIMATOR} (array index ${SLURM_ARRAY_TASK_ID}) ==="

# ============ Paths ============
# Maze SFT model is bundled in the repo
MODEL_PATH=/home/anangia/maxrl-theory/maze/ckpt-1500
TRAIN_DATA=${DATA_DIR}/maze_17x17/train.parquet
VAL_DATA=${DATA_DIR}/maze_17x17/test.parquet

# ============ Hyperparameters ============
TRUNCATE_ORDER=64
LR=1e-4
N_ROLLOUTS=128
N_VAL=2048
TRAIN_BATCH_SIZE=256

PROJECT_NAME=MaxRL_Maze_17x17
EXPERIMENT_NAME=${ADVANTAGE_ESTIMATOR}_${N_ROLLOUTS}rollouts

# ============ Ray Setup (single-node, 4 GPUs) ============
start_ray_head 4

# ============ Training ============
python3 -m verl.trainer.main_ppo \
  ray_init.ray_dir=/tmp/ray \
  algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
  algorithm.use_kl_in_reward=False \
  algorithm.pass_k=4 \
  algorithm.truncate_order=${TRUNCATE_ORDER} \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${VAL_DATA} \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=320 \
  data.max_response_length=180 \
  data.apply_chat_template=False \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.dtype=float16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4096 \
  actor_rollout_ref.rollout.name=hf \
  +actor_rollout_ref.rollout.micro_batch_size=4096 \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4096 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4096 \
  actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
  actor_rollout_ref.rollout.val_kwargs.n=${N_VAL} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  reward_model.reward_manager=prime \
  +reward_model.reward_kwargs.num_processes=64 \
  +reward_model.reward_kwargs.chunksize=64 \
  algorithm.kl_ctrl.kl_coef=0.0 \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=250 \
  trainer.test_freq=250 \
  trainer.max_actor_ckpt_to_keep=300 \
  trainer.default_local_dir=${CKPT_DIR}/maze/${EXPERIMENT_NAME} \
  trainer.total_epochs=10

echo "=== Maze ${ADVANTAGE_ESTIMATOR} complete ==="
