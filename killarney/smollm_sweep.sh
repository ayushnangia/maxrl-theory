#!/bin/bash
#SBATCH --job-name=smollm-sweep
#SBATCH --account=aip-rgrosse
#SBATCH --partition=gpubase_h100_b4
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/anangia/logs/smollm_%A_%a.out
#SBATCH --error=/scratch/anangia/logs/smollm_%A_%a.err

# Phase 1: SmolLM2-360M on GSM8K — 7-estimator sweep (job array)

source /home/anangia/maxrl-theory/killarney/common.sh
source /home/anangia/maxrl-theory/killarney/ray_cluster.sh

# ============ Estimator Selection ============
ADVANTAGE_ESTIMATOR=${ESTIMATORS[$SLURM_ARRAY_TASK_ID]}
echo "=== Running estimator: ${ADVANTAGE_ESTIMATOR} (array index ${SLURM_ARRAY_TASK_ID}) ==="

# ============ Paths ============
MODEL_PATH=${MODEL_DIR}/SmolLM2-360M-Instruct
TRAIN_DATA=${DATA_DIR}/gsm8k/train.parquet
VAL_DATA=${DATA_DIR}/gsm8k/test.parquet

# ============ Hyperparameters ============
TRUNCATE_ORDER=64
LR=1e-5
N_ROLLOUTS=128
N_VAL=32

PROJECT_NAME=MaxRL_SmolLM-360M
EXPERIMENT_NAME=${ADVANTAGE_ESTIMATOR}_${N_ROLLOUTS}rollouts

# ============ Ray Setup (single-node) ============
start_ray_head 8

# ============ Training ============
python3 -m verl.trainer.main_ppo \
  ray_init.ray_dir=/tmp/ray \
  algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
  algorithm.use_kl_in_reward=False \
  algorithm.pass_k=${TRUNCATE_ORDER} \
  algorithm.truncate_order=${TRUNCATE_ORDER} \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${VAL_DATA} \
  data.train_batch_size=256 \
  data.filter_overlong_prompts=True \
  data.max_prompt_length=512 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=${N_VAL} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  algorithm.kl_ctrl.kl_coef=0.0 \
  reward_model.reward_manager=multi_thread \
  +reward_model.reward_kwargs.num_reward_actors=64 \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.default_local_dir=${CKPT_DIR}/smollm/${EXPERIMENT_NAME} \
  trainer.total_epochs=200

echo "=== SmolLM ${ADVANTAGE_ESTIMATOR} complete ==="
