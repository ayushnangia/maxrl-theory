# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MaxRL is the official PyTorch implementation of "Maximum Likelihood Reinforcement Learning" (Tajwar, Zeng et al.). It implements scalable RL training for LLMs, built on top of the [verl](https://github.com/verl-project/verl) framework. The package is installed as `verl` (v0.4.0.dev).

Paper: https://arxiv.org/abs/2602.02710

## Setup

```bash
conda create -n maxrl python==3.10 && conda activate maxrl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Flash Attention: build from source (git clone + python setup.py install)
pip install vllm==0.8.4 wandb math-verify
pip install -e .
```

## Common Commands

```bash
# Lint
ruff check verl/

# Run CPU-only tests (suffix convention: _on_cpu.py)
python -m pytest tests/test_protocol_on_cpu.py
python -m pytest tests/trainer/ppo/test_core_algos_on_cpu.py

# Run a specific test
python -m pytest tests/utils/test_torch_functional.py -k "test_name"

# Distributed/GPU tests (require Ray + GPUs)
python -m pytest tests/special_distributed/
python -m pytest tests/special_e2e/

# Run experiments
bash smollm/smollm.sh              # SmolLM on GSM8K
bash maze/maze_17.sh               # 17x17 Maze
bash qwen3_experiments/run_qwen3_training.sh  # Qwen3

# Data preprocessing
python examples/maxrl_data_preprocess/gsm8k.py --local_dir /path/to/output
```

## Architecture

### Core Data Protocol
- **`verl/protocol.py`** — `DataProto` is the unified data container used across all components. Built on TensorDict with numpy array support and auto-padding for distributed training (`VERL_AUTO_PADDING` env var).

### Training Loop (Single Controller Pattern)
The system uses a **single controller** architecture where one trainer process orchestrates distributed workers via Ray:

- **`verl/trainer/ppo/ray_trainer.py`** — `RayTrainer` is the main entry point. It manages `ResourcePoolManager` and coordinates the training loop: rollout → reward → advantage estimation → policy update.
- **`verl/trainer/ppo/core_algos.py`** — Algorithm implementations using **registry pattern**. Advantage estimators (`@register_adv_est`) and policy losses (`@register_policy_loss`) are pluggable. Implements GAE, Maclaurin variants, REINFORCE, PPO clip, etc.
- **`verl/trainer/ppo/maclaurin.py`** — MaxRL's key contribution: Maclaurin weight calculations for variance-reduced advantage estimation (cross-fitted, leave-one-out, oversampled subset variants).

### Distributed Workers (Role-Based)
Workers are organized by `Role` enum (Actor, Rollout, Critic, RefPolicy, RewardModel):

- **`verl/workers/rollout/`** — Generation workers with vLLM and SGLang backends (interchangeable)
- **`verl/workers/actor/`** — Policy model training workers
- **`verl/workers/critic/`** — Value model workers for advantage computation
- **`verl/workers/reward_manager/`** — Reward computation (naive, DAPO, PRIME, batch variants)
- **`verl/workers/sharding_manager/`** — FSDP and Megatron sharding strategies

### Ray Orchestration
- **`verl/single_controller/ray/base.py`** — `RayWorkerGroup` and `RayResourcePool` manage distributed execution
- **`verl/single_controller/base/decorator.py`** — Distributed execution decorators for worker methods

### Model Support
- **`verl/models/transformers/`** — HuggingFace models (LLaMA, Qwen2, Qwen2.5, DeepSeek, vision-language)
- **`verl/models/{llama,qwen2}/megatron/`** — Native Megatron-LM implementations with tensor parallelism
- **`verl/models/registry.py`** — Model registry for runtime model selection

### Configuration
- **Hydra/OmegaConf** for all config management
- YAML configs in `verl/trainer/config/` (base) and `recipe/*/config/` (algorithm-specific)
- Experiment shell scripts in `recipe/*/run_*.sh`, `smollm/`, `maze/`, `qwen3_experiments/`

## Algorithms Implemented

PPO, GRPO, DAPO (decoupled clip + dynamic sampling), SPPO (self-play preference optimization), SPIN, PRIME (process reward models). Each has its own recipe under `recipe/`.

## Linting

Ruff is configured in `pyproject.toml` with rules E, F, UP, B, I, G. Line length is set to 300. Ignored: F405, F403, E731, B007, UP032, UP007, G004.

## Key Patterns

- **Registry pattern** for algorithms — add new advantage estimators or policy losses via `@register_adv_est("name")` / `@register_policy_loss("name")` decorators in `core_algos.py`
- **Test naming convention** — CPU-only tests are suffixed `_on_cpu.py`; distributed tests live in `tests/special_distributed/`; E2E tests in `tests/special_e2e/`
- **Environment variables** — `VERL_AUTO_PADDING` (auto-pad for distributed), `VERL_USE_MODELSCOPE` (ModelScope hub), `RAY_ADDRESS` (Ray cluster)
- **Paprika** (`verl/paprika/`) — Game environments (Wordle, Hangman, Mastermind, etc.) with their own trainer and reward manager
