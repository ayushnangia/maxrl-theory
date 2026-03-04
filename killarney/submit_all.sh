#!/bin/bash
# submit_all.sh — Step-by-step submission for Killarney
#
# Usage:
#   bash killarney/submit_all.sh          # Show what would be submitted
#   bash killarney/submit_all.sh step0    # Submit setup only
#   bash killarney/submit_all.sh step1    # Submit CIFAR-10 (cheapest, no deps)
#   bash killarney/submit_all.sh step2    # Submit Maze sweep (after setup)
#   bash killarney/submit_all.sh step3    # Submit SmolLM sweep (after setup)
#   bash killarney/submit_all.sh step4    # Submit Qwen3 sweep (biggest, last)
#
# Recommended order: step0 → wait for completion → step1 + step2 → step3 → step4

set -euo pipefail

# Slurm requires submitting from /scratch, not /home
SCRIPT_DIR=/scratch/anangia/killarney
cd /scratch/anangia

# Ensure log directory exists
mkdir -p /scratch/anangia/logs

STEP="${1:-help}"

case "$STEP" in
    step0)
        echo "=== Step 0: Setup (build vllm from source + download data) ==="
        echo "This runs on a compute node (1 H100, 3h)"
        JOB0=$(sbatch --parsable killarney/setup_env.sh)
        echo "Submitted setup job: ${JOB0}"
        echo "Monitor: tail -f /scratch/anangia/logs/setup_${JOB0}.out"
        echo ""
        echo "When done, run: bash killarney/submit_all.sh step1"
        ;;

    step1)
        echo "=== Step 1: CIFAR-10 experiments (cheapest, 1-4 GPUs, 3h) ==="
        echo "No data dependencies — can run even before setup finishes"
        JOB_C1=$(sbatch --parsable killarney/cifar10_experiments.sh)
        JOB_C2=$(sbatch --parsable killarney/cifar10_ddp.sh)
        echo "Submitted CIFAR-10 single-GPU: ${JOB_C1}"
        echo "Submitted CIFAR-10 DDP:        ${JOB_C2}"
        echo ""
        echo "Monitor: tail -f /scratch/anangia/logs/cifar10_*.out"
        ;;

    step2)
        echo "=== Step 2: Maze 17x17 sweep (4 GPUs x 7 estimators, 12h each) ==="
        echo "Make sure setup (step0) completed successfully first!"
        read -p "Enter setup job ID to add dependency (or 'none'): " SETUP_JOB
        if [[ "$SETUP_JOB" == "none" ]]; then
            JOB_M=$(sbatch --parsable killarney/maze_sweep.sh)
        else
            JOB_M=$(sbatch --parsable --dependency=afterok:${SETUP_JOB} killarney/maze_sweep.sh)
        fi
        echo "Submitted Maze sweep (7 array jobs): ${JOB_M}"
        echo "Monitor: tail -f /scratch/anangia/logs/maze_${JOB_M}_*.out"
        ;;

    step3)
        echo "=== Step 3: SmolLM GSM8K sweep (8 GPUs x 7 estimators, 3 days each) ==="
        echo "Make sure setup (step0) completed successfully first!"
        read -p "Enter setup job ID to add dependency (or 'none'): " SETUP_JOB
        if [[ "$SETUP_JOB" == "none" ]]; then
            JOB_S=$(sbatch --parsable killarney/smollm_sweep.sh)
        else
            JOB_S=$(sbatch --parsable --dependency=afterok:${SETUP_JOB} killarney/smollm_sweep.sh)
        fi
        echo "Submitted SmolLM sweep (7 array jobs): ${JOB_S}"
        echo "Monitor: tail -f /scratch/anangia/logs/smollm_${JOB_S}_*.out"
        ;;

    step4)
        echo "=== Step 4: Qwen3-1.7B math sweep (32 GPUs x 7 estimators, 3 days each) ==="
        echo "This is the BIGGEST job (~10,752 GPU-hours total)."
        echo "Consider running only a subset of estimators first."
        echo "Make sure setup (step0) completed successfully first!"
        read -p "Enter setup job ID to add dependency (or 'none'): " SETUP_JOB
        if [[ "$SETUP_JOB" == "none" ]]; then
            JOB_Q=$(sbatch --parsable killarney/qwen3_sweep.sh)
        else
            JOB_Q=$(sbatch --parsable --dependency=afterok:${SETUP_JOB} killarney/qwen3_sweep.sh)
        fi
        echo "Submitted Qwen3 sweep (7 array jobs): ${JOB_Q}"
        echo "Monitor: tail -f /scratch/anangia/logs/qwen3_${JOB_Q}_*.out"
        ;;

    help|*)
        echo "MaxRL Killarney Experiment Submission"
        echo "====================================="
        echo ""
        echo "Recommended order (step-by-step, cheapest first):"
        echo ""
        echo "  Step 0: bash killarney/submit_all.sh step0"
        echo "          Setup: build vllm from source + download data"
        echo "          Resources: 1 H100, 3h"
        echo ""
        echo "  Step 1: bash killarney/submit_all.sh step1"
        echo "          CIFAR-10 experiments (no data deps)"
        echo "          Resources: 1-4 H100s, 3h      (~15 GPU-hours)"
        echo ""
        echo "  Step 2: bash killarney/submit_all.sh step2"
        echo "          Maze 17x17 × 7 estimators"
        echo "          Resources: 4 H100s × 7, 12h   (~168 GPU-hours)"
        echo ""
        echo "  Step 3: bash killarney/submit_all.sh step3"
        echo "          SmolLM GSM8K × 7 estimators"
        echo "          Resources: 8 H100s × 7, 3d    (~2,688 GPU-hours)"
        echo ""
        echo "  Step 4: bash killarney/submit_all.sh step4"
        echo "          Qwen3-1.7B math × 7 estimators"
        echo "          Resources: 32 H100s × 7, 3d   (~10,752 GPU-hours)"
        echo ""
        echo "Check status: squeue -u \$USER"
        echo "Cancel all:   scancel -u \$USER"
        ;;
esac
