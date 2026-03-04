#!/bin/bash
# ray_cluster.sh — Ray cluster start/stop functions for Killarney
# Source this file after common.sh:
#   source "$(dirname "$0")/ray_cluster.sh"

# Start a single-node Ray head (uses all GPUs on this node)
start_ray_head() {
    local num_gpus="${1:-${SLURM_GPUS_ON_NODE:-8}}"
    echo "=== Starting Ray head (single-node, ${num_gpus} GPUs) ==="
    ray stop --force 2>/dev/null || true
    ray start --head \
        --num-gpus "${num_gpus}" \
        --num-cpus "${SLURM_CPUS_PER_TASK:-48}" \
        --temp-dir /tmp/ray
    sleep 5
    ray status
    echo "=== Ray head ready ==="
}

# Start a multi-node Ray cluster (head + workers via srun)
start_ray_cluster() {
    echo "=== Starting Ray cluster (${SLURM_JOB_NUM_NODES} nodes) ==="
    ray stop --force 2>/dev/null || true

    # Get node list
    local nodes
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    local nodes_array=($nodes)

    local head_node=${nodes_array[0]}
    local head_node_ip
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # Handle IPv6 → IPv4 conversion
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$head_node_ip"
        if [[ ${#ADDR[0]} -gt 16 ]]; then
            head_node_ip=${ADDR[1]}
        else
            head_node_ip=${ADDR[0]}
        fi
        echo "IPv6 detected, using IPv4: $head_node_ip"
    fi

    local port=6379
    export ip_head="${head_node_ip}:${port}"
    echo "IP Head: $ip_head"

    # Start head node
    echo "Starting HEAD on ${head_node} ($(hostname))"
    ray start --head \
        --node-ip-address="$head_node_ip" \
        --port=$port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus "${SLURM_GPUS_ON_NODE}"
    sleep 10
    ray status
    echo "Head node ready"

    # Start worker nodes
    local worker_num=$((SLURM_JOB_NUM_NODES - 1))
    for ((i = 1; i <= worker_num; i++)); do
        local node_i=${nodes_array[$i]}
        echo "Starting WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL bash -c "
            module load python/3.10.13 cuda/12.6 arrow/17.0.0 opencv/4.11.0
            source ~/maxrl_env/bin/activate
            export NCCL_ASYNC_ERROR_HANDLING=1
            export NCCL_DEBUG=WARN
            export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
            export RAY_TMPDIR=/tmp/ray
            ray start --address '${ip_head}' \
                --num-cpus ${SLURM_CPUS_PER_TASK} \
                --num-gpus ${SLURM_GPUS_ON_NODE}
        " &
        sleep 10
    done

    # Wait for cluster to stabilize
    sleep 20
    echo "=== Ray cluster started (${SLURM_JOB_NUM_NODES} nodes) ==="
    ray status
}

# Stop Ray cleanly
stop_ray() {
    echo "=== Stopping Ray ==="
    ray stop --force 2>/dev/null || true
}

# Cleanup trap — call stop_ray on exit
trap stop_ray EXIT
