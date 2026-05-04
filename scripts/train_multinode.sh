#!/usr/bin/env bash
# Slurm-aware multi-node launcher. Per-node, srun a single task that re-enters
# scripts/train.sh with MACHINE_RANK/NUM_MACHINES/MAIN_PROCESS_IP set; train.sh
# renders its own accelerate.yaml and runs `accelerate launch --config_file ...`.
#
# Usage (after setting/overriding the SBATCH header to your cluster):
#     sbatch scripts/train_multinode.sh
# or interactive:
#     SLURM_NNODES=2 SLURM_JOB_NODELIST="node1,node2" \
#         bash scripts/train_multinode.sh
#
# Override OUTPUT_DIR / LOCATIONS_FILE / MAX_STEPS / MODEL_SIZE / etc. via env.

#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=view_transfer_mn
#SBATCH --output=logs/%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/view_transfer_via_query/scripts"

# ── Cluster topology from slurm ─────────────────────────────────────────────
: "${SLURM_NNODES:?must be set (run under sbatch/srun, or set manually)}"
: "${SLURM_JOB_NODELIST:?must be set}"
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
NUM_MACHINES="${SLURM_NNODES}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))

# Use the first node's IP as the rendezvous endpoint.
MAIN_PROCESS_IP=$(getent hosts "${NODES[0]}" | awk '{print $1}' | head -n1)
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/mn_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}/logs"

echo "Multi-node launch:"
echo "  NUM_MACHINES   = ${NUM_MACHINES}"
echo "  GPUS_PER_NODE  = ${GPUS_PER_NODE}"
echo "  NUM_PROCESSES  = ${NUM_PROCESSES}"
echo "  MAIN           = ${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
echo "  NODES          = ${NODES[*]}"
echo "  OUTPUT_DIR     = ${OUTPUT_DIR}"

# Build CUDA_VISIBLE_DEVICES "0,1,...,GPUS_PER_NODE-1"
CVD=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

# Per-node srun
for i in "${!NODES[@]}"; do
    srun --nodelist="${NODES[i]}" --ntasks=1 --nodes=1 \
        env \
            NUM_MACHINES="${NUM_MACHINES}" \
            NUM_PROCESSES="${NUM_PROCESSES}" \
            MACHINE_RANK="${i}" \
            MAIN_PROCESS_IP="${MAIN_PROCESS_IP}" \
            MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
            CUDA_VISIBLE_DEVICES="${CVD}" \
            OUTPUT_DIR="${OUTPUT_DIR}" \
        bash "${SCRIPT_DIR}/train.sh" \
        > "${OUTPUT_DIR}/logs/node_${i}.log" 2>&1 &
done
wait
echo "All ranks finished."
