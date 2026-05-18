#!/usr/bin/env bash
# DeepSpeed-driven launch for ViewTransferDiT.  Single-node by default; for
# multi-node, see scripts/train_multinode.sh which sets MACHINE_RANK / NUM_MACHINES /
# MAIN_PROCESS_IP per srun task and then re-enters this script.
#
# Self-contained: works from any CWD. All paths derive from script location
# via _common.sh, which exports PROJECT_ROOT (= view_transfer_via_query/) and
# DIFFSYNTH_ROOT (= DiffSynth-Studio/, where pretrained Wan + diffsynth lib live).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SCRIPT_DIR="${PROJECT_ROOT}/scripts"
TMPL_DIR="${PROJECT_ROOT}/configs/accelerate"
DS_CONFIG_DIR="${PROJECT_ROOT}/configs/deepspeed"

# ── Defaults (override via env) ─────────────────────────────────────────────
MODEL_SIZE="${MODEL_SIZE:-14B}"   # 1.3B | 14B
LOCATIONS_FILE="${LOCATIONS_FILE:-/work/nvme/beab/rluo2/viewpoint-transfer/data/split_files/train.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${MODEL_SIZE}_test}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-81}"
PERS_H="${PERS_H:-480}"
PERS_W="${PERS_W:-832}"
LORA_RANK="${LORA_RANK:-64}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-2}"   # batches in-flight on the side CUDA stream
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_EVERY="${SAVE_EVERY:-200}"
LOG_EVERY="${LOG_EVERY:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
SEED="${SEED:-}"   # empty → train.py draws a fresh random seed (logged at startup)
RESUME="${RESUME:-}"            # empty → fresh run; else path to a checkpoint dir (same world size + ZeRO)
INIT_TRAINABLE="${INIT_TRAINABLE:-}"   # empty → no warm-start; else path to trainable_params.pt
LOG_VIDEO_EVERY="${LOG_VIDEO_EVERY:-0}"
LOG_VIDEO_FPS="${LOG_VIDEO_FPS:-16}"
PROFILE_STEPS="${PROFILE_STEPS:-0}"
PROFILE_SHAPES="${PROFILE_SHAPES:-0}"
USE_MESH="${USE_MESH:-1}"       # cubemap-mesh lift+render (default ON, matches infer.sh); set USE_MESH= (empty) for point-cloud
TRAIN_DTYPE="${TRAIN_DTYPE:-bf16}"
DATA_DTYPE="${DATA_DTYPE:-bf16}"

# ── Distributed config (single-node by default) ─────────────────────────────
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DS_CONFIG_DIR}/zero2_bf16.json}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

# Model-size-specific checkpoint defaults (live under DIFFSYNTH_ROOT/models/Wan-AI/)
case "${MODEL_SIZE}" in
  1.3B)
    DIT_CKPT="${DIT_CKPT:-${DIFFSYNTH_ROOT}/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors}"
    VAE_CKPT="${VAE_CKPT:-${DIFFSYNTH_ROOT}/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth}"
    ;;
  14B)
    DIT_CKPT="${DIT_CKPT:-${DIFFSYNTH_ROOT}/models/Wan-AI/Wan2.1-T2V-14B}"
    VAE_CKPT="${VAE_CKPT:-${DIFFSYNTH_ROOT}/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth}"
    ;;
  *) echo "Unknown MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

# Outputs / configs / accelerate.yaml live under PROJECT_ROOT. Cd there so any
# relative paths (in templates, configs) resolve under the project tree, not under
# wherever the user happened to launch from.
cd "${PROJECT_ROOT}"

# PYTHONPATH already set by _common.sh (DIFFSYNTH_ROOT — makes both diffsynth.X
# and view_transfer_via_query.X importable).
export OPENCV_IO_ENABLE_OPENEXR=1
# Reduce CUDA caching-allocator fragmentation; without this, ZeRO-2 + grad-ckpt
# at 14B+481×832×81 leaves ~13 GB reserved-but-unallocated and OOMs around step 3.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}" "${OUTPUT_DIR}"

# Default to all visible GPUs on this node
GPUS_PER_NODE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_MACHINES * GPUS_PER_NODE))}"

# ── Render accelerate.yaml from template ────────────────────────────────────
ACCEL_YAML="${OUTPUT_DIR}/accelerate_rank${MACHINE_RANK}.yaml"
sed -e "s|__DS_CONFIG__|${DEEPSPEED_CONFIG}|g" \
    -e "s|__MACHINE_RANK__|${MACHINE_RANK}|g" \
    -e "s|__NUM_MACHINES__|${NUM_MACHINES}|g" \
    -e "s|__NUM_PROCESSES__|${NUM_PROCESSES}|g" \
    -e "s|__MAIN_IP__|${MAIN_PROCESS_IP}|g" \
    -e "s|__MAIN_PORT__|${MAIN_PROCESS_PORT}|g" \
    "${TMPL_DIR}/deepspeed.yaml.tmpl" > "${ACCEL_YAML}"

# ── Launch ──────────────────────────────────────────────────────────────────
echo "==================================================================="
echo "  MODEL_SIZE       = ${MODEL_SIZE}"
echo "  DIT_CKPT         = ${DIT_CKPT}"
echo "  VAE_CKPT         = ${VAE_CKPT}"
echo "  LOCATIONS_FILE   = ${LOCATIONS_FILE}"
echo "  OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "  RES              = ${PERS_H}x${PERS_W}x${NUM_VIDEO_FRAMES}"
echo "  LORA_RANK        = ${LORA_RANK} (0 = full FT)"
echo "  MAX_STEPS        = ${MAX_STEPS}, LR=${LR}, BSZ=${BATCH_SIZE}, GA=${GRADIENT_ACCUMULATION_STEPS}"
echo "  TRAIN_DTYPE      = ${TRAIN_DTYPE}, DATA_DTYPE = ${DATA_DTYPE}"
echo "  DEEPSPEED_CONFIG = ${DEEPSPEED_CONFIG}"
echo "  NUM_MACHINES     = ${NUM_MACHINES} (rank ${MACHINE_RANK}, main=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT})"
echo "  NUM_PROCESSES    = ${NUM_PROCESSES}"
echo "  ACCELERATE_YAML  = ${ACCEL_YAML}"
[ -n "${RESUME}" ] && echo "  RESUME           = ${RESUME}"
echo "==================================================================="

accelerate launch \
    --config_file "${ACCEL_YAML}" \
    --module view_transfer_via_query.train \
        --locations_file               "${LOCATIONS_FILE}" \
        --pretrained_dit               "${DIT_CKPT}" \
        --vae_ckpt                     "${VAE_CKPT}" \
        --model_size                   "${MODEL_SIZE}" \
        --output_dir                   "${OUTPUT_DIR}" \
        --lora_rank                    "${LORA_RANK}" \
        --lr                           "${LR}" \
        --weight_decay                 "${WEIGHT_DECAY}" \
        --gradient_accumulation_steps  "${GRADIENT_ACCUMULATION_STEPS}" \
        --batch_size                   "${BATCH_SIZE}" \
        --num_workers                  "${NUM_WORKERS}" \
        --max_steps                    "${MAX_STEPS}" \
        --save_every                   "${SAVE_EVERY}" \
        --log_every                    "${LOG_EVERY}" \
        --warmup_steps                 "${WARMUP_STEPS}" \
        --num_video_frames             "${NUM_VIDEO_FRAMES}" \
        --pers_h                       "${PERS_H}" \
        --pers_w                       "${PERS_W}" \
        $([ -n "${SEED}" ]            && echo "--seed ${SEED}") \
        $([ -n "${RESUME}" ]          && echo "--resume ${RESUME}") \
        $([ -n "${INIT_TRAINABLE}" ]  && echo "--init_trainable ${INIT_TRAINABLE}") \
        --log_video_every              "${LOG_VIDEO_EVERY}" \
        --log_video_fps                "${LOG_VIDEO_FPS}" \
        --profile_steps                "${PROFILE_STEPS}" \
        $([ "${PROFILE_SHAPES}" = "1" ] && echo "--profile_shapes") \
        $([ -n "${USE_MESH}" ]        && echo "--use_mesh") \
        --train_dtype                  "${TRAIN_DTYPE}" \
        --data_dtype                   "${DATA_DTYPE}" \
        --gradient_checkpointing
