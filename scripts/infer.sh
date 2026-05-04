#!/usr/bin/env bash
# Batched inference for ViewTransferDiT. Self-contained: works from any CWD.
#
# All knobs are env vars (override on the command line):
#
#   MODEL_SIZE          1.3B | 14B            (default 14B)
#   LORA_RANK           int                   (default 64; ignored if LORA_CKPT empty)
#   LORA_ALPHA          float                 (default 64.0)
#   NUM_INFERENCE_STEPS int                   (default 50)
#   GUIDANCE_SCALE      float                 (default 1.5)
#   NUM_PER_LOCATION    int                   (default 1)
#   SRC_IDX / TGT_IDX   "00" | "01"           (default "00" / "01")
#   DUAL_PROJECTION     "1" to enable         (default unset)
#
#   LOCATIONS_FILE      path to .txt          (default train_locations_v2.txt)
#   OUT_DIR             path                  (default ${PROJECT_ROOT}/infer_out/<run_tag>)
#   LORA_CKPT           path to .pt           (default ${PROJECT_ROOT}/runs/14B_debut_v2/checkpoint-1600/trainable_params.pt)
#   DIT_CKPT / VAE_CKPT see train.sh; same defaults
#
# Examples:
#   bash scripts/infer.sh
#   GUIDANCE_SCALE=2.5 bash scripts/infer.sh
#   LORA_CKPT=runs/exp/checkpoint-2000/trainable_params.pt OUT_DIR=infer_out/exp-2k bash scripts/infer.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ── Defaults (override via env) ─────────────────────────────────────────────
MODEL_SIZE="${MODEL_SIZE:-14B}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"
NUM_PER_LOCATION="${NUM_PER_LOCATION:-1}"
SRC_IDX="${SRC_IDX:-00}"
TGT_IDX="${TGT_IDX:-01}"
DUAL_PROJECTION="${DUAL_PROJECTION:-}"   # set to any non-empty value to enable

LOCATIONS_FILE="${LOCATIONS_FILE:-/work/nvme/beab/rluo2/viewpoint-transfer/data/train_locations_v2.txt}"

# Auto-tag the output directory by checkpoint step + guidance scale, unless overridden.
LORA_CKPT="${LORA_CKPT:-${PROJECT_ROOT}/runs/14B_debut_v2/checkpoint-1600/trainable_params.pt}"
_ckpt_tag="$(basename "$(dirname "${LORA_CKPT}")" 2>/dev/null || echo unknown)"   # e.g. checkpoint-1600
_g_tag="g${GUIDANCE_SCALE}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/infer_out/${MODEL_SIZE}_v2-${_ckpt_tag}-${_g_tag}}"

# Model-size-specific pretrained-Wan defaults (mirror train.sh)
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

# ── Env hygiene ─────────────────────────────────────────────────────────────
source /work/nvme/beab/rluo2/anaconda3/etc/profile.d/conda.sh
conda activate wan

cd "${PROJECT_ROOT}"
export OPENCV_IO_ENABLE_OPENEXR=1
mkdir -p "${OUT_DIR}"

echo "==================================================================="
echo "  MODEL_SIZE          = ${MODEL_SIZE}"
echo "  DIT_CKPT            = ${DIT_CKPT}"
echo "  VAE_CKPT            = ${VAE_CKPT}"
echo "  LORA_CKPT           = ${LORA_CKPT}"
echo "  LOCATIONS_FILE      = ${LOCATIONS_FILE}"
echo "  OUT_DIR             = ${OUT_DIR}"
echo "  NUM_INFERENCE_STEPS = ${NUM_INFERENCE_STEPS}"
echo "  GUIDANCE_SCALE      = ${GUIDANCE_SCALE}"
echo "  NUM_PER_LOCATION    = ${NUM_PER_LOCATION}"
echo "  SRC/TGT_IDX         = ${SRC_IDX} / ${TGT_IDX}"
[ -n "${DUAL_PROJECTION}" ] && echo "  DUAL_PROJECTION     = on"
echo "==================================================================="

python -m view_transfer_via_query.infer \
    --locations_file        "${LOCATIONS_FILE}" \
    --out_dir               "${OUT_DIR}" \
    --num_per_location      "${NUM_PER_LOCATION}" \
    --src_idx               "${SRC_IDX}" \
    --tgt_idx               "${TGT_IDX}" \
    $([ -n "${DUAL_PROJECTION}" ] && echo "--dual_projection") \
    --dit_ckpt              "${DIT_CKPT}" \
    --vae_ckpt              "${VAE_CKPT}" \
    --lora_ckpt             "${LORA_CKPT}" \
    --model_size            "${MODEL_SIZE}" \
    --lora_rank             "${LORA_RANK}" \
    --lora_alpha            "${LORA_ALPHA}" \
    --num_inference_steps   "${NUM_INFERENCE_STEPS}" \
    --guidance_scale        "${GUIDANCE_SCALE}"
