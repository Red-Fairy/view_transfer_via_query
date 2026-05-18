#!/usr/bin/env bash
# Grouped-guidance sweep: 4 GPUs, 4 chained-guidance configs, one per GPU.
# Every job runs the SAME locations file (full split) with a different
# (geom, src, text) triplet, writing to its own auto-tagged OUT_DIR under
#   infer_out/<run_tag>/ckpt<step>_<pairing>_geom<g>_src<s>_text<t>
#
# All knobs are env vars (see scripts/infer.sh for the full list). Defaults
# target runs/14B_4gpu_640P_0507/checkpoint-10000 on the 0507 test split.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LORA_CKPT="${LORA_CKPT:-${HERE}/../runs/14B_4gpu_640P_0507/checkpoint-10000/trainable_params.pt}"
LOCATIONS_FILE="${LOCATIONS_FILE:-/share/ma/scratch/rundong/Unreal_Projects/split_files/view_transfer_0507/test.txt}"
LOG_DIR="${LOG_DIR:-${HERE}/../infer_out/14B_4gpu_640P_0507/_sweep_logs}"
# nvdiffrast / unicorn-cluster libstdc++ (mesh lift+render path).
SYS_LIBSTDCPP="${SYS_LIBSTDCPP:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

# GPU : "geom src text"  — see proposal: control + geom-axis isolation + balanced-high
declare -a CFG=(
  "3 3 3"      # GPU0  baseline (≡ monolithic g=3, telescoping identity)
  "5 2 3"      # GPU1  geometry-emphasized
  "7 1.5 2"    # GPU2  strong-geom / lean-src
  "5 3 3"      # GPU3  balanced-high
)

mkdir -p "${LOG_DIR}"
echo "Launching ${#CFG[@]} guidance jobs (LORA_CKPT=${LORA_CKPT})"
echo "  locations = ${LOCATIONS_FILE}"
echo "  logs      = ${LOG_DIR}"

# --low_vram keeps the full 14B model resident in CPU RAM and transiently
# ~2x that while the sharded state-dict loads. Under a shared SLURM mem cgroup
# (job_380825 caps at 256 GiB) four simultaneous load peaks OOM-kill jobs.
# So gate each launch: don't start job i+1 until job i is past model-load
# (first per-sample marker in its log) or has exited. Steady-state RSS
# (~40 GB/job) then coexists fine.
STAGGER_GATE="${STAGGER_GATE:-VAE encoding|^gen |\[done\]|\[skip\]}"
STAGGER_TIMEOUT="${STAGGER_TIMEOUT:-900}"   # s; safety cap if a job hangs in load

pids=()
for i in "${!CFG[@]}"; do
  read -r G S T <<< "${CFG[$i]}"
  log="${LOG_DIR}/gpu${i}_geom${G}_src${S}_text${T}.log"
  echo "  GPU${i}: geom=${G} src=${S} text=${T}  → ${log}"
  CUDA_VISIBLE_DEVICES="${i}" \
  LD_PRELOAD="${SYS_LIBSTDCPP}" \
  LOW_VRAM=1 \
  LORA_CKPT="${LORA_CKPT}" \
  LOCATIONS_FILE="${LOCATIONS_FILE}" \
  GUIDANCE_GEOM="${G}" GUIDANCE_SRC="${S}" GUIDANCE_TEXT="${T}" \
  bash "${HERE}/infer.sh" > "${log}" 2>&1 &
  pid=$!
  pids+=("${pid}")
  # Gate the next launch (skip wait after the last job).
  if [ "${i}" -lt $(( ${#CFG[@]} - 1 )) ]; then
    waited=0
    while kill -0 "${pid}" 2>/dev/null \
          && ! grep -qE "${STAGGER_GATE}" "${log}" 2>/dev/null \
          && [ "${waited}" -lt "${STAGGER_TIMEOUT}" ]; do
      sleep 10; waited=$(( waited + 10 ))
    done
    echo "    gate cleared for GPU${i} after ${waited}s"
  fi
done

echo "PIDs: ${pids[*]}"
fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[ok]   GPU${idx} (pid ${pids[$idx]})"
  else
    echo "[FAIL] GPU${idx} (pid ${pids[$idx]}) — see ${LOG_DIR}"
    fail=1
  fi
done
echo "Sweep done (fail=${fail})."
exit "${fail}"
