#!/bin/bash
# Run viz_mesh_render.py across 4 GPUs in parallel.  Each shard uses the same
# --seed so they generate the same global plan; each renders a disjoint subset
# (i % num_shards == shard_idx).  Requires nvdiffrast (CUDA only).

set -e

DATA_ROOT="${DATA_ROOT:-/share/ma/scratch/rundong/Unreal_Projects/outputs_arranged}"
OUT_DIR="${OUT_DIR:-/home/rl897/viewpoint-transfer/wan/DiffSynth-Studio/view_transfer_via_query/viz_mesh_render}"
PYTHON="${PYTHON:-/home/rl897/anaconda3/envs/wan-view-transfer/bin/python}"
NUM_SAME="${NUM_SAME:-50}"
NUM_DIFF="${NUM_DIFF:-50}"
NUM_FRAMES="${NUM_FRAMES:-81}"
EQUI_H="${EQUI_H:-512}"
EQUI_W="${EQUI_W:-1024}"
EQUI_LOAD_H="${EQUI_LOAD_H:-2048}"
EQUI_LOAD_W="${EQUI_LOAD_W:-4096}"
PERS_H="${PERS_H:-480}"
PERS_W="${PERS_W:-832}"
MESH_FACE_RES="${MESH_FACE_RES:-1024}"
SRC_IDX="${SRC_IDX:-00}"
TGT_IDX="${TGT_IDX:-01}"
SEED="${SEED:-0}"
N=4

# nvdiffrast's C++ ext needs GLIBCXX_3.4.32; the conda env's libstdc++.so.6 only
# provides up to 3.4.29, so preload the system libstdc++ for these subprocesses.
SYS_LIBSTDCPP="${SYS_LIBSTDCPP:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

cd /home/rl897/viewpoint-transfer/wan/DiffSynth-Studio
mkdir -p "$OUT_DIR"

echo "Launching $N shards into $OUT_DIR (mesh-rendered tgt perspective)"
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i \
  LD_PRELOAD="$SYS_LIBSTDCPP" \
  $PYTHON -m view_transfer_via_query.tools.viz_mesh_render \
    --data_root    "$DATA_ROOT" \
    --out_dir      "$OUT_DIR" \
    --num_same     "$NUM_SAME" --num_diff "$NUM_DIFF" \
    --num_frames   "$NUM_FRAMES" \
    --equi_h       "$EQUI_H" --equi_w "$EQUI_W" \
    --equi_load_h  "$EQUI_LOAD_H" --equi_load_w "$EQUI_LOAD_W" \
    --pers_h       "$PERS_H" --pers_w "$PERS_W" \
    --mesh_face_res "$MESH_FACE_RES" \
    --src_idx      "$SRC_IDX" --tgt_idx "$TGT_IDX" \
    --seed         "$SEED" \
    --num_shards   $N --shard_idx $i \
    > "$OUT_DIR/shard_${i}.log" 2>&1 &
  pids+=($!)
  echo "  shard $i started (pid ${pids[-1]}, GPU $i, log $OUT_DIR/shard_${i}.log)"
done

echo "Waiting for all shards..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
if [[ $fail -ne 0 ]]; then
  echo "ERROR: at least one shard failed; check $OUT_DIR/shard_*.log"
  exit 1
fi

echo "DONE: $(ls $OUT_DIR/{same,diff}_*.mp4 2>/dev/null | wc -l) videos in $OUT_DIR"
