#!/bin/bash
# Run viz_trajectories.py across 4 GPUs in parallel.
set -e

LOCATIONS_FILE="${LOCATIONS_FILE:-/share/ma/scratch/rundong/Unreal_Projects/split_files/train.txt}"
OUT_DIR="${OUT_DIR:-/home/rl897/viewpoint-transfer/wan/DiffSynth-Studio/view_transfer_via_query/viz_trajectories}"
PYTHON="${PYTHON:-/home/rl897/anaconda3/envs/wan-view-transfer/bin/python}"
NUM_SAME="${NUM_SAME:-50}"
NUM_DIFF="${NUM_DIFF:-50}"
NUM_FRAMES="${NUM_FRAMES:-81}"
EQUI_H="${EQUI_H:-512}"
EQUI_W="${EQUI_W:-1024}"
TOTAL_VIDEO_FRAMES="${TOTAL_VIDEO_FRAMES:-160}"
MIN_OVERLAP="${MIN_OVERLAP:-0.25}"
SEED="${SEED:-0}"
N=4

cd /home/rl897/viewpoint-transfer/wan/DiffSynth-Studio
mkdir -p "$OUT_DIR"

echo "Launching $N shards into $OUT_DIR"
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i \
  $PYTHON -m view_transfer_via_query.tools.viz_trajectories \
    --locations_file "$LOCATIONS_FILE" \
    --out_dir   "$OUT_DIR" \
    --num_same  "$NUM_SAME" --num_diff "$NUM_DIFF" \
    --num_frames "$NUM_FRAMES" --equi_h "$EQUI_H" --equi_w "$EQUI_W" \
    --total_video_frames "$TOTAL_VIDEO_FRAMES" \
    --min_overlap "$MIN_OVERLAP" \
    --seed "$SEED" \
    --num_shards $N --shard_idx $i \
    > "$OUT_DIR/shard_${i}.log" 2>&1 &
  pids+=($!)
  echo "  shard $i started (pid ${pids[-1]}, GPU $i)"
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
