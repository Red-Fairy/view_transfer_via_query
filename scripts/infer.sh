#!/bin/bash

project_root=${1:-/work/nvme/beab/rluo2/viewpoint-transfer/DiffSynth-Studio}
location_file_path=${2:-//work/nvme/beab/rluo2/viewpoint-transfer/data/train_locations.txt}

python -m view_transfer_via_query.infer \
  --locations_file ${location_file_path} \
  --out_dir ./infer_out/14B-debut-step1600-guidance1 \
  --num_per_location 1 \
  --src_idx 00 --tgt_idx 01 \
  --dit_ckpt ${project_root}/models/Wan-AI/Wan2.1-T2V-14B \
  --vae_ckpt ${project_root}/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
  --lora_ckpt ${project_root}/runs/14B-debut/checkpoint-1600/trainable_params.pt \
  --model_size 14B --lora_rank 64 \
  --num_inference_steps 50 --guidance_scale 1