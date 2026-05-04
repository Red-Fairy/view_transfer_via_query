# View Transfer via Query (Stage 2) — TODO

See PLAN.md for full architecture spec.  This file tracks progress only.

## Done
- [x] **Step 1** — `model.py`: ViewTransferDiT + LoRA + weight loading from Wan2.1-T2V-14B
- [x] **Step 2** — utilities: `plucker.py`, `mask_utils.py`
- [x] **Step 3** — `tests/test_model.py`: shape, gradient, freeze/LoRA, KQ-bias
- [x] **Step 4** — `dataset.py`: ONLINE pipeline (workers do PNG/EXR I/O only; raw 360 windows + sampled trajectories + single-frame static RGB+depth at t0)
- [x] **Step 4b** — `prepare_data/`: parse_cameras (UE→OpenCV), encode_text (T5), agent_mask, extract_perspectives (equi2pers), lift_and_render (online point-cloud render), encode_latents (VAE), video_io, run_prep
- [x] **Step 4c** — `gpu_preprocess.py` + `prefetcher.py`: side-stream GPU preprocessing overlapping with model fwd/bwd
- [x] **Step 5** — `train.py`: Accelerate, flow-matching v-prediction, CFG dropout, LoRA/full-FT, gradient checkpointing, BF16, cosine LR + warmup
- [x] **Tests** — 66 unit tests passing across model, dataset, train, prefetcher, pipeline, prepare_data
- [x] Settled `blob_dir` convention: `Pano_{tgt}/blobs/` — matches actual UE data; consistent across `dataset.py`, `run_prep.py` docstring, and tests

## Done (cont'd)
- [x] **Step 6** — `pipeline.py`: `ViewTransferPipeline` (flow-matching sampling, CFG, VAE decode) + 6 passing tests
- [x] Fix latent bug in `compute_plucker_at_latent_timestamps` — accept K as `[B, 4]` in addition to time-varying formats
- [x] `gpu_preprocess` supports inference path (skips target encoding when `rgb_tgt_360` absent)
- [x] Fix dtype promotion in `train.training_step` (sigmas → target dtype) and `pipeline.generate` (cond → model dtype, z re-cast after scheduler.step)
- [x] **End-to-end smoke test** (`tests/smoke_test.py`) PASSED on real UE data with Wan2.1-T2V-1.3B + LoRA

## Active
- [ ] Encode real T5 prompt (replace placeholder `text_emb.pt`) before full training run

## Future
- [ ] Stage 1 (panorama prediction model)
- [ ] EXR depth → fp16 packed tensor for faster I/O
