# Changelog — view_transfer_via_query

## 2026-04-29 — Batch data prep across all scenes
- Added `prepare_data/prep_all_scenes.py`: walks a scene root, matches each scene folder to a key in a prompts JSON via longest-prefix match (e.g. `Desert_0car_2-8pp_task200` → `Desert`), and runs cameras + T5 text encoding for every location. T5 is loaded once and unique prompts are cached, so cost scales with #scenes not #locations.
- **Fix**: `prepare_data/encode_text.load_wan_text_encoder` now supports both `.safetensors` (via `safetensors.torch.load_file`) and `.pth` (with `weights_only=False`) — Wan T5 .pth tripped on `weights_only=True`.
- Ran on `outputs_non_arranged_cars_v2`: 14 scenes / 56 locations, 13 unique prompts encoded (RuralAustralia_Road and RuralAustralia_Wood share). All locations now have real T5 embeddings (shapes 82–131 × 4096 depending on prompt length) replacing placeholders.
- Smoke test re-verified with real T5 embeddings: forward loss=0.169, decode shape (1, 3, 17, 128, 224).

## 2026-04-29 — End-to-end smoke test on real UE data
- Added `tests/smoke_test.py`: standalone script exercising the full pipeline (build Wan2.1-T2V-1.3B + LoRA → load VAE → discover entries → load batch → online preprocess → train step fwd/bwd → inference 2-step sampling + VAE decode) on a real location.
- Smoke test PASSED on `outputs_non_arranged_cars_v2/Desert_0car_2-8pp_task200/x-30184_y43482_s1200_m8_v0_n2_p2_p2`: 4 valid pairings discovered, train step loss=0.198, 154 trainable params with non-zero grads, decoded video shape (1, 3, 17, 128, 224) uint8.
- **Fix**: `dataset.SampleEntry.blob_dir` reverted to `Pano_{tgt}/blobs` to match actual UE data layout (run_prep.py docstring was stale).
- **Fix**: `train.training_step` casts `scheduler.sigmas` to `target_latent.dtype` to avoid float32→BF16 promotion when running with explicit BF16 weights.
- **Fix**: `pipeline.generate` casts cond inputs to model dtype and re-casts `z` after `scheduler.step` (which internally promotes via float32 sigma).
- Updated `tests/test_dataset.py` for new blob layout: 4 pairings now valid instead of 2.

## 2026-04-29 — Inference pipeline (Step 6) + bug fixes
- Created `pipeline.py`: `ViewTransferPipeline` with `from_pretrained` (DiT + VAE + LoRA loading), flow-matching sampling loop, multi-condition CFG (zero-all-conds uncond), VAE decode.
- Created `tests/test_pipeline.py`: 6 tests covering uncond construction, CFG scale variation, latent / video output shapes, target-ignore behavior. All pass.
- `gpu_preprocess`: now skips target encoding when `rgb_tgt_360` is absent (inference path).
- **Fix**: `compute_plucker_at_latent_timestamps` now accepts K as `[B, 4]` (constant intrinsics) in addition to `[B, T, 4]` / `[B, T, 3, 3]`. Bug was latent — only triggered through `gpu_preprocess` which passes `[B, 4]` and was untested end-to-end.
- **Fix**: `SampleEntry.blob_dir` corrected to `blob_360_{src}_to_{tgt}` (was wrongly returning `Pano_{tgt}/blobs`); now matches `run_prep.py` and the dataset docstring.
- All 66 tests passing.


## 2026-04-29 — Online lift-and-render
- **Added** `prepare_data/lift_and_render.py`: equirect (RGB + radial depth) → world point cloud → z-buffer rasterize to target perspective. Uses `scatter_reduce(amin)` for the depth buffer; chunked over target frames for memory. Output: rendered RGB + visibility mask, both in target perspective (skips the 360 detour).
- **Updated** `dataset.py`: removed `rendered_360_*` / `visibility_360_*` directories from required layout; added single-frame load of `Pano_{src}_static/rgb/*.png` + `Pano_{src}_static/depth/*` at the chosen `t0`. Depth loader accepts `.exr` / `.npy` / `.pt`.
- **Updated** `gpu_preprocess.py`: replaced equi2pers of pre-rendered 360 with `lift_and_render` call, producing rendered+visibility directly at target perspective resolution.
- **Updated** `run_prep.py verify`: required dirs now reflect the simpler layout (no rendered/visibility 360, but include `Pano_XX_static/depth`).
- **Why**: pre-computing lift-and-render forced `t0=0` for all training samples, wasting 2/3 of every 240-frame sequence. Online unlocks 160× more temporal windows per pairing.
- 13 new lift-and-render tests; **60/60 total tests passing**.

## 2026-04-29 — Online encoding refactor
- **Replaced** `dataset.py`: now an online dataset that workers (CPU) use to load raw 360 equirect uint8 windows and pre-computed camera/text tensors. Auto-discovers training entries via both Pano_00→01 and Pano_01→00 pairings. New `collate_view_transfer` handles ragged text embeddings via right-padding.
- **Added** `gpu_preprocess.py`: GPU-side equi2pers + VAE encode + plücker compute + mask packing. Consumes the dataset's batch dict, produces the model's input dict.
- **Added** `prefetcher.py`: `CUDAStreamPrefetcher` runs `gpu_preprocess` on a side CUDA stream so batch N+1 preprocessing overlaps with batch N's model fwd/bwd.
- **Updated** `train.py`: now wraps the DataLoader in `CUDAStreamPrefetcher`, loads VAE once at startup, and CFG-dropout operates on the post-preprocess GPU dict.
- **Slimmed** `prepare_data/run_prep.py`: now a 3-subcommand CLI (`cameras` / `text` / `verify`) handling only the offline pieces. Equi2pers + VAE encode + mask diff are all gone (moved to dataset / gpu_preprocess).
- **Renamed** `compute_masks.py` → `agent_mask.py` with clarifying docstring: NOT the model's mask channel; just an agent-detection utility for blob-video generation.
- **New tests**: `test_dataset.py` (6), `test_prefetcher.py` (5). Total **46/46 tests passing**.

## 2026-04-29 — Design clarification + PLAN.md
- Added `PLAN.md` documenting the locked-in architecture, conditioning streams, hybrid online/pre-compute split, coordinate conventions, and open work.
- **Caught bug**: `prepare_data/compute_masks.py` computes an *agent-detection* mask (diff of dynamic vs static panorama), but the model's mask channel is the **visibility mask** from the lift-and-render (which pixels of the warped panorama are valid). Visibility mask is now in user's pre-compute scope (produced alongside `rendered_360`).
- Decision: keep online VAE encoding given NVMe storage; T5 + lift-and-render + 360 blobs stay pre-computed.
- Pending refactor: rename/repurpose `compute_masks.py`, slim `run_prep.py` down, refactor `dataset.py` for online encoding with prefetcher.

## 2026-04-28 — Data preparation pipeline (Step 4: prep)
- Created `prepare_data/parse_cameras.py`: UE camera_params.json → OpenCV c2w + intrinsics. Handles LHS-Z-up → RHS-Y-down handedness flip and centimeters → meters. 9 unit tests.
- Created `prepare_data/extract_perspectives.py`: equi2pers projection (custom GPU implementation via F.grid_sample). Includes random trajectory sampler (FOV / yaw / pitch / roll with smooth jitter), yaw_pitch_roll_to_R rotation builder (OpenCV convention with +pitch=down), and end-to-end `extract_perspective_from_equi`. 13 unit tests.
- Created `prepare_data/video_io.py`: PNG sequence + MP4 loaders (cv2 / decord backends).
- Created `prepare_data/encode_latents.py`: Wan2.1 VAE wrapper for batch encoding perspective videos to 16-channel latents.
- Created `prepare_data/encode_text.py`: UMT5-XXL text encoder + HuggingfaceTokenizer wrapper for prompt → [L, 4096] embeddings.
- Created `prepare_data/compute_masks.py`: agent mask via dynamic-vs-static pano diff + dilation, plus mask projection to perspective view.
- Created `prepare_data/run_prep.py`: orchestrator producing source/target/mask latents + camera tensors per sample. Handles symmetric Pano_00↔Pano_01 pairing, temporal random offset, optional same-orientation flag, chunked GPU equi2pers.
- All 36 existing tests still pass.

**Deferred for now:**
- `lift_and_render.py` (rendered_latent computation): needs depth-warp / point-cloud rasterizer (PyTorch3D or custom z-buffer) — will write next.
- `extract_blobs.py`: depends on user's pre-generated 360 blob videos (RGB equirect format).
- Updating `dataset.py` to gracefully handle missing fields (rendered/blob/text) so we can train with placeholders while the deferred pieces land.

## 2026-04-28 — Dataset + Training loop (Steps 4-5)

## 2026-04-28 — Dataset + Training loop (Steps 4-5)
- Created `dataset.py`: ViewTransferDataset loads pre-computed latents + cameras,
  computes plücker on-the-fly.
- Created `train.py`: Accelerate-based training loop with flow-matching loss,
  CFG dropout (p=0.1 per stream), LoRA/full-FT toggle, gradient checkpointing,
  BF16, cosine LR with warmup, tensorboard logging.
- Created `tests/test_train.py`: CFG dropout, training step, backward, dataset tests.
- All 14 tests passing.

## 2026-04-28 — Initial scaffold (Steps 1-3)
- Created `model.py`: ViewTransferDiT, ViewTransferSelfAttention (KQ-bias),
  ViewTransferDiTBlock (per-block plucker encoder), LoRALinear, apply_lora,
  weight-loading from Wan2.1-T2V-14B.
- Created `plucker.py`: re-exports ray_condition, adds compute_plucker_at_latent_timestamps.
- Created `mask_utils.py`: pack_mask (video-res binary mask → 4ch latent-aligned tensor).
- Created `tests/test_model.py`: shape, gradient, LoRA, mask packing, KQ-bias tests.
- Created `TODO.md` with full step-by-step plan.
