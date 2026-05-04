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

## Active — Architecture v2 migration (see PLAN.md §10)

**Trigger**: v1 (joint sequence-concat self-attn) produced blurry / structureless preds at step 1600 even on training data and at `guidance_scale=1.0`. Root cause: joint attn dilutes target self-attn, applies target timestep modulation to clean source tokens, RoPE temporal extrapolation. Fix: VACE-style adapter for rendered+blob+mask + zero-gated source cross-attn + joint-drop CFG fix. **All design decisions locked by user; awaiting approval before code edits.**

### v2 implementation steps
- [x] **v2.0** — User approval of PLAN.md §10 (all 4 open decisions answered 2026-05-04)
- [x] **v2.1** — Refactor `model.py`
  - [x] Delete `patch_embed_target` / joint sequence-concat / RoPE temporal offset / head slicing
  - [x] Add `AdapterDiTBlock` with `before_proj` (block 0) + `after_proj` (every block), zero-init
  - [x] Add `geoada_patch_embedding` Conv3d(36→dim), `geoada_blocks` ModuleList, `geoada_layers=[0,4,8,...,36]` (N=10)
  - [x] Add `ViewTransferCrossAttention` (q/k/v/o + RMSNorm; o zero-init); insert at `cross_attn_src_layers=[2,6,10,...,38]` (M=10) inside selected main blocks
  - [x] Add `plucker_encoder_src` (zero-init) per cross-attn-equipped block
  - [x] Rewrite `ViewTransferDiT.forward` (target-only RoPE, sequential `forward_geoada`, hint-add at adapter sites, cross-attn at interleaved sites)
  - [x] Update `from_pretrained`: copy main DiT block weights into adapter blocks; copy `self_attn.{q,k,v}` into `cross_attn_src.{q,k,v}` base; zero-init the seven listed inits
  - [x] Add `assert_step0_invariant()` method
- [x] **v2.2** — Update `train.py`
  - [x] Extend `_TRAINABLE_KEY_PREFIXES` (lora, plucker_encoder, patch_embed_source, geoada_, cross_attn_src)
  - [x] Updated `freeze_base()` (now lives on the model class) with v2 trainable substrings
  - [x] Add joint-drop branch in `apply_cfg_dropout` (per-stream 5% + joint 10%, exposed via `--cfg_drop_prob` and new `--cfg_joint_drop_prob`)
  - [x] Call `assert_step0_invariant()` after `accelerator.prepare()`, before training loop (skippable via `--skip_step0_invariant_check`)
- [x] **v2.3** — `pipeline.py` / `gpu_preprocess.py` / `infer.py`: confirmed no edits needed (model `forward(**cond)` interface unchanged)
- [~] **v2.4** — Tests
  - [x] Marked v1-specific `test_model.py` tests with `@pytest.mark.skip`
  - [x] Rewrote `test_train.py` `test_cfg_dropout_*` against the new `(per_stream_prob, joint_prob)` API (4 tests now)
  - [x] **Empirical step-0 equivalence verified** (CPU smoke): forward-with-conds vs forward-with-zeroed-conds max diff = 0.00e+00
  - [ ] (deferred) Dedicated `tests/test_model_v2.py` once GPU node is available — current CPU + FA-disabled run already proves the architecture-side invariant; the leftover test would just formalise it as a checked-in test
- [x] **v2.5** — Constructed `train_locations_v2.txt` (73) + `val_locations.txt` (8, deterministic split, seed=20260504) under `/work/nvme/beab/rluo2/viewpoint-transfer/data/`
- [x] **v2.5b** — Self-contained project layout: `scripts/_common.sh` + refactored `train.sh`/`infer.sh`/`prep_data.sh`/`train_multinode.sh` + `conftest.py`. All scripts/tests/python entry points runnable from inside `view_transfer_via_query/`. `OUTPUT_DIR` defaults moved under `${PROJECT_ROOT}/runs/`.
- [ ] **v2.6** — Mini smoke training: 4 locs × 200 steps, single GPU, `--log_video_every 50`. Loss must trend down from pretrained-Wan baseline within 100 steps.
- [ ] **v2.7** — Profile peak GPU memory; decide ZeRO-2 vs ZeRO-3 (default: start ZeRO-2 per user-locked decision, switch only if mem-bound)
- [ ] **v2.8** — Full training (multi-GPU, fresh from base Wan, 10–20k steps minimum, save every 500). Use `LOCATIONS_FILE=.../train_locations_v2.txt`.
- [ ] **v2.9** — Inference verification
  - [ ] Eval on both `train_locations_v2.txt` and `val_locations.txt`
  - [ ] CFG sweep `guidance_scale ∈ {1.0, 1.5, 2.5}` after ≥5k steps
- [ ] **v2.10** — Rename `runs/14B-debut/` → `runs/14B-debut-v1-deprecated/`

## Backlog
- [ ] Encode real T5 prompt (replace placeholder `text_emb.pt`) before full training run

## Future
- [ ] Stage 1 (panorama prediction model)
- [ ] EXR depth → fp16 packed tensor for faster I/O
