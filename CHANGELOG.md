# Changelog

## 2026-05-18

- **`scripts/train.sh`: `USE_MESH` default flipped to `1` (was empty / point-cloud).**
  Audit finding: training defaulted to point-cloud lift-and-render (`USE_MESH:-` empty → `--use_mesh` not passed → `args.use_mesh=False`) while inference defaults to mesh (`infer.py` `BooleanOptionalAction default=True`, `scripts/infer.sh` `USE_MESH:-1`). The conditioning fed to the model — `rendered_latent` and the visibility `mask_packed` — therefore came from a different renderer at train vs. inference: point-cloud scatter is sparse/holey with speckled visibility, mesh is a filled antialiased surface with dense visibility. That is a train/inference distribution shift in a conditioning input. Fix: `USE_MESH="${USE_MESH:-1}"` so the existing conditional passthrough (`[ -n "${USE_MESH}" ] && echo "--use_mesh"`) now passes `--use_mesh` by default, matching `infer.sh`. Override with `USE_MESH=` (empty) for the old point-cloud behaviour. `train.py` is unchanged (`args.use_mesh` still respected); no Python call sites or signatures touched. nvdiffrast is now required for default training runs.
- **`pipeline.py` / `infer.py`: inspection videos now come from the pipeline itself; `render_warp_visibility` deleted.**
  Second audit finding: `render_warp_visibility` (which produced `rendered.mp4` / `visibility.mp4` next to `pred.mp4`) was a *parallel reimplementation* of the `gpu_preprocess` render path — it independently re-derived `yaw_pitch_roll_to_R` → `compose_perspective_c2w` → `fov_to_intrinsics` → `lift_and_render`, and called `lift_and_render` with no `use_mesh` so it always used point-cloud scatter even though `pipe.generate` → `gpu_preprocess` conditioned the model on mesh renders. The debug videos thus showed a different signal than the model received, and any future drift in those helpers (dtype, intrinsics, lift mode) would silently re-desync them. Fix (single source of truth, chosen over a same-knobs re-render): `gpu_preprocess` already produces `videos_rendered` / `mask_vis` under `return_videos=True`. `ViewTransferPipeline.generate` gains an opt-in `return_cond_videos: bool = False`; when set it passes `return_videos=True`, moves those two uint8 streams to CPU right after preprocess (so they don't pin GPU memory through the denoise loop — matters under `--low_vram`), strips all non-model keys, and returns `(videos_or_latent, {"rendered","mask_vis"})`. `render_and_save` consumes that tuple and writes the artifacts directly; `render_warp_visibility` and its now-unused imports (`lift_and_render`, `compose_perspective_c2w`, `fov_to_intrinsics`) are removed. Existing `generate` callers (smoke/test/pipeline-doc) are unaffected — default `return_cond_videos=False` keeps the original single-tensor return. Net: the saved inspection videos are now byte-identical to the pre-VAE conditioning the model consumed, and the second lift-and-render per sample is eliminated. Supersedes the earlier same-session `render_warp_visibility` kwarg patch.

## 2026-05-11 (audit pass 2 — prefetcher safety + real overlap, finding #2)

- **`CUDAStreamPrefetcher`: `record_stream` the consumed batch (audit finding #2 — silent-corruption fix).**
  Batch tensors are allocated on the side/preprocess stream but read on the consumer (default) stream. PyTorch's caching allocator only tracks a block's *allocation* stream, so when the consumer dropped its refs the allocator deferred reuse only until the *side* stream was clear — with no record that the consumer stream had touched the memory. The next `_try_fill()` could then hand that block to a new `gpu_preprocess` while the model's fwd/bwd was still reading it → intermittent, input-dependent silent corruption / NaNs (worse at `prefetch_depth=2` than `=1`; the existing `wait_stream` only orders producer→consumer, not this consumer→producer reuse). Fix: new `_record_consumer_stream(batch)` calls `tensor.record_stream(torch.cuda.current_stream(device))` on every CUDA tensor in the popped batch, before `_try_fill()`, so the allocator also waits for the consumer stream before reusing those blocks (standard NVIDIA data-prefetcher pattern). CPU path (`stream is None`) and non-tensor / CPU-tensor batch entries are skipped — no behavior change there.

- **`train.py`: `--sync_timing` flag (default OFF) so `prefetch_depth>1` actually overlaps.**
  The bottleneck-discovery loop wrapped the prefetcher wait and the fwd/bwd/optim block in three full `torch.cuda.synchronize()` calls. Those fully serialized the side-stream preprocess and the consumer every step — accurate for `t_wait`/`t_compute` attribution, but it meant `prefetch_depth=2` delivered **zero** overlap (you paid ~2 slots of VRAM for no throughput gain) and also masked finding #2. The three syncs are now gated behind `--sync_timing` (default **False**). Default run: GPU preprocess on the side stream overlaps fwd/bwd, so `prefetch_depth` is effective; `t_wait`/`t_compute` become approximate CPU-enqueue times (near-zero `t_wait` ⇒ prefetcher hiding data prep) while `t_iter` stays the reliable throughput metric (`loss.item()` in the tqdm postfix forces one sync/step, bounding CPU run-ahead to ~1 iteration). `--sync_timing` restores the exact, serializing measurement for profiling. Safe to remove the syncs *only because* the `record_stream` fix above makes the prefetcher correct by construction. Instrumentation comment block + `--sync_timing` help text document both modes.

- **Tests:** `tests/test_prefetcher.py` — CUDA-gated `test_record_stream_called_on_cuda_tensors` (spies `Tensor.record_stream`; asserts exactly one call per batch on the *current* stream for the CUDA tensor, none for CPU/non-tensor entries, values still correct, `depth=2`) and `test_cpu_heterogeneous_batch_passthrough` (mixed tensor/str/int/list batch passes through unchanged on the CPU path). Full suite: 61 passed, 2 skipped (pre-existing, unrelated) — the CUDA spy test ran and passed on-device here.

## 2026-05-17

- **Two-phase guidance sweep infrastructure.**
  - **`infer.py --seed` (correctness blocker fix).** The per-location RNG was
    `np.random.default_rng()` (unseeded) → every job drew different `t0` +
    src/tgt trajectories, so configs in a sweep were scored on *different*
    inputs. New `make_location_rng(seed, loc)` keys the stream on
    `(seed, md5(loc))` → identical samples per location across all runs,
    independent of order/sharding. Recorded in `info.json`. `infer.sh`: `SEED`
    env passthrough. Tests: `tests/test_infer_seed.py` (4 cases — reproducible,
    seed-sensitive, location-sensitive, None=nondeterministic), all pass.
  - **`scripts/guidance_sweep_manager.py` → spec-driven.** Now takes
    `--sweep-spec <json>` ({phase_tag, locations_file, seed, n_expected,
    configs}); per-task `OUT_DIR`/`SEED`/`LOCATIONS_FILE` exported into the
    wrapper; outputs under `infer_out/<run>/<phase_tag>/geomG_srcS_textT/` so
    phases never collide; state/logs under `_sweep_logs/<phase_tag>/`.
  - **`scripts/make_sweep_spec.py`.** phase1 = geom{2..8}×src{1,1.5,2,3,4}
    (text=3) + diagonal (k,k,k) k∈{1..5}, deduped → **39 configs**, 4-loc
    subset (`sweep_loc4.txt`, indices 0/4/7/11 of the 12-loc test split),
    seed 0. phase2 = explicit finalists + text ablation on full 12 locs.
  - **`scripts/score_sweep.py`.** Pairs pred.mp4↔tgt_gt.mp4 per sample,
    torchmetrics PSNR/SSIM/LPIPS (frame-mean→sample-mean), writes
    `leaderboard.{csv,json}` sorted by LPIPS. LPIPS/AlexNet weights pre-cached
    on the login node (compute nodes lack internet).
  - **`--max-concurrent` (default 50)** added to the manager: caps QUEUED+RUNNING
    sbatch jobs (counted each cycle, submission pass breaks at the cap).
  - **Execution:** probe (3/3/3, 4-loc, seed 0) validated end-to-end — all
    outputs + grid.mp4 present, `info.json` seed=0/t_offset=56 deterministic,
    50-step denoise clean. Confirmed timing: **~49.5 s/step → ~41.5 min
    denoise/sample, ~3.2 h per 4-loc job** (→ phase-1 ≈ 125 GPU-h at 50 steps).
    Probe cancelled after validation; its 1 done sample reused by the manager's
    3/3/3 job via infer.sh resume-skip. **Phase-1 launched** (manager pid-nohup,
    `--sweep-spec sweep_spec_phase1.json --max-concurrent 50 --max-retries 5`):
    39 jobs submitted to `ma`. Next: leaderboard → finalists → phase-2 (12 locs).

- **SLURM job manager for the guidance sweep (`scripts/guidance_sweep_manager.py`).**
  The interactive 4-way launch OOM-killed 2 jobs: the shared SLURM allocation
  (job_380825) has a 256 GiB memory cgroup, and four concurrent `--low_vram`
  loads (each ~60 GiB transient while the 14B state-dict loads) crossed it
  (`memory.events oom_kill 2`). Fix: one independent `sbatch` job per config
  (own mem allocation). Slim adaptation of UE-Render's `slurm_batch_manager.py`:
  per-attempt wrapper + `timeout … ; touch success.marker`; `squeue`→`sacct`
  two-step state; success = `COMPLETED` + marker; resubmit any other terminal
  state up to `--max-retries 5` (resume-safe — `infer.sh` skips done samples);
  tiered partition escalation `ma → snavely → gpu` (priority nodes first, then
  general; escalates on infra failure or >600 s queue stall); `state.json`
  read back on startup (restart-safe). Submitted with
  `--gres=gpu:nvidia_rtx_a6000:1 --account=ma --mem=128G --cpus-per-task=8
  --time=06:00:00`. Interactive survivors killed; all 4 resubmitted clean.

- **Grouped-guidance sweep tooling + launch.** `scripts/infer.sh`: new
  `GUIDANCE_GEOM/GUIDANCE_SRC/GUIDANCE_TEXT` env passthrough (all-or-none;
  errors on partial), grouped `OUT_DIR` tag `…_geom<g>_src<s>_text<t>`,
  monolithic path unchanged when unset. New `scripts/infer_guidance_sweep_4gpu.sh`:
  4 chained-guidance configs, one per GPU, `LOW_VRAM=1`, backgrounded with
  per-job logs under `infer_out/14B_4gpu_640P_0507/_sweep_logs/`.
  - Run: checkpoint-10000 on 0507 test split (12 locs, all complete), src00→tgt01.
    Configs (geom/src/text): GPU0 3/3/3 (≡ monolithic g=3 control via telescoping
    identity), GPU1 5/2/3, GPU2 7/1.5/2, GPU3 5/3/3. Outputs →
    `infer_out/14B_4gpu_640P_0507/ckpt10000_diff_geom<…>_src<…>_text<…>/`.

- **Grouped (chained) classifier-free guidance at inference.** The model is trained
  with independent per-stream CFG dropout (`train.apply_cfg_dropout`), but inference
  previously collapsed everything into one scalar `guidance_scale` with an all-zero
  uncond — discarding the trained ability to weight conditioning groups separately.
  Added a 3-level nested decomposition to `ViewTransferPipeline.generate`:
  `v = v_uncond + w_geom·(v_geom−v_uncond) + w_src·(v_geomsrc−v_geom) + w_text·(v_full−v_geomsrc)`
  where geom = {plucker_tgt, rendered_latent, mask_packed, blob_latent},
  src = {source_latent, plucker_src}, text = {text_emb}.
  - New `generate` kwargs `guidance_geom/guidance_src/guidance_text`; pass all three
    to enable grouped mode, leave all `None` for the original monolithic CFG path
    (unchanged default behaviour). Partial specification raises `ValueError`.
  - Cost: 4 model forwards/step (uncond, geom, geom+src, full) vs 2 for monolithic.
  - New helper `ViewTransferPipeline._subset_cond` (zeros inactive streams, preserves
    shape/dtype). `infer.py`: `--guidance_geom/--guidance_src/--guidance_text` CLI
    flags, threaded through `render_and_save`, recorded in `info.json`.
  - Tests: 4 new cases in `tests/test_pipeline.py` (subset masking, all-three
    requirement, grouped run shape, grouped ≠ monolithic). Also un-rotted `_StubVAE`
    (`.model`/`.parameters()` — `gpu_preprocess`/decode now read VAE dtype there);
    full file 10/10 green.

## 2026-05-11 (audit pass 2 — frame ordering, finding #5)

- **`video_io`: natural frame ordering + empty-window guard (audit finding #5).**
  `list_png_frames` used `str.sort()`, which orders `frame_10.png` before `frame_2.png` — silently shuffling any frame sequence whose filenames aren't fixed-width zero-padded. Added `natural_sort_key` (splits the basename into digit/non-digit runs, compares digit runs as ints) and switched `list_png_frames` to it. It is a strict no-op for the zero-padded UE renders actually on disk (lexical order == numeric order), so existing data/checkpoints are unaffected; it only *corrects* the order for non-padded names.
  - **RGB↔depth consistency:** depth frames are indexed `depth_files[t0]` from `_list_files`, and must stay in the same order as `list_png_frames`' RGB output (frame t0 of RGB ↔ frame t0 of depth). Fixing only the RGB sort would *desync* RGB vs depth for non-padded data, so `_list_files` in `dataset.py` and `infer.py`, plus the inline depth listing in `infer.py`'s diff-camera overlap path, were switched to the same `natural_sort_key`. Net effect on the real (zero-padded) data: none.
  - **Empty-window guard:** `load_png_sequence` with `num_frames=0` (or an auto-computed non-positive count when `start >= len(paths)`) hit `max(indices)` on an empty list → opaque `ValueError: max() arg is an empty sequence`. Now raises a descriptive `ValueError("Empty frame window ...")` before the index math; the out-of-range case still raises `IndexError` as before. (`load_mp4` left as-is — not on the training/inference path.)
  - **Tests:** new `tests/test_video_io.py` — natural key orders non-padded names numerically and is a no-op for zero-padded; `list_png_frames` / `load_png_sequence` return frames in natural order (pixel-fingerprint check); empty window → `ValueError`; out-of-range → `IndexError`.

## 2026-05-11 (audit pass 2 — mask causal alignment)

- **`mask_utils.pack_mask`: align the visibility mask to the Wan2.1 VAE's *causal* temporal compression (audit finding #3).**
  The Wan2.1 VAE temporal map is causal (`WanVideoVAE.encode`: `iter_ = 1 + (T-1)//tf`): latent 0 ← raw frame {0} (a single frame), latent j≥1 ← raw frames {tf·(j-1)+1 … tf·j}. `pack_mask` previously grouped uniformly from frame 0 (`latent j ← frames {tf·j … tf·j+tf-1}`) and padded the **tail**, so the packed mask was shifted ~(tf-1)≈3 frames *later* than the rendered / source / blob latents it is channel-concatenated with in `model.forward` (`torch.cat([rendered_latent, blob_latent, mask_packed], dim=1)`). Worst at latent 0: content = 1 real frame (frame 0), mask = OR of frames {0,1,2,3}.
  - Fix: prepend `(tf-1)` copies of frame 0 before the stride-`tf` fold (the canonical Wan-I2V mask construction). This collapses the first fold group to `{f0,f0,f0,f0}` (→ latent 0) and aligns every subsequent group of `tf` raw frames to one latent (→ latent j). Tail pad/trim retained, but now only fires for non-canonical `T` (Wan uses `T = tf·k + 1`, e.g. 81, where the prepended length is exactly `T_lat·tf` and no tail work happens).
  - `T_lat` rewritten as `1 + (T-1)//tf` to read in the causal form; mathematically identical to the old `(T+tf-1)//tf` (= ceil(T/tf)), so output **shape is unchanged** and the channel-concat in `model.forward` is unaffected.
  - **Behavioral change:** the mask conditioning stream now carries different (correctly-registered) values per latent frame. Checkpoints trained on the old (misaligned) packing will see a shifted mask at inference unless retrained/fine-tuned with the corrected packing — intended, this is the bug fix.
  - **Tests:** new `tests/test_mask_utils.py` — first-frame-only mask fills exactly latent 0 (all `tf` channels) and nothing else (canonical Wan-I2V property; the old fold gave `[1,0,0,0]` and fails this); a parametrized single-frame test asserting every raw frame lands in the latent/channel the VAE causal map assigns it to; and a regression guard that `frame == tf` is in latent 1, never latent 0. Existing `test_model.test_pack_mask` (shape + value range) still passes (`T_lat` unchanged).

## 2026-05-11 (later)

- **`encode_video_to_latent(..., keep_on_device=True)`: force `.to(device)` on the way out.**
  Crash: `RuntimeError: Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type) should be the same` on the first DiT forward in low-VRAM inference. Root cause: `gpu_preprocess` passes `tiled_vae=self.low_vram` → `WanVideoVAE.tiled_encode` (`wan_video_vae.py:1155`), which hardcodes `data_device = "cpu"` for the accumulator and silently returns CPU latents even though the VAE itself lives on GPU. The Fix-4 audit assumed `vae.encode` returned on `device` whenever the VAE was — true for `single_encode`, false for `tiled_encode`. Fix is in `encode_video_to_latent`: explicit `out.to(device)` when `keep_on_device=True`. No-op in the non-tiled path (training + non-low-VRAM inference) since `single_encode` already returns on `device`; corrects the tiled path used by low-VRAM inference. Decode is unaffected — `pipeline.generate` stages latents on CPU before `vae.decode` by design.

- **`tools/viz_mesh_render.py`: switch to overlap-constrained `sample_trajectory_pair`.**
  Previously called `sample_perspective_trajectory` twice (independent draws), so diff-camera renders rarely shared content with the source view. Now mirrors the `viz_trajectories.py` update from earlier today: `sample_trajectory_pair(..., pano_c2w_src_at_t0, pano_c2w_tgt_at_t0, depth_equirect, min_overlap)` for both same- and diff-camera. For diff-camera draws, the t0 static depth EXR is loaded so the matching direction is depth-based (with parallax correction), not heuristic.
- **`tools/viz_mesh_render.py`: `--locations_file` arg (mutually exclusive with `--data_root`) + `--min_overlap`.** Reuses `_load_locations_from_file` from `viz_trajectories.py`. Default `--min_overlap=0.25` matches the train recipe.
- **`scripts/viz_mesh_render_4gpu.sh`: `LOCATIONS_FILE` (default = `view_transfer_0507/train.txt`), `MIN_OVERLAP` env knobs.** If `DATA_ROOT` is set, it takes precedence (back-compat); otherwise the launcher uses `--locations_file`.
- **Run:** `NUM_SAME=50 NUM_DIFF=50` across 4× A6000 — 100 mp4s in `viz_mesh_render/` (~9.5 min wall, 22-25 s/sample). All shards exited 0.

## 2026-05-11

Audit pass — seven fixes spanning train / infer / pipeline / preprocess. None change model semantics; defaults preserve prior behaviour.

- **Fix 1 — `train.py`: `args.json` write now mkdir's and runs on rank 0 only.**
  The write was previously the first thing `main()` did, before `Accelerator()` and any `os.makedirs(args.output_dir)`. Direct `accelerate launch view_transfer_via_query/train.py …` (i.e. anything that didn't go through `scripts/train.sh`'s `mkdir -p`) crashed with `FileNotFoundError` on a fresh `output_dir`, and multi-rank runs raced to write the same file. The write is now gated on `accelerator.is_main_process` and preceded by `os.makedirs(args.output_dir, exist_ok=True)`.

- **Fix 2 — `save_trainable(model, save_path, full: bool = False)`; `infer.py` auto-detects LoRA vs full.**
  The previous filter (`_TRAINABLE_KEY_PREFIXES`) silently dropped main-DiT params during full-finetune (`--lora_rank=0`), so `trainable_params.pt` only contained adapter / plücker / patch-embed-source / cross-attn-src keys; the main blocks' tuned weights existed in the accelerator state but never made it into an inference-loadable file. `save_trainable` now takes `full=True` (passed from the call site when `args.lora_rank == 0`) which persists the full `state_dict`. `infer.py` peeks at the loaded checkpoint and only runs `apply_lora()` when `lora_` keys are present, so the same `--lora_ckpt /path/to/trainable_params.pt` flag works for both training modes without a new CLI option.

- **Fix 4 — `encode_video_to_latent(..., keep_on_device=True)`; online preprocess skips the D2H+H2D round trip.**
  `gpu_preprocess._encode_batch` used to receive each latent on CPU (`encode_video_to_latent` finished with `.cpu()`) and then re-stage it on GPU via `.to(device)` for every stream every step. The new kwarg returns the latent on `device` directly; offline encoders keep the default (`False`) so on-disk shards still get CPU writes. Note: `WanVideoVAE.encode` still copies its *input* to CPU internally — that bounce is upstream and out of scope.

- **Fix 5 — `pipeline.py`: fuse the decode-time device+dtype cast on the CPU side.**
  `[z[b].to(vae_dtype) for b in range(B)]` promoted bf16→fp32 on GPU before `vae.decode` immediately moved the result to CPU (twice the PCIe bytes). Now `z[b].detach().to(device="cpu", dtype=vae_dtype)` copies bf16 across PCIe and upcasts on CPU. Behaviour is identical in the low-VRAM path (vae_dtype = bf16).

- **Fix 6 — `infer.py`: `render_and_save` returns `True`/`False`; summary distinguishes generated vs skipped-existing.**
  Previously, `pred.mp4 exists` early-returns still incremented `n_done`, so the final progress line over-counted on resumed runs. New summary: `generated=… skipped_existing=… skipped_locations=…`.

- **Fix 7 — `infer.py`: one RNG per location; optional `_k{k}` folder suffix.**
  `rng = np.random.default_rng()` was inside `for k in range(num_per_location)`, so each k drew an independently-seeded random `t0` and two draws that landed on the same t0 hit the same folder name → second one silently skipped. RNG is now hoisted out of the k-loop, and when `num_per_location > 1` the folder name gains a `_k{k}` suffix so distinct draws stay distinct. Single-sample runs keep their original folder names (resume paths unchanged).

- **Fix 8 — `dataset.py`: removed dead `.permute(0, 1, 2)` on `static_rgb_t0`.**
  Identity permute, no behavioural effect.

- **Tests:** `test_train.py` gains two `save_trainable` regression tests — one verifying LoRA mode (default) only writes `lora_*` / `plucker_encoder` / `patch_embed_source` / `geoada_*` / `cross_attn_src` keys, one verifying `full=True` writes every key in `model.state_dict()`. All 27 (+2 skipped) tests in `test_train / test_dataset / test_model / test_prefetcher` pass after the changes. `test_pipeline.py` has a pre-existing failure unrelated to this audit: `_StubVAE` lacks the `.model` attribute that `gpu_preprocess.py:146` (`vae.model.parameters()`) has required since the 2026-05-04 initial commit.

## 2026-05-10
- **Opt-in low-VRAM inference path** (`--low_vram` / `LOW_VRAM=1`). Default-off; when unset, code paths and behavior match the pre-2026-05-10 original byte-for-byte. Training (`train.py`) is unaffected.
  - `infer.py`: new `--low_vram` CLI flag. When set, `build_pipeline` keeps DiT on CPU (cast to `dtype`), loads VAE in `dtype` instead of fp32, and wraps both with `diffsynth.core.vram.layers.enable_vram_management` — Linear/Conv/Norm via `AutoWrappedLinear`/`AutoWrappedModule`, full `ViewTransferDiTBlock` / `AdapterDiTBlock` via `AutoWrappedNonRecurseModule`. `vram_limit = free_vram - 4 GB` caps GPU residency; overflow layers demand-page per forward. When unset, restores the original eager-residency path: `model.to(device, dtype)` and `load_wan_vae(..., dtype=torch.float32)`.
  - `pipeline.py`: `ViewTransferPipeline.__init__` takes `low_vram: bool = False`. In `generate()`, the encode/denoise/decode-boundary `_vram_advance_modules(...)` calls (mirroring `BasePipeline.load_models_to_device` since we don't subclass it) and `tiled=` on `vae.decode` are all gated behind `self.low_vram`. Decode latents go through `to(vae_dtype)` — equivalent to `.float()` when VAE is fp32, so semantics are unchanged in the default path.
  - `gpu_preprocess.py`: new keyword `tiled_vae: bool = False`. Threads through to `encode_video_to_latent(..., tiled=tiled_vae)`. `pipeline.py` passes `tiled_vae=self.low_vram`; `train.py` doesn't pass it, so training keeps untiled encode.
  - `scripts/infer.sh`: new `LOW_VRAM` env var. Non-empty → appends `--low_vram` to the python invocation; documented in the header comment block.
  - **Why:** at 14B / 81 frames / 368×640, the original eager-residency path OOMs on 48 GB A6000 around `vae.decode` (DiT still pinned + fp32 VAE decode). Verified working under `LOW_VRAM=1` on a 48 GB A6000: peak ≈45.8 GB, 4-step smoke run completed end-to-end. On 80 GB cards there's no need for the overhead — leave the flag unset.

## 2026-05-09
- `scripts/_common.sh`: export `LIBRARY_PATH`, `LD_LIBRARY_PATH`, `CPATH` to include `/opt/xpmem/lib64` and `/opt/xpmem/include` when that directory exists.
  - **Why:** On DeltaAI (gh-nodes / Cray + GH200), the `xpmem` module only populates runtime `LD_LIBRARY_PATH`. DeepSpeed JIT-compiles ops that pull in cray-mpich → `-lxpmem`, and the linker fails with `cannot find -lxpmem` because compile-time `LIBRARY_PATH` is empty.
  - **Scope:** guarded by `[ -d /opt/xpmem/lib64 ]`, so it's a no-op on non-Cray machines.
