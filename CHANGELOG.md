# Changelog

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
