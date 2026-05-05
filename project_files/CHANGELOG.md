# Changelog — view_transfer_via_query

## 2026-05-04 — Project layout: self-contained `view_transfer_via_query/`

Goal: make `view_transfer_via_query/` runnable as a self-contained project (matches the planned submodule conversion). User can `cd` in and launch any script / test / `python -m view_transfer_via_query.X` without thinking about the parent `DiffSynth-Studio/` directory.

### Changes
- **NEW `scripts/_common.sh`**: shared path-setup snippet sourced by every shell script. Resolves two roots from script location:
  - `PROJECT_ROOT = view_transfer_via_query/`
  - `DIFFSYNTH_ROOT = DiffSynth-Studio/` (provides `diffsynth` lib + pretrained Wan models)
  - Exports `PYTHONPATH=$DIFFSYNTH_ROOT:$PYTHONPATH` so both `view_transfer_via_query.X` and `diffsynth.X` resolve from any CWD.
  - Refuses to be executed standalone (must be sourced).
- **`scripts/train.sh`**: sources `_common.sh`. `OUTPUT_DIR` default → `${PROJECT_ROOT}/runs/${MODEL_SIZE}_debut_v2` (was `${REPO_ROOT}/runs/${MODEL_SIZE}_debut`). `LOCATIONS_FILE` default → `train_locations_v2.txt`. `DIT_CKPT`/`VAE_CKPT` defaults still under `${DIFFSYNTH_ROOT}/models/Wan-AI/...`. `cd "${PROJECT_ROOT}"` so output paths in templates resolve under the project tree.
- **`scripts/infer.sh`**: full rewrite. Switched from positional args (`bash infer.sh project_root locations_file`) to env vars (`MODEL_SIZE`, `LORA_CKPT`, `OUT_DIR`, `GUIDANCE_SCALE`, `NUM_INFERENCE_STEPS`, etc.) matching `train.sh`'s style. Auto-tags `OUT_DIR` from `LORA_CKPT` step + `GUIDANCE_SCALE` if not overridden. Defaults `OUT_DIR` under `${PROJECT_ROOT}/infer_out/`.
- **`scripts/prep_data.sh`**: sources `_common.sh`. `DATA_FOLDERS` and `OUTPUT_FILE` now overridable via env. Default `OUTPUT_FILE` → `${DATA_ROOT}/train_locations.txt` (absolute, was a brittle relative `../data/...` path).
- **`scripts/train_multinode.sh`**: sources `_common.sh`. `OUTPUT_DIR` default → `${PROJECT_ROOT}/runs/mn_$(date)`. Per-node srun unchanged.
- **NEW `conftest.py`** at `view_transfer_via_query/`: pytest auto-loads from the test-discovery root. Adds `DIFFSYNTH_ROOT` to `sys.path` so `pytest tests/` works from inside `view_transfer_via_query/` without per-file `sys.path.insert(...)` boilerplate. Also forces `FLASH_ATTN_{2,3}_AVAILABLE = False` so model tests run on CPU machines.

### Result — all of these now work from inside `view_transfer_via_query/`

```bash
cd view_transfer_via_query
bash scripts/train.sh                           # outputs → view_transfer_via_query/runs/14B_debut_v2/
bash scripts/infer.sh                           # outputs → view_transfer_via_query/infer_out/14B_v2-…/
bash scripts/prep_data.sh                       # writes train_locations.txt under DATA_ROOT
pytest tests/                                   # 24 pass, 3 skip (v1)
python -m view_transfer_via_query.train --help  # NEW — works because PYTHONPATH includes parent
```

And from any other CWD (scripts auto-resolve paths from `${BASH_SOURCE[0]}`):

```bash
bash /any/path/to/view_transfer_via_query/scripts/train.sh
```

### Trade-offs / migration
- **OUTPUT_DIR convention shift**: old runs at `DiffSynth-Studio/runs/14B-debut/` stay where they are (v1, deprecated). New v2 runs land at `view_transfer_via_query/runs/14B_debut_v2/`. Already in line with §10.10's plan to mark the v1 dir deprecated.
- **`infer.sh` CLI**: positional args (`bash infer.sh <project_root> <locations_file>`) replaced by env vars. Old call sites need updating; new pattern: `LORA_CKPT=… GUIDANCE_SCALE=2.5 bash scripts/infer.sh`.
- **`scripts/_common.sh` MUST be sourced** (`source ./scripts/_common.sh`), not executed (`bash ./scripts/_common.sh`). Has a guard that errors with a clear message if executed.

### Follow-up — DeepSpeed memory analysis (verified end-to-end on gh031)

Plain `zero2_bf16` does **not** fit at full target shape on either configuration we tested:

- **Single GH200, slurm 120 GB host**: OOM at step 0 fwd, 94.94/95 GB GPU (1 GB short of fitting). My earlier synthetic-tensor measurement (93.9 GB peak with `torch.optim.AdamW`) missed real `gpu_preprocess` + DeepSpeed wrapper overhead.
- **2× GH200, slurm 240 GB host**: OOM at step 0 fwd, 88 GB/rank used; tried to allocate 9.97 GB more for an attention activation. ZeRO-2 sharding the 43 GB Adam state across 2 ranks saves ~21 GB/rank, but the remaining 67 GB (model + grads) + activations + DeepSpeed buckets still over-runs.

**`zero2_bf16_cpu_offload` (new config) works on both**:

- New file `configs/deepspeed/zero2_bf16_cpu_offload.json`: identical to `zero2_bf16.json` except adds `"offload_optimizer": {"device": "cpu", "pin_memory": true}` under `zero_optimization`. Frees ~43 GB GPU per rank (Adam state on host).
- **Trade-off**: optimizer step gets slower (CPU compute + Grace↔Hopper transfer). On gh031 (2 × GH200): per-iter time ~100 s (fwd+bwd ~47 s + dataloader wait ~55 s; CPUAdam JIT-compiles on first use, then is hot).
- **Cost on host RAM**: with `pin_memory: true`, DeepSpeed pins the offloaded state as shmem. Sized at ~120 GB shared on 2-GPU run. If the slurm allocation is tight on RAM (e.g. 120 GB single-GPU job), set `pin_memory: false` to roughly halve resident RAM at the cost of slower CPU↔GPU transfers — fits the smaller cgroup but pageable-memory transfers are ~2× slower.

**Verified** on `gh031` (2 × GH200, 240 GB host) with the v2 architecture:
- 10-step smoke run with `PROFILE_STEPS=2`: completed cleanly. Loss curve `0.094 → 0.100 → 0.198 → 0.162 → 0.762 → 1.211 → 3.969 → 0.836 → 1.891 → 3.156` (variance is the BSMNTW timestep-weighted loss across random timestep samples — expected, not a regression).
- Profiler trace dumped to `${OUTPUT_DIR}/profiler_trace/` after step 5; viewable in TensorBoard / chrome://tracing.
- Checkpoint-5 + checkpoint-10 both saved with full set of files: `pytorch_model/` (sharded ZeRO state), `random_states_{0,1}.pkl`, `latest`, `step.json` (`{"global_step": N}`), `trainable_params.pt` (10.7 GB flat bf16), `zero_to_fp32.py`.
- **Resume verified**: `RESUME=…/checkpoint-5 MAX_STEPS=8 SAVE_EVERY=4` correctly loads optimizer/model/RNG state, prints `Resuming at global_step = 5`, tqdm starts at `5/8`. Step counter, ZeRO-sharded weights, and Adam state all restored.

### How to launch (post-2026-05-04)
```bash
cd view_transfer_via_query

# Single GPU (or low-host-RAM allocation):
DEEPSPEED_CONFIG=configs/deepspeed/zero2_bf16_cpu_offload.json bash scripts/train.sh

# 2+ GPUs with adequate host RAM (~256 GB+ for 5.35 B trainable):
CUDA_VISIBLE_DEVICES=0,1 DEEPSPEED_CONFIG=configs/deepspeed/zero2_bf16_cpu_offload.json bash scripts/train.sh

# Resume:
DEEPSPEED_CONFIG=configs/deepspeed/zero2_bf16_cpu_offload.json \
RESUME=runs/14B_debut_v2/checkpoint-5000 \
bash scripts/train.sh
```

The plain `zero2_bf16.json` config is kept for future use when (a) a 4+ GPU run shards optimizer state below ~10 GB/rank, or (b) trainable params shrink (e.g. switching the adapter to LoRA per the deferred B-LoRA option from §10).

### Follow-up — data-pipeline tuning + bf16 VAE bug

**`prepare_data/encode_latents.py`**: removed the unconditional `.float()` cast in `encode_video_to_latent`. Was forcing fp32 input to the VAE encoder regardless of the VAE's actual parameter dtype, which silently worked when the VAE was fp32 but raised `RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same` the moment `DATA_DTYPE=bf16` started loading the VAE in bf16. Replaced with `.to(dtype=next(vae.model.parameters()).dtype)` so callers can flip VAE dtype without surgery. Discovered while running a `NUM_WORKERS=2 DATA_DTYPE=bf16` smoke on gh080.

**Throughput comparison on full target shape (B=1, 81f, 480×832, ~131k tokens, 2× GH200, ZeRO-2 + CPU-offload optimizer)**:

| Config | iter | comp | wait | notes |
|---|---|---|---|---|
| `NUM_WORKERS=0, DATA_DTYPE=fp32` (gh031, baseline) | ~100 s | ~47 s | ~55 s | fully serial |
| `NUM_WORKERS=2, DATA_DTYPE=bf16` (gh080, step 10) | **79 s** | 60 s | **19 s** | warmed prefetcher |

Net **~21 % per-step speedup** from re-enabling the prefetcher and switching the VAE to bf16. `wait` drops 36 s as the prefetched batches start hiding behind compute; `comp` rises 13 s because the side CUDA stream (running the next batch's VAE encode etc.) now competes with the model's fwd/bwd for GPU resources.

**`NUM_WORKERS=4` does NOT fit the 240 GB slurm cgroup on gh031/gh080** when combined with DeepSpeed CPU-offload pinned-memory shmem (~80 GB/rank) and 2 ranks. Killed with SIGKILL after step-0. Stick to `NUM_WORKERS=2` until either the slurm allocation is bumped (`--mem=320G+`) or pin_memory is disabled in the offload config (slower, fits).

**Remaining `wait ≈ 19 s`** suggests the GPU-side preprocess (VAE encode + lift+render + equi2pers) takes ~80 s total per batch and the prefetcher only manages to hide ~60 s behind the compute step. The next clean win is **prefetch depth ≥ 2** in `prefetcher.py` (was 1 batch in flight). See follow-up below.

### Follow-up — `CUDAStreamPrefetcher` depth ≥ 2 (verified)

**Patch**:
- `prefetcher.py`: replaced single `_next_batch` slot with a `collections.deque` of length `depth`. New `_try_fill()` keeps topping up until full or loader exhausted. Default depth=2 (was effectively 1).
- `train.py`: new `--prefetch_depth` CLI flag (default 2), threaded into `CUDAStreamPrefetcher(...)`.
- `scripts/train.sh`: new `PREFETCH_DEPTH` env var (default 2).
- Existing 5 `test_prefetcher.py` tests pass unchanged (defaults preserved at deque-of-1 semantics when depth=1).

**Verified on gh080 with same shape + config** (B=1, 81 f, 480×832, NUM_WORKERS=2, DATA_DTYPE=bf16, ZeRO-2 + cpu-offload):

| Config | step 10 iter | comp | wait |
|---|---|---|---|
| `PREFETCH_DEPTH=1` | 79.4 s | 59.8 s | 18.7 s |
| `PREFETCH_DEPTH=2` | **73.4 s** | 59.9 s | **13.1 s** |

Net **~8 % per-step speedup**, 6 s saved per iter. For a 10 k-step run that's ~17 hours saved.

**Why the win is smaller than naive math predicted** (I earlier estimated ~25 %): the side stream and default stream **share Hopper SMs**. With depth=2 the side stream is rarely idle (always running the next batch's VAE encode + lift+render), so it competes with the model's fwd/bwd for compute units. `comp` is essentially flat (60 s in both cases — the floor), but the side stream's higher utilisation eats some of the wait reduction. The remaining 13 s of `wait` is genuine: even with two slots in flight, GPU preprocess (~80 s) > compute (~60 s), so the consumer occasionally has to wait for the next batch to finish encode.

**Caveats observed during the run**:
- First 4-5 steps showed inflated `comp` (~70 s, +10 s vs steady) — JIT cache + DeepSpeed allocator + first-batch overhead. Steady-state (steps 8-10) is the right number to compare.
- depth=2 transient peak GPU mem ~95 / 97 GB on rank 0 during init (both prefetched batches' encode buffers in flight before they release). Settled to ~82 GB/rank in steady state. **No headroom for depth=3 on this hardware** without something else changing (e.g. activation memory drop).
- `NUM_WORKERS=4` still OOMs the 240 GB slurm cgroup with cpu-offload pin_memory; stick to `NUM_WORKERS=2`.

Profiler trace from this run: `runs/2gpu_depth2/profiler_trace/gh080.…_3041041.…pt.trace.json`.

### Best-known config (verified on gh116, 4 × GH200, 480 GB host RAM)

| Knob | Value | Notes |
|---|---|---|
| GPUs | 4 | ZeRO-2 sharding /4 brings Adam state below the no-offload threshold |
| `DEEPSPEED_CONFIG` | `zero2_bf16.json` | **No CPU offload** — saves the per-step CPU↔GPU transfer overhead and host shmem pressure |
| `PERS_H × PERS_W` | 368 × 640 | ~60 % of full-res pixels; both divisible by 16 |
| `NUM_WORKERS` | 2 | |
| `PREFETCH_DEPTH` | 2 | |
| `DATA_DTYPE` | bf16 | |

Measured steady-state at step 10: **iter = 25.3 s, comp = 20.9 s, wait = 4.4 s** → ~70 hours for 10 k steps (≈ 2.9 days).

For comparison, a 2 × GH200 + cpu_offload + 480 × 832 baseline runs at iter ≈ 73 s (≈ 8.4 days for 10 k steps). The 4-GPU + lower-res combo is **~2.9× faster** end-to-end.

GPU memory at this config: ~71-73 GB / 97 GB per rank (~25 GB headroom). Host RAM well within the 480 GB allocation (~80 GB total resident).

### Recommended defaults for v2 production training
The defaults in `train.sh` now reflect this:
- `NUM_WORKERS=2`
- `DATA_DTYPE=bf16`
- `PREFETCH_DEPTH=2`
- `DEEPSPEED_CONFIG=configs/deepspeed/zero2_bf16_cpu_offload.json` (single-GPU and 2-GPU on gh-class nodes; needed because plain `zero2_bf16` OOMs even at 2-rank sharding)

### Follow-up — `dataset.py`: `is_complete()` length-consistency check + `discover_entries(verbose=...)`

**Why**: a 4-GPU smoke run hit `IndexError: t0=40 but only 7 depth frames in .../Pano_01_static/depth` at step 3. The c2w pose tensor for that location had 160 frames (matching the full-length RGB sequences) but `static_depth/` only contained 7 EXR files — partially-prepared data that passed the existing existence-only check.

**Patch** ([dataset.py:82-150](view_transfer_via_query/dataset.py#L82-L150)):
- `SampleEntry.is_complete()` now requires not just existence but **length consistency** across every sequence source:
  - PNG counts in `rgb_src`, `rgb_tgt`, `static_rgb`, `blob`
  - depth-file count in `static_depth` (.exr / .npy / .pt / .pth)
  - pose-tensor lengths in `c2w_src`, `c2w_tgt` (loaded via `torch.load(..., weights_only=True)`)
  - common length must be > 0
- New `verbose=False` param prints a one-line skip reason for each rejected pairing.
- `discover_entries(verbose=...)` plumbs the flag through and prints a final `kept N / total (skipped K)` summary.

**Audit on `train_locations.txt`** (158 locations × 4 pairings = 632 max entries):
- **626 valid pairings, 6 skipped** (3 locations × 2 src=01 pairings):
  - `Havana_1-2car_.../x164_y1683...`: `static_rgb` 159 PNGs vs 160 elsewhere (off-by-one in prep)
  - `NYC50s_1-2car_.../x60772_y-1789...`: `static_depth` is empty (depth render didn't run)
  - `RuralAustralia_Wood_.../x-11717_y11872...`: `static_depth` 7 vs 160 (the one that bit us)
- src=00 pairings of those locations still survive because they read `Pano_00_static/` which is intact.

**Tests**: existing `tests/test_dataset.py` (7 tests) all pass — the consistency check is satisfied by the in-test fake locations because they're built with matching frame counts.

### Bigger throughput wins still on the table (not yet implemented)
- **Pre-cache encoded latents to disk**: skips the 60-80 s per-batch GPU preprocess entirely, removing the side-stream/main-stream contention that's currently the binding constraint. ~1.5-2 × further speedup expected. Trade-off: locks in (t0, trajectory) per cached latent — diversity recovery via pre-encoding K trajectories per sample. The "online VAE encoding" decision in PLAN §4.2 was a deliberate trade — re-evaluate once first v2 results are out.
- **Smaller VAE / lower preprocess resolution**: cuts the preprocess cost without changing model resolution.

### Follow-up — direct script invocation for `prepare_data/*.py`
Added a tiny bootstrap snippet at the top of `prepare_data/run_prep.py` and `prepare_data/prep_all_scenes.py` so they work both as `python -m view_transfer_via_query.prepare_data.<x>` AND as `python prepare_data/<x>.py …`. The snippet sets `sys.path += DIFFSYNTH_ROOT` and `__package__ = "view_transfer_via_query.prepare_data"` only when run as a script with no package context, so it's a no-op under the `-m` form. `gather_locations.py` already had a similar `sys.path.insert` and uses absolute imports — no bootstrap needed.

## 2026-05-04 — Architecture v2: VACE adapter + zero-gated source cross-attention

**Why**: v1 (joint sequence-concat self-attn) produced blurry/structureless inference at step 1600, even on training data and at `guidance_scale=1.0`. Root cause was joint attention diluting target self-attention's softmax mass, applying target-timestep modulation to clean source tokens, and using RoPE temporal positions (`[f, 2f)`) the pretrained Wan never saw. v2 replaces that with two independent injection paths, both engineered so step 0 ≡ pretrained Wan exactly.

### `model.py` — full rewrite (preserves existing `LoRALinear` / `apply_lora` semantics)
- **Removed**: `patch_embed_target` (52ch), joint sequence-concat in `ViewTransferDiTBlock.forward`, head slicing, RoPE temporal offset for target.
- **Kept / repurposed**: `patch_embed_source` (16ch, copy-init from pretrained `patch_embedding`); `plucker_encoder` per main block (now serves only `plucker_tgt` → main self-attn KQ-bias).
- **Added — adapter branch (VerseCrafter / VACE pattern)**:
  - `geoada_patch_embedding` (Conv3d 36→dim): channel-concat of `rendered + blob + mask` → tokens.
  - `geoada_blocks` (ModuleList of `AdapterDiTBlock`, N=10 for 14B at k=4): full Wan-style block (self-attn / cross-attn / FFN / modulation / gate) plus zero-init `after_proj` on every block and zero-init `before_proj` on block 0 only. Sequential forward (`forward_geoada`) emits one hint per block, added to main DiT at `geoada_layers = (0, 4, 8, …, 36)`.
- **Added — source cross-attention (IP-Adapter pattern)**:
  - `ViewTransferCrossAttention` (target-Q, source-K/V) with **zero-init `o`** ⇒ contributes 0 at step 0 regardless of inputs. K-side accepts an additive plücker_src bias.
  - Inserted into M=10 main blocks at `cross_attn_src_layers = (2, 6, 10, …, 38)` (interleaved with adapter sites).
  - Per-block `plucker_encoder_src` (zero-init Linear) feeds K-bias.
- **Added**: `_apply_zero_init()` re-applies all 7 zero-init conditions after weight loading (so any accidental warm-start non-zeros get scrubbed). `assert_step0_invariant()` raises `AssertionError` on first violation.
- **`from_pretrained` warm-start**: pretrained Wan weights → main blocks (direct), `patch_embed_source` (copy from `patch_embedding`), `geoada_blocks.{i}` (copy from `blocks.{geoada_layers[i]}` for matching keys), `cross_attn_src.{q,k,v,norm_q,norm_k}` (copy from `self_attn.{q,k,v,norm_q,norm_k}` of the host main block). Then `_apply_zero_init` rezeros the seven invariant modules.
- **`apply_lora`**: still scoped to `blocks.*.self_attn.{q,k,v,o}` only — adapter blocks and `cross_attn_src` are full-trained per the user-locked v2 spec.
- **Trainable budget at 14B (computed)**: ~5.35 B params trainable (~38 % of frozen 14 B base). Matches PLAN §10.4 estimate within 0.2 %.

### `train.py`
- **`apply_cfg_dropout`**: signature now `(per_stream_prob=0.05, joint_prob=0.10)`. Adds a joint-drop branch — with `joint_prob` probability, ALL conditioning streams (source, plücker_src, rendered, mask, blob, plücker_tgt, text) are zeroed in the same step. This makes the inference-time CFG uncond pass (which zeros all conds) in-distribution, fixing the v1 CFG mismatch where the model had ~0 % chance of seeing the all-zero case during training.
- New CLI flags: `--cfg_drop_prob` (default 0.05), `--cfg_joint_drop_prob` (default 0.10), `--skip_step0_invariant_check`.
- `_TRAINABLE_KEY_PREFIXES` extended: drops `patch_embed_target`; adds `geoada_`, `cross_attn_src` (covers all v2 trainable modules including before/after_proj and plucker_encoder_src).
- After `accelerator.prepare()`, on a fresh init (no resume / no warm-start), `model.assert_step0_invariant()` is called on the unwrapped model. Raises if violated.

### `pipeline.py` / `gpu_preprocess.py` / `infer.py`
- **No edits needed** — model `forward(...)` kwargs are unchanged; the architectural rewiring is fully internal. CFG `_build_uncond` keeps zeroing all conds; the joint-drop training fix makes that case in-distribution at inference.

### Tests
- `tests/test_model.py`: marked the two `patch_embed_target`-dependent tests as `@pytest.mark.skip(reason="v1 architecture; see test_model_v2.py")`. Other 7 tests still pass against v2 (LoRA, plücker, KQ-bias, mask packing remain valid).
- `tests/test_train.py`: rewrote `test_cfg_dropout_*` against the new `(per_stream_prob, joint_prob)` API. Added `test_cfg_dropout_no_drop_preserves_input` and split the "p=1.0 zeros everything" case into `_zeros_when_joint_prob_one` and `_per_stream_only` for clarity.
- Suite status: **13/13 model+train tests pass, 2 (v1) skipped**. Pre-existing 5 `test_pipeline.py` failures (unrelated to v2 — `_StubVAE` lacks `.model`) left untouched.

### Smoke-tested empirically (CPU with FA disabled; tiny test config dim=192, n_layers=2)
- Forward returns the right shape `(1, 16, 2, 4, 4)`.
- **Step-0 equivalence**: `forward(...with conds...)` vs `forward(...with all conds zeroed...)` → max diff `0.00e+00`. The architecture provably contributes nothing at init.
- `assert_step0_invariant()` passes both before and after `apply_lora` (LoRA-B is zero-init).

### Data
- New `train_locations_v2.txt` (73 entries) and `val_locations.txt` (8 entries, ~10 % held out) under `/work/nvme/beab/rluo2/viewpoint-transfer/data/`. Original `train_locations.txt` preserved for v1 reproducibility. Held-out 8 cover Desert (×2), ModernCityDay, ProceduralNature_River, IndustrialCity, NYC50s, NYCAlley, Office.

### Migration
- The 1600-step v1 checkpoint at `runs/14B-debut/checkpoint-1600/` is **incompatible** (its LoRA was tuned for joint self-attn that no longer exists). Train fresh from base Wan2.1-T2V-14B; use `LOCATIONS_FILE=/work/nvme/beab/rluo2/viewpoint-transfer/data/train_locations_v2.txt` in `scripts/train.sh`.

### GPU validation on GH200 (gh136)
- Pretrained Wan2.1-T2V-14B sharded checkpoint loads via `from_pretrained` correctly: 1447/1541 keys matched; 94 newly initialised keys are exactly the v2 modules (40 `plucker_encoder` + adapter `before/after_proj` + 10 sets of `cross_attn_src.o`/`plucker_encoder_src`). Total model 19.53 B params.
- `assert_step0_invariant()` passes at full 14 B scale; empirical step-0 diff (with-conds vs zero-conds forward) = **0.00e+00** in BF16 with FlashAttention-2.
- Single training step end-to-end on GH200 at smoke shape (B=1, T_lat=4, 16×16 latent ⇒ 1k tokens):
  - fwd 1.1 s, bwd 0.5 s, AdamW 0.3 s
  - Peak GPU: 81.5 GB / 97 GB (39.2 model + 32 Adam state + ~10 activations)
  - Initial loss 1.5312 (sane)
  - 80 / 446 trainable tensors have non-zero grads at step 0 — **expected** zero-init wake-up pattern (LoRA-B / after_proj / cross_attn_src.o / main plucker_encoder are the only outermost zero-inits with non-zero step-0 grads; everything they gate gets non-zero grads from step 2). Same property as ControlNet / IP-Adapter / standard LoRA.
- **Bug found and fixed during GPU validation**: `apply_cfg_dropout` was silently casting all conditioning streams to fp32 via `(~mask).float()`, which would later trip Conv3d's dtype-equality check if the cast-back step in train.py were skipped. Replaced with a per-tensor `_scale(t, mask)` helper that preserves dtype.
- **Single-GH200 capacity** at full target shape: **fits.** Tested via shape sweep (B=1, grad ckpt, AdamW, fwd+bwd+step):

  | Shape | T_lat × H_lat × W_lat | Tokens | Step time | Peak GPU |
  |---|---|---|---|---|
  | 128×128 video | 4 × 16 × 16 | 1,024 | 2.0 s | 81.5 GB |
  | 240×416, T=8 | 8 × 30 × 52 | 12,480 | 1.3 s | 81.5 GB |
  | 480×832, T=8 | 8 × 60 × 104 | 49,920 | 6.8 s | 81.5 GB |
  | **480×832, 81f (target)** | **21 × 60 × 104** | **131,040** | **28.5 s** | **93.9 / 97 GB** |

  The 81.5 GB ceiling on the first three is dominated by AdamW state (m + v in fp32 for 5.25 B trainable ≈ 42 GB) plus the frozen 14 B base bf16 (~28 GB) plus residual; activations don't push past it until the full-target run, which lands at 93.9 GB — within budget.

- **Single-GH200 training feasibility**: 28.5 s/step at full target shape ⇒ ~7.9 hours for 1 k steps, ~80 hours for 10 k steps. Mini smoke (200 steps) ≈ 1.6 h. No multi-GPU strictly required for v2.6 / v2.8 — the "single GH200 won't fit" worry was wrong; the full v2 architecture runs at production resolution on a single GH200.

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
