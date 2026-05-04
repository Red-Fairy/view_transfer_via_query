# View Transfer via Query — Project Plan

Stage-2 model for **camera-controlled video reshooting**: given a source perspective video
and a novel relative trajectory, generate the perspective video that camera would have
captured. Source and target share resolution and intrinsics; only the trajectory differs.

---

## 1. Two-Stage Approach

| Stage | Goal | Status |
|---|---|---|
| 1 | Input perspective video → 360° panorama at the input's first-frame camera position. Static (no agents) so depth lift is clean. | Not started — easier, deferred |
| 2 | (Source video, source first-frame 360 panorama, target trajectory) → target perspective video | **Active** |

Stage 2 assumes an agent-free static panorama already exists (in training, comes from
ground-truth `Pano_XX_static`).

---

## 2. Stage-2 Architecture

### 2.1 Base model
- **Wan2.1-T2V-14B** (`dim=5120`, 40 layers, 40 heads, head_dim=128, patch_size=[1,2,2])
- VAE: 16-ch latent, 8× spatial / 4× temporal compression
- Resolution: **480 × 832 × 81 frames** → latent **[16, 21, 60, 104]** → **32,760 tokens** per stream
- T5-XXL (UMT5) cross-attention; **CLIP path removed** (no image encoder)

### 2.2 Conditioning streams

| Stream | Shape (per sample) | Injection | Why |
|---|---|---|---|
| Noisy target latent | `[16, 21, 60, 104]` | channel-concat slot 0 | the diffusion target |
| Rendered (warped pano) latent | `[16, 21, 60, 104]` | channel-concat slot 1 | static-scene novel-view geometry |
| **Visibility mask** (4-frame packed) | `[4, 21, 60, 104]` | channel-concat slot 2 | which rendered pixels are valid vs need inpainting |
| Blob latent (gaussian agents) | `[16, 21, 60, 104]` | channel-concat slot 3 | dynamic-agent location hint |
| **Source perspective latent** | `[16, 21, 60, 104]` | **sequence-concat** with target tokens | appearance prior from input video |
| Source plücker rays | `[6, 21, 480, 832]` → 1536-dim per token | **KQ-bias** in self-attn (Lyra-2 strategy F) | source camera pose |
| Target plücker rays | same | KQ-bias on target tokens | target camera pose |
| Pre-computed T5 embedding | `[L_text, 4096]` | T5 cross-attention | text guidance (kept active even if empty) |

Total channels into target patch-embed: **52** (16+16+4+16). Patch-embed surgery: copy
pretrained 16-ch weights into first 16 input channels, zero-init the other 36.

### 2.3 Plücker pipeline (Strategy F, Lyra-2)
1. Compute 6D plücker rays at the **VAE-aligned latent timestamps only** (`[0, 4, 8, …, 80]` for 81 frames → 21 plücker frames). Skip per-sub-frame variation; the VAE has already aliased it.
2. Pixel-unshuffle 8× spatially: 6 → **384 ch** at latent grid.
3. Patchify (1×2×2): 384 × 4 = **1536** features per token.
4. Per-block `Linear(1536, dim)` (zero-initialized; identity at step 0).
5. Add as **pre-projection bias on Q and K only**: `q = W_Q(x + p), k = W_K(x + p), v = W_V(x)`. V is untouched, so pretrained content statistics are preserved.

### 2.4 Joint self-attention
- Concat `[source_tokens, target_tokens]` along sequence axis → **65,520 tokens**.
- 3D RoPE: source uses time indices `[0, 20]`, target uses `[21, 41]` (temporal offset disambiguates streams; spatial indices identical).
- Output: slice off source half, run `head` only on target tokens, unpatchify to `[16, 21, 60, 104]`.

### 2.5 Loss
**Flow matching, v-prediction** on target slice only:
- Sample timestep `t ~ Uniform[0, 1000]`, get `sigma` from `FlowMatchScheduler(template="Wan", training=True)`
- `z_t = (1-σ) · z_0 + σ · ε`
- Target: `v = ε - z_0`
- Loss: `weight(t) · MSE(v_pred, v)` with bell-shaped (BSMNTW) timestep weights from DiffSynth-Studio.

### 2.6 Classifier-free guidance dropout (training)
Independent per-stream drop with **p = 0.1** each:
- Source video + source plücker (coupled)
- Rendered + visibility mask (coupled)
- Blob video
- Target plücker
- Text embedding

---

## 3. Training Setup

### 3.1 Modes
| Mode | Trainable | Storage | Use |
|---|---|---|---|
| **LoRA** (default, rank=64) | LoRA adapters on attn QKVO + new modules (patch_embed_target, patch_embed_source, plücker encoders) | ~420M params | first iteration |
| Full finetune | all 14B | ~28 GB BF16 weights + optimizer | once LoRA proves the design |

Toggle with `--lora_rank 0` for full FT.

### 3.2 Optimizer / schedule
- AdamW, lr=1e-4, weight_decay=0.01, BF16 (Accelerate)
- Cosine schedule with linear warmup (1000 steps default)
- Gradient checkpointing always-on for the seq-concat'd transformer (65K tokens)

### 3.3 Hardware
- GH200, single-GPU initially. Multi-GPU via Accelerate for scale-up.
- Seq-concat doubles attention FLOPs; with FlashAttention-2/3 still tractable.

---

## 4. Data Pipeline

### 4.1 Source data (Unreal Engine renders)
Already on NVMe at `/share/ma/scratch/rundong/Unreal_Projects/outputs_non_arranged_cars/`.
20 scenes × ~6 locations × 2 panoramic trajectories × 240 frames @ 24 fps (4096×2048 equirect).

Per location:
```
camera_params.json       # UE LHS-Z-up, cm
agent_poses.json
Pano_00/{rgb.mp4, rgb/*.png, depth/*.exr}    dynamic
Pano_00_static/{rgb/*.png, depth/*.exr}      static (no agents)
Pano_01/...
Pano_01_static/...
```

### 4.2 Hybrid pre-compute / online split

**Offline (one-time per location):**
| Output | Producer | Why offline |
|---|---|---|
| `c2w.pt` per pano (OpenCV convention, meters) | `prepare_data/parse_cameras.py` | small, format conversion |
| `text_emb.pt` per scene caption | `prepare_data/encode_text.py` | UMT5-XXL is large; once is enough |
| `Pano_{tgt}/blobs/` (PNG dirs, `[T, 3, 2048, 4096]`) | user pre-generates from `agent_poses.json` | 3D gaussian agents rasterized into equirect at the target pano's camera positions |

User's `Pano_XX_static/{rgb,depth}` already exists from UE renders — no extra prep needed.

**Online (training time, on GPU):**
1. Sample `(source_pano, target_pano)` from `{(00, 01), (01, 00)}`
2. Sample temporal offset `t0 ∈ [0, 240-81]`
3. Sample two random perspective trajectories (yaw, pitch, roll, FOV with smooth jitter)
4. Load from NVMe:
   - 3 equirect windows of 81 frames: source dynamic RGB, target dynamic RGB, blob RGB
   - 1 static RGB frame at t0 (just one, not 81)
   - 1 static depth frame at t0 (one EXR; ~250 ms decode)
5. **Lift-and-render**: lift source-static (RGB + radial depth) at t0 → 3D world points → render to all 81 target perspective camera poses with z-buffer; outputs `rendered` and `visibility` directly in target perspective (skip 360 detour).
6. **equi2pers** the 3 equirect windows (source, target, blob) to `480×832×81`.
7. **VAE encode** the 4 RGB streams (source, target, rendered, blob) — skip mask.
8. Pack visibility mask: max-pool 8× spatial + 4-frame fold → 4-ch latent-aligned.
9. Compose perspective c2w: `pers_c2w = pano_c2w @ R_crop_4x4`
10. Plücker rays at latent timestamps (cheap).

### 4.3 Worker / GPU separation
- DataLoader workers (CPU): PNG decode from NVMe, return raw equirect tensors
- Main process (GPU): equi2pers + VAE encode + plücker, all in `training_step`
- Async prefetching overlaps I/O with model fwd/bwd

### 4.4 Latency budget (estimated per sample, BF16, GH200)
| Stage | Time |
|---|---|
| PNG decode 3× 81 frames @ 4K from NVMe | 0.7–2 s (parallel workers) |
| Static RGB (1 PNG) + depth (1 EXR) decode | 0.3–0.5 s |
| Lift-and-render (online, 81 target frames) | 2–4 s |
| Equi2pers 3 streams × 81 frames | 0.3–1 s |
| VAE encode 4 RGB videos | 2–6 s |
| Plücker, mask packing, etc. | <0.1 s |
| Model fwd+bwd (Wan2.1-14B, seq-concat 65K tokens) | 3–5 s |

**Key win from online lift-and-render**: the temporal offset `t0` is sampled per
sample, so each of the 240 frames in a sequence can serve as the source first-frame
panorama. With `t0=0` only (pre-compute) we get 1 sample per pano-pairing; online,
we get up to 160 (240−81+1) distinct samples per pairing — full data utilization.

Workers' I/O + equi2pers should overlap with the model step; VAE on the main GPU is the
most likely bottleneck. Mitigations if needed: smaller VAE encode batches, BF16 VAE,
splitting VAE across multiple GPUs.

---

## 5. File Structure (post-2026-05-04 refactor — self-contained submodule layout)

```
view_transfer_via_query/                       ← PROJECT_ROOT (the submodule root)
│
├── conftest.py                  pytest path setup; auto-disables FlashAttn for CPU runs
├── project_files/
│   ├── PLAN.md                      ← this file
│   ├── TODO.md
│   └── CHANGELOG.md
│
├── model.py                     ViewTransferDiT, AdapterDiTBlock, ViewTransferCrossAttention, LoRA (v2)
├── pipeline.py                  ViewTransferPipeline (inference)
├── plucker.py                   ray_condition + latent-timestamp helper
├── mask_utils.py                pack_mask (max-pool + 4-frame fold)
├── dataset.py                   online dataset (workers do PNG/EXR I/O)
├── gpu_preprocess.py            equi2pers + lift+render + VAE encode (GPU side)
├── prefetcher.py                CUDA-stream prefetcher
├── train.py                     Accelerate flow-matching loop (joint+per-stream CFG dropout)
├── infer.py                     batched CLI inference
│
├── scripts/
│   ├── _common.sh               ← shared path setup (sourced by all scripts)
│   ├── train.sh
│   ├── train_multinode.sh
│   ├── infer.sh
│   └── prep_data.sh
│
├── configs/
│   ├── accelerate/deepspeed.yaml.tmpl
│   └── deepspeed/zero2_bf16.json
│
├── prepare_data/
│   ├── parse_cameras.py         UE → OpenCV c2w
│   ├── extract_perspectives.py  equi2pers + trajectory sampler
│   ├── lift_and_render.py       point-cloud z-buffer rasterise
│   ├── encode_latents.py        Wan2.1 VAE wrapper
│   ├── encode_text.py           UMT5-XXL wrapper
│   ├── agent_mask.py            (utility, NOT the model's mask channel)
│   ├── video_io.py
│   ├── gather_locations.py
│   ├── prep_all_scenes.py
│   └── run_prep.py              cameras / text / verify subcommands
│
├── runs/                        ← v2 training outputs land here (was DiffSynth-Studio/runs/)
├── infer_out/                   ← v2 inference outputs land here (was DiffSynth-Studio/infer_out/)
│
└── tests/                       28 tests; runnable via `pytest tests/` from project root
    ├── test_model.py            (2 v1 tests skipped via @pytest.mark.skip)
    ├── test_train.py            (CFG dropout + training step)
    ├── test_dataset.py
    ├── test_prefetcher.py
    └── test_pipeline.py         (5 tests — pre-existing _StubVAE failure, unrelated to v2)
```

### Path conventions

- **`PROJECT_ROOT`** = `view_transfer_via_query/` — owns `runs/`, `infer_out/`, `configs/`, all checkpoints written by training.
- **`DIFFSYNTH_ROOT`** = `view_transfer_via_query/..` = `DiffSynth-Studio/` — provides the `diffsynth` library and pretrained Wan2.1 model files (under `${DIFFSYNTH_ROOT}/models/Wan-AI/`).
- All bash scripts source `scripts/_common.sh` to derive both roots from script location and export `PYTHONPATH=${DIFFSYNTH_ROOT}:${PYTHONPATH}` — so they work from any CWD, and `python -m view_transfer_via_query.X` works without manual path setup.
- `conftest.py` does the same for `pytest`.

---

## 6. Coordinate Conventions

### 6.1 Camera frames
- **UE camera**: +X forward, +Y right, +Z up; LHS
- **OpenCV camera**: +X right, +Y down, +Z forward; RHS
- Permutation `M_CV_TO_UE_CAM @ v_cv = v_ue`:
  ```
  cv_x = ue_y;  cv_y = -ue_z;  cv_z = ue_x
  ```

### 6.2 World frame
- UE world is LHS Z-up. We flip Y to get RHS: `F_UE_TO_CV_WORLD = diag(1, -1, 1)`.
- Units: cm → m.
- All saved `c2w.pt` is OpenCV convention in meters.

### 6.3 Equirectangular sampling
The equirect was rendered by UE, so the `(u, v)` ↔ ray mapping uses UE convention:
- `lon = atan2(y_ue, x_ue)`, `lat = atan2(z_ue, sqrt(x²+y²))`
- `u = (lon + π)/(2π) · W`, `v = (π/2 - lat)/π · H`
- Center pixel `(W/2, H/2)` → ray `(1, 0, 0)` in UE camera frame (forward).

### 6.4 Yaw/pitch/roll for perspective crops (OpenCV)
- `+yaw` = look right (rotate around +Y down-axis)
- `+pitch` = look down (rotate around +X right-axis; sign flipped from standard right-hand rule for intuitiveness)
- `+roll` = clockwise tilt (rotate around +Z forward-axis)

---

## 7. Critical Decisions (and Why)

| Decision | Rationale |
|---|---|
| Wan2.1-T2V-14B (not Wan2.2-TI2V-5B as v1 used) | 16-ch VAE is more standard; T2V removes CLIP we don't want; GH200 handles 14B |
| Strategy F plücker (latent-timestamp sample + KQ-bias) | Same backbone as Lyra-2 ships; cheapest; preserves V (content) statistics |
| Joint self-attention over `[source, target]` (not cross-attn) | Matches Lyra-2 / 360Anything; richer interaction, doubles tokens |
| Raw mask downsample (not VAE-encode) | Mask is binary; VAE adds noise; matches Wan2.1-I2V's 4-channel pattern |
| Pre-compute lift-and-render in 360 space | Avoids per-trajectory rendering at training time; preserves perspective diversity (any perspective can be sampled from the 360 rendered) |
| Online VAE encoding | Diversifies trajectories; on NVMe + BF16 the latency is acceptable; matches Cosmos-Predict-2.5 practice |
| LoRA first (rank=64), full-FT supported | Safer for first iteration; LoRA validates the design before committing to 14B fine-tune |
| CFG dropout p=0.1 per stream | Standard; allows multi-condition CFG at inference |

---

## 8. Open Work / Next Steps

| # | Task | Owner | Status |
|---|---|---|---|
| 1 | ~~Pre-compute lift-and-render in 360 space~~ — **superseded by online lift-and-render** | — | done (online) |
| 2 | Pre-generate 360 blob videos (gaussian-rasterized agents) | user | in progress |
| 3 | Rename `compute_masks.py` → `agent_mask.py` (utility, not training-path) | claude | done |
| 4 | Slim `run_prep.py` to `(parse cameras, encode T5, verify)` | claude | done |
| 5 | Refactor `dataset.py` for online equi2pers + VAE encoding | claude | done |
| 6 | Add prefetcher / async pipeline so disk I/O overlaps model step | claude | done |
| 7 | Implement online lift-and-render module | claude | done |
| 8 | (optional) Convert EXR depth → fp16 packed tensor for faster I/O | future | deferred |
| 9 | End-to-end smoke test on one location with all components | claude + user | pending |
| 10 | Stage 1 (panorama prediction) | future | not started |

---

## 9. References (papers / code we follow)

- **Wan2.1-T2V-14B** (DiffSynth-Studio fork) — base DiT, VAE, T5
- **Lyra-2** (NVIDIA, arXiv 2604.13036) — plücker KQ-bias pattern, same backbone
- **VACE / VerseCrafter** (`/work/nvme/beab/rluo2/VerseCrafter`) — adapter-branch design with `before_proj` / `after_proj` zero-init injection (drives v2; see §10)
- **IP-Adapter** (arXiv 2308.06721) — decoupled cross-attention with zero-init output projection (the `cross_attn_src` design in §10)
- **ControlNet** (Zhang & Agrawala, ICCV 2023) — zero-conv injection principle
- **Wan2.1-I2V** — 4-channel mask packing convention for visibility
- **TrajectoryCrafter** (arXiv 2503.05638) — point-cloud render as channel-concat condition
- **Cosmos-Predict-2.5** — online VAE encoding with prefetching at 14B scale

---

## 10. Architecture v2 — VACE adapter + zero-gated source cross-attention (2026-05-04)

**Status**: spec; awaiting user approval before code edits.
**Supersedes**: §2.4 (joint self-attention), §2.6 (per-stream CFG dropout — joint dropout added).
**Backward compatibility**: none. The 1600-step `runs/14B-debut/checkpoint-1600/trainable_params.pt` is incompatible with the new architecture (its LoRA was tuned for joint self-attn that no longer exists). Train fresh from base Wan2.1-T2V-14B.

### 10.1 Why the revision

Inference at step 1600 produced blurry, structureless outputs even on training data and even at `guidance_scale=1.0`. Confirmed root causes:

1. **Joint sequence-concat self-attention dilutes target self-attn**. With `x = [x_src; x_tgt]`, target queries softmax over `[k_src; k_tgt]`. Clean source K's high-magnitude scores dominate noisy target K, starving target↔target attention. The pretrained prior (which only ever saw target↔target attention) is broken at step 0 and LoRA never recovers it within 1600 steps.
2. **Timestep modulation is applied to clean source tokens**. `shift/scale/gate_msa` derived from target `t` is broadcast across the concatenated sequence, warping clean source content by an unrelated noise level in every block.
3. **RoPE temporal extrapolation**. Target tokens at positions `[f, 2f)` were never seen in pretraining.
4. **CFG mismatch (independent contributor)**. Training drops conditioning streams *independently* at p=0.1 per stream; P(all-zero) ≈ 1e-5. Inference uncond zeros all streams jointly → OOD prediction. Confirmed not the dominant cause (g=1.0 still degrades), but worth fixing.

The v2 architecture replaces (1)-(3) with two zero-init injection paths so step 0 is exactly pretrained Wan; (4) is fixed by adding a joint-drop branch in `apply_cfg_dropout`.

### 10.2 Architecture overview (replaces §2.4)

```
inputs ─────────────────────────────────────────────────────────────────┐
 noisy_latent (16ch) ──► patch_embedding (frozen, original Wan) ──► x  │
                                                                        │
 rendered_latent (16ch) ┐                                               │
 blob_latent (16ch)    ─┼─ channel-concat (36ch) ──► geoada_patch_emb ──┤──► c
 mask_packed (4ch)      ┘                                               │
                                                                        │
 source_latent (16ch) ───► patch_embed_source ──────────────► x_src     │
 plucker_src ────► prepare_plucker ──────► plucker_src_tokens           │
 plucker_tgt ────► prepare_plucker ──────► plucker_tgt_tokens           │
 text_emb ───────► text_embedding (pretrained) ──► text_context         │
                                                                        │
 ┌────────────────────────────────────────────────────────────┐         │
 │ Adapter branch (sequential over N=10 adapter blocks)        │  c ────┘
 │   block 0 :  c = before_proj(c) + x      (before_proj=0)    │  x ────► (block 0 of main DiT input)
 │   block i :  c ← AdapterDiTBlock(c)                         │
 │   hint_i  =  after_proj_i(c)              (after_proj_i=0)  │
 │ Outputs hints[0..N-1], one per main-DiT injection layer.    │
 └────────────────────────────────────────────────────────────┘
                                                                        │
 ┌──────────────────────────────────────────────────────────┐           │
 │ Main DiT (40 blocks)                                      │           │
 │   for i in range(40):                                     │           │
 │     x = block_i(x, ...)                          # frozen + LoRA      │
 │       (self_attn target-only, +plucker_tgt KQ-bias)       │           │
 │       (text cross_attn — frozen)                          │           │
 │       (FFN — frozen)                                       │           │
 │     if i in cross_attn_src_layers:               # M=10               │
 │       x = x + cross_attn_src_i(x, x_src, plucker_src)     │           │
 │     if i in geoada_layers:                       # N=10               │
 │       x = x + hints[geoada_layers_mapping[i]]              │           │
 │ x = head(x, t_emb)                                         │           │
 └──────────────────────────────────────────────────────────┘
```

**Layer assignments (k=4, interleaved)**:
- `geoada_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]` (N=10 hint injection points; matches VerseCrafter default at k=4)
- `cross_attn_src_layers = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]` (M=10, interleaved between adapter sites)

### 10.3 Components — what's added / kept / removed

| Component | Status | Where | Init |
|---|---|---|---|
| `patch_embedding` (16 → dim, k=1×2×2) | **kept, frozen** | model root | pretrained Wan |
| `patch_embed_target` (52 → dim) | **REMOVED** | — | — |
| `patch_embed_source` (16 → dim, k=1×2×2) | **kept, full-train** | model root | copy from `patch_embedding` |
| `geoada_patch_embedding` (36 → dim, k=1×2×2) | **NEW, full-train** | model root | Xavier (or copy first 16ch from `patch_embedding`); bias zero |
| Adapter blocks `geoada_blocks` (N=10) | **NEW, full-train** | ModuleList | base WanAttentionBlock weights copied from corresponding main DiT block |
| `geoada_blocks[0].before_proj` (Linear dim→dim) | **NEW, full-train** | inside adapter block 0 | weight=0, bias=0 |
| `geoada_blocks[i].after_proj` (Linear dim→dim) | **NEW, full-train** | inside every adapter block | weight=0, bias=0 |
| Main DiT `self_attn.{q,k,v,o}` LoRA (rank 64) | **kept** | inside every main block | A: Kaiming, B: 0 |
| Main DiT `plucker_encoder` (1536 → dim) | **kept, full-train** | inside every main block | weight=0 (zero-init) |
| Main DiT joint-attn source-token concat | **REMOVED** | — | — |
| Main DiT 3D-RoPE temporal offset for target | **REMOVED** (back to `[0, f)`) | model `forward` | — |
| Main DiT text cross-attn / FFN / norms / modulation | **kept, frozen** | inside every main block | pretrained Wan |
| `cross_attn_src` (M=10 of 40 main blocks) | **NEW, full-train** | inside selected main blocks | base q/k/v copied from main block's `self_attn.{q,k,v}`; o weight=0, o bias=0 |
| `cross_attn_src.norm_q`, `norm_k` (RMSNorm) | **NEW, full-train** | inside `cross_attn_src` | one |
| `norm_src_q`, `norm_src_k` (LayerNorm, params-free) | **NEW** | wrapping `cross_attn_src` | n/a |
| `plucker_encoder_src` (1536 → dim) | **NEW, full-train** | per `cross_attn_src` block (M=10) | weight=0 (zero-init) |
| `head` | **kept, frozen** | model root | pretrained Wan |

**Plücker routing** (locked):
- `plucker_tgt` → main self-attn KQ-bias via per-block `plucker_encoder` (unchanged; weight=0 at init).
- `plucker_src` → `cross_attn_src` K-side bias via per-block `plucker_encoder_src` (new; weight=0 at init).

**Mask** (locked): `mask_packed` is a *channel-concat input* to `geoada_patch_embedding` (not separately injected). 36 = 16 (rendered) + 16 (blob) + 4 (mask).

### 10.4 Trainable parameter budget (Wan2.1-T2V-14B, dim=5120, ffn_dim=13824, num_layers=40)

Per-block reference numbers:
- Wan main DiT block ≈ 350M params (self_attn ~105M + cross_attn ~105M + FFN ~141.6M + norms/modulation/gate trivial).

| Component | Per-instance trainable | Count | Total |
|---|---|---|---|
| **A. Main DiT augmentation** | | | |
| LoRA on `self_attn.{q,k,v,o}` (rank 64) | 4 × 2 × 64 × 5120 = 2.62M | 40 blocks | 105M |
| `plucker_encoder` (1536 → 5120) | 7.86M | 40 blocks | 314M |
| **A subtotal** | | | **~420M** |
| **B. Adapter branch (full-train, N=10, k=4)** | | | |
| `geoada_patch_embedding` (Conv3d 36→5120, k=1×2×2) | 36×5120×4 + 5120 ≈ 742K | 1 | 0.74M |
| Adapter WanAttentionBlock weights | ≈ 350M | 10 blocks | 3,500M |
| `before_proj` (Linear 5120→5120) | 5120² + 5120 ≈ 26.22M | 1 (block 0 only) | 26.22M |
| `after_proj` (Linear 5120→5120) | ≈ 26.22M | 10 blocks | 262.2M |
| **B subtotal** | | | **~3.79B** |
| **C. Source cross-attention (full-train, M=10, interleaved)** | | | |
| `patch_embed_source` (Conv3d 16→5120, k=1×2×2) | 16×5120×4 + 5120 ≈ 333K | 1 | 0.33M |
| `cross_attn_src.{q,k,v,o}` + 2 RMSNorms | 4×26.22M + 2×5120 ≈ 105M | 10 blocks | 1,049M |
| `plucker_encoder_src` (1536 → 5120) | 7.86M | 10 blocks | 78.6M |
| **C subtotal** | | | **~1.13B** |
| **GRAND TOTAL TRAINABLE** | | | **~5.34B on top of frozen 14B** |

Sanity reference: full Wan2.1-T2V-14B is ~14.0B params; trainable footprint is ~38% of the base.

### 10.5 Memory budget (8 × 80 GB GPUs, BF16, ZeRO-2)

| Bucket | Per-rank size | Notes |
|---|---|---|
| Frozen base (14B, BF16, replicated) | ~28 GB | not sharded under ZeRO-2 |
| Trainable params (5.34B, BF16, replicated) | ~10.7 GB | not sharded under ZeRO-2 |
| Master weights (5.34B, FP32, sharded) | 5.34B × 4 / 8 = ~2.7 GB | ZeRO-2 shards |
| Adam state (m + v, FP32, sharded) | 5.34B × 8 / 8 = ~5.3 GB | ZeRO-2 shards |
| Gradients (BF16, sharded) | 5.34B × 2 / 8 = ~1.3 GB | ZeRO-2 shards |
| Activations (with grad checkpointing, 81f×480×832, ~33K target tokens × 5120 dim) | ~10–20 GB | depends on cross-attn KV cache + adapter sequential graph |
| **Total per rank (peak)** | **~58–68 GB** | fits 80 GB; tight enough that we may want ZeRO-3 |

**If activations push us over**, switch DeepSpeed config to ZeRO-3 — that shards the 5.34B trainable params and the 14B frozen base, freeing ~35 GB / rank. Saves us at the cost of slower forward (allgather on every block). Bake a `zero3_bf16.json` next to the existing `zero2_bf16.json`; pick at launch via env var.

### 10.6 Step-0 invariant

The architecture is constructed so that at step 0:

```
ViewTransferDiT.forward(noisy_latent, rendered_latent, mask_packed, blob_latent,
                       source_latent, plucker_src, plucker_tgt, timestep, text_emb)
≡ WanModel.forward(noisy_latent, timestep, text_emb)
```

This requires **all** of the following inits:

1. `patch_embedding.weight, .bias` ← pretrained Wan (unchanged).
2. `geoada_blocks[0].before_proj.weight = 0`, `.bias = 0` → first adapter block sees `c = 0 + x = x`.
3. `geoada_blocks[i].after_proj.weight = 0`, `.bias = 0` for all i → all `hints[i] = 0`.
4. `geoada_patch_embedding.bias = 0` (so a zeroed `before_proj` truly maps to zero regardless of conditioning input).
5. All LoRA `lora_B.weight = 0` (default).
6. All `plucker_encoder.weight = 0` and `plucker_encoder_src.weight = 0` (zero-init).
7. `cross_attn_src[i].o.weight = 0`, `.bias = 0` for all M cross-attn-equipped blocks.

We add `model.assert_step0_invariant()` that checks all seven conditions, called inside `from_pretrained` after weight loading and (separately) inside `train.py` right before the loop starts. The unit test `tests/test_model_v2.py::test_step0_equivalence` does the empirical check: forward through `ViewTransferDiT` with random `noisy_latent` and zero conds, compare to a pretrained-Wan reference forward, must match within `atol=1e-4`.

### 10.7 Implementation steps (no code changes until user approval)

Each step is independently verifiable; do not proceed to the next until the current step's check passes.

**Step v2.1 — Refactor `model.py`** (~1 day)
- Delete: `patch_embed_target`, `ViewTransferDiTBlock` joint sequence-concat path, source half of RoPE, head slicing.
- Add: `AdapterDiTBlock` (= `ViewTransferDiTBlock` minus plücker, plus `before_proj` and `after_proj`, sequential output convention matching VerseCrafter's `forward(c, x)` signature).
- Add: `geoada_patch_embedding` Conv3d, `geoada_blocks` ModuleList, `geoada_layers` / `geoada_layers_mapping`.
- Add: `ViewTransferCrossAttention` (q/k/v/o + RMSNorms; o zero-init).
- Add: `cross_attn_src_layers`; insert `cross_attn_src`, `norm_src_q`, `norm_src_k`, `plucker_encoder_src` into selected main blocks.
- Modify `ViewTransferDiT.forward`:
  - patch-embed target (16ch only, plain `patch_embedding`).
  - patch-embed source (`patch_embed_source`).
  - Build `c` from `geoada_patch_embedding(concat([rendered, blob, mask]))`.
  - Build target-only RoPE freqs at `[0, f)`. Source RoPE freqs also at `[0, f)`.
  - Run `forward_geoada(x, c, ...)` → produces `hints[0..N-1]` via sequential adapter forward.
  - Loop over 40 main blocks; in each: (a) self-attn + plücker_tgt KQ-bias; (b) optional `cross_attn_src(x, x_src, plucker_src)`; (c) text cross-attn; (d) FFN; (e) optional `+ hints[geoada_layers_mapping[i]]`.
  - Head on `x` (target-only, no slicing).
- Modify `ViewTransferDiT.from_pretrained`:
  - Load Wan state dict into `patch_embedding`, all main DiT blocks, head, text_embedding, time_embedding, time_projection (unchanged).
  - Initialize `patch_embed_source` as copy of `patch_embedding` first 16ch.
  - Initialize `geoada_blocks[i]` from copy of corresponding main DiT block at `geoada_layers[i]` (i.e., adapter block 0 ← main block 0, adapter block 1 ← main block 4, etc.). The `before_proj` / `after_proj` get zero-init.
  - Initialize `cross_attn_src[j].{q,k,v}` base weights as copy of corresponding main block's `self_attn.{q,k,v}` (helps optimization). `cross_attn_src[j].o` zero-init.
  - Call `assert_step0_invariant()`.
- Update `apply_lora` to apply to main DiT `self_attn.{q,k,v,o}` only (not adapter blocks, not cross_attn_src). Adapter and cross_attn_src are full-trained.

**Step v2.2 — Update `train.py`**
- `_TRAINABLE_KEY_PREFIXES` ← `("lora", "plucker_encoder", "plucker_encoder_src", "patch_embed_source", "geoada_", "cross_attn_src", "before_proj", "after_proj")`.
- `freeze_base()`: extend trainable prefixes to include the new modules.
- `apply_cfg_dropout`: keep per-stream drops at 5% each, add a joint-drop branch at p=0.10 (zero ALL streams simultaneously; trains the all-zero unconditional case for proper CFG).
- Call `model.assert_step0_invariant()` after `accelerator.prepare()` and before the training loop starts.

**Step v2.3 — Update `pipeline.py`**
- `_build_uncond` keeps zeroing all conditioning streams. With the joint-drop training fix this is now in-distribution.
- No structural changes needed — generate signature unchanged.

**Step v2.4 — Update `gpu_preprocess.py`**
- Output dict keys unchanged. The model now interprets `rendered_latent`, `blob_latent`, `mask_packed` differently internally (channel-concat into adapter input, not target patch-embed), but the preprocessing produces the same tensors. No edit needed.

**Step v2.5 — Tests** (`tests/test_model_v2.py`)
- `test_step0_equivalence`: empirical zero-init forward equals pretrained Wan forward (atol=1e-4).
- `test_assert_step0_invariant`: passes at init, fails after a fake non-zero perturbation to any of the 7 conditions.
- `test_param_counts`: assert each component's trainable param count matches §10.4.
- `test_grad_flow`: one fake training step at lr=0; verify nonzero grads on `cross_attn_src.o.weight`, `after_proj.weight`, `before_proj.weight`, `lora_B`, `plucker_encoder*`.
- `test_forward_shape`: end-to-end forward with realistic shapes returns correct output shape.

**Step v2.6 — Mini smoke training** (~half day, single GPU)
- Train on 4 locations × 200 steps with `--log_video_every 50`.
- Loss should monotonically trend below pretrained-Wan-on-noise baseline within 100 steps (because plücker/source/adapter signals give the model new information).
- Logged training-artifact videos should still show valid conditioning streams.

**Step v2.7 — Decision: ZeRO-2 vs ZeRO-3**
- After mini smoke, profile peak GPU memory. If close to 80 GB, switch to ZeRO-3.
- Bake `zero3_bf16.json` config; toggle via `DEEPSPEED_CONFIG` env in `train.sh`.

**Step v2.8 — Full training**
- Multi-GPU via `train_multinode.sh`. Fresh start (no warm-load).
- Target: 10–20k steps before next eval. Save every 500.

**Step v2.9 — Inference verification**
- Hold out a `val_locations.txt` (split a few entries from `train_locations.txt` BEFORE step v2.8; never include them in training). Eval on both train and val.
- Sweep `guidance_scale ∈ {1.0, 1.5, 2.5}` once joint-drop training has had ≥5k steps to make CFG well-defined.

### 10.8 Migration checklist (what to delete / refactor)

- `model.py`: delete `patch_embed_target` field and `from_pretrained` zero-pad logic for it. Delete joint-concat in `ViewTransferDiT.forward`. Delete RoPE temporal offset for target.
- `train.py`: update `_TRAINABLE_KEY_PREFIXES`. Update `freeze_base`. Update `apply_cfg_dropout`. Add invariant assert call.
- `pipeline.py`: no changes.
- `gpu_preprocess.py`: no changes.
- `infer.py`: no changes.
- `tests/test_model.py`: many tests will reference the old joint-attn signatures → mark with `@pytest.mark.skip(reason="superseded by v2 architecture; see test_model_v2.py")` rather than delete (preserves history).
- `runs/14B-debut/checkpoint-1600/`: leave on disk; do NOT load. Rename run dir to `runs/14B-debut-v1-deprecated/` for clarity.

### 10.9 Open decisions (all locked 2026-05-04)

| # | Decision | Final |
|---|---|---|
| 1 | k for adapter (every k-th block) | **k=4 → N=10** |
| 2 | M for `cross_attn_src` insertion sites | **M=10**, layers [2,6,10,…,38] (interleaved with adapter) |
| 3 | Adapter / cross_attn training mode | **full-train** (no LoRA on new modules) |
| 4 | Plücker routing | **plucker_tgt → main self-attn KQ-bias; plucker_src → cross_attn_src K-bias** |
| 5 | Mask packing routes via | **adapter input concat (36ch)** |
| 6 | Backward compat with old checkpoint | **none — train fresh** |
| 7 | DeepSpeed config | **start ZeRO-2**; switch to ZeRO-3 only if profiling shows we're tight |
| 8 | CFG dropout strategy | **per-stream 5% + joint 10%** |
| 9 | LR / warmup for v2 | start with current 1e-4 / 100, only change if smoke shows instability |
| 10 | Held-out val split size | 8 of 81 (~10%); see `data/val_locations.txt` (deterministic, seed=20260504) |

### 10.10 Risks

1. **Memory blow-up at 81-frame latent + 10 cross-attn-equipped blocks**. Cross-attn complexity is O(T_lat × H × W)² per block. Mitigate by (i) confirming `AttentionModule` routes through FlashAttention; (ii) ZeRO-3 fallback; (iii) reducing M to 5 if needed.
2. **Adapter init from main block weights produces immediate non-trivial hint contributions** as soon as `after_proj` moves off zero. This is desired (signal flow) but means early-training instability is possible. Mitigate with conservative LR (start at 5e-5 instead of 1e-4) and longer warmup (2000 steps).
3. **`cross_attn_src.o` zero-init at block scale ~5120²**: the gradient for `o.weight` at step 0 is `(d L / d output) · attn_out^T`. Since `output = 0` at init, downstream gradients are well-defined; the gradient on `o` itself is `attn_out^T · upstream_grad`, both nonzero. So learning starts on step 1. Verify in test_grad_flow.
4. **Source tokens and target tokens have different effective noise levels** (source clean, target noisy). The cross-attention has to learn to attend across this mismatch — should be straightforward but might benefit from passing source through the same modulation as target (alternative we can A/B later).
5. **No held-out eval set today**: §10.7 step v2.9 requires us to construct one *before* committing to full training, otherwise we'll re-make the "infer-on-train-data" mistake from v1.

