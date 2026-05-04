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

## 5. File Structure

```
view_transfer_via_query/
├── PLAN.md                      ← this file
├── TODO.md
├── CHANGELOG.md
│
├── model.py                     ViewTransferDiT, KQ-bias self-attn, LoRA
├── plucker.py                   ray_condition + latent-timestamp helper
├── mask_utils.py                pack_mask (max-pool + 4-frame fold)
├── dataset.py                   ⚠ to be refactored for online encoding
├── train.py                     Accelerate flow-matching loop
│
├── prepare_data/
│   ├── parse_cameras.py         ✓ UE → OpenCV c2w
│   ├── extract_perspectives.py  ✓ equi2pers + trajectory sampler (lib for online use)
│   ├── encode_latents.py        ✓ Wan2.1 VAE wrapper
│   ├── encode_text.py           ✓ UMT5-XXL wrapper
│   ├── compute_masks.py         ⚠ misnamed (does agent diff, NOT visibility) — to be renamed
│   ├── video_io.py              ✓ PNG / mp4 loaders
│   └── run_prep.py              ⚠ to be slimmed: only camera parse + T5 encode now
│
└── tests/                       36 unit tests, all passing
    ├── test_model.py
    ├── test_train.py
    └── prepare_data/tests/test_parse_cameras.py, test_extract_perspectives.py
```

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
- **Lyra-2** (NVIDIA, arXiv 2604.13036) — plücker KQ-bias, joint self-attn pattern, same backbone
- **360Anything** (arXiv 2601.16192) — sequence-concat conditioning + temporal-offset RoPE
- **Wan2.1-I2V** — 4-channel mask packing convention for visibility
- **TrajectoryCrafter** (arXiv 2503.05638) — point-cloud render as channel-concat condition
- **Cosmos-Predict-2.5** — online VAE encoding with prefetching at 14B scale
