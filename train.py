"""Training loop for ViewTransferDiT (Stage 2) with online VAE encoding.

Pipeline at each step:
    DataLoader workers (CPU)         : decode PNG sequences from NVMe → uint8 tensors
    CUDAStreamPrefetcher (side stream): equi2pers + VAE + plucker + mask pack
    Main process (default stream)    : flow-matching fwd/bwd

Usage:
    accelerate launch view_transfer_via_query/train.py \\
        --data_root /path/to/prepped_data \\
        --pretrained_dit /path/to/wan2.1-t2v-14b.safetensors \\
        --vae_ckpt      /path/to/Wan2.1_VAE.pth \\
        --output_dir    ./runs/exp01 \\
        --lora_rank 64 --lr 1e-4 --batch_size 1 --gradient_checkpointing
"""

from __future__ import annotations

import os
import sys
import argparse
import math
import time
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from tqdm.auto import tqdm
from torch.profiler import (
    ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.core.loader.file import load_state_dict as _diffsynth_load_state_dict

from .model import MODEL_CONFIGS, ViewTransferDiT, apply_lora
from .dataset import ViewTransferDataset, collate_view_transfer
from .gpu_preprocess import gpu_preprocess
from .prefetcher import CUDAStreamPrefetcher
from .prepare_data.encode_latents import load_wan_vae


_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse_dtype(name: str) -> torch.dtype:
    return _DTYPE_MAP[name]


def load_dit_state_dict(ckpt_path: str) -> dict:
    """Load a DiT checkpoint via diffsynth's loader.

    Accepts:
      - a single .safetensors / .pth file
      - a directory: globs `diffusion_pytorch_model*.safetensors` (sharded HF release)
    """
    import glob
    if os.path.isdir(ckpt_path):
        files = sorted(glob.glob(os.path.join(ckpt_path, "diffusion_pytorch_model*.safetensors")))
        if not files:
            raise FileNotFoundError(
                f"No `diffusion_pytorch_model*.safetensors` files in {ckpt_path}"
            )
        print(f"  Sharded checkpoint: {len(files)} files")
        return _diffsynth_load_state_dict(files)
    return _diffsynth_load_state_dict(ckpt_path)


# ── CFG dropout (v2: per-stream + joint, see PLAN.md §10.9) ─────────────────


def apply_cfg_dropout(
    batch: dict,
    per_stream_prob: float = 0.05,
    joint_prob: float = 0.10,
) -> dict:
    """v2 CFG dropout: each stream is dropped independently at `per_stream_prob`,
    AND with `joint_prob` ALL streams are jointly zeroed in the same step. The joint-drop
    branch trains the model on the all-zero unconditional case so inference-time CFG
    (which uses zero-everything as uncond) is in-distribution.

    Effective per-stream drop rate ≈ per_stream_prob + joint_prob - per_stream_prob*joint_prob.
    P(all streams zero) ≈ joint_prob (per-stream contribution is negligible for 5 streams at 5%).
    """
    B = batch["source_latent"].shape[0]
    device = batch["source_latent"].device

    # Per-tensor mask: keep dtype identical to the input (avoids silent bf16→fp32
    # promotion that would later trip Conv3d's dtype-equality check).
    def _scale(t: torch.Tensor, drop_mask: torch.Tensor) -> torch.Tensor:
        keep = (~drop_mask).view([B] + [1] * (t.dim() - 1)).to(t.dtype)
        return t * keep

    # Joint drop mask — when True, zeros every stream simultaneously
    drop_all = (torch.rand(B, device=device) < joint_prob)

    # Per-stream drops (independent of each other), OR-combined with joint drop
    drop_src    = (torch.rand(B, device=device) < per_stream_prob) | drop_all
    drop_render = (torch.rand(B, device=device) < per_stream_prob) | drop_all
    drop_blob   = (torch.rand(B, device=device) < per_stream_prob) | drop_all
    drop_plk    = (torch.rand(B, device=device) < per_stream_prob) | drop_all
    drop_txt    = (torch.rand(B, device=device) < per_stream_prob) | drop_all

    # source video coupled with source plücker
    batch["source_latent"] = _scale(batch["source_latent"], drop_src)
    batch["plucker_src"]   = _scale(batch["plucker_src"],   drop_src)
    # rendered + mask coupled
    batch["rendered_latent"] = _scale(batch["rendered_latent"], drop_render)
    batch["mask_packed"]     = _scale(batch["mask_packed"],     drop_render)
    # blob
    batch["blob_latent"]   = _scale(batch["blob_latent"],   drop_blob)
    # target plücker
    batch["plucker_tgt"]   = _scale(batch["plucker_tgt"],   drop_plk)
    # text
    batch["text_emb"]      = _scale(batch["text_emb"],      drop_txt)

    return batch


# ── Training step (assumes batch already preprocessed) ───────────────────────


def training_step(
    model: ViewTransferDiT,
    batch: dict,
    scheduler: FlowMatchScheduler,
    gradient_checkpointing: bool = False,
) -> torch.Tensor:
    target_latent = batch["target_latent"]
    B = target_latent.shape[0]
    device = target_latent.device

    num_steps = len(scheduler.timesteps)
    t_idx = torch.randint(0, num_steps, (B,), device=device)
    timesteps = scheduler.timesteps.to(device)[t_idx]
    # Cast sigmas to target_latent dtype to avoid float promotion (e.g., when target is BF16)
    sigmas = scheduler.sigmas.to(device=device, dtype=target_latent.dtype)[t_idx].view(B, 1, 1, 1, 1)

    noise = torch.randn_like(target_latent)
    noisy_latent = (1 - sigmas) * target_latent + sigmas * noise
    velocity_target = noise - target_latent

    velocity_pred = model(
        noisy_latent=noisy_latent,
        rendered_latent=batch["rendered_latent"],
        mask_packed=batch["mask_packed"],
        blob_latent=batch["blob_latent"],
        source_latent=batch["source_latent"],
        plucker_src=batch["plucker_src"],
        plucker_tgt=batch["plucker_tgt"],
        timestep=timesteps,
        text_emb=batch["text_emb"],
        use_gradient_checkpointing=gradient_checkpointing,
    )

    weights = scheduler.linear_timesteps_weights.to(device=device, dtype=target_latent.dtype)[t_idx].view(B, 1, 1, 1, 1)
    loss = (weights * F.mse_loss(velocity_pred, velocity_target, reduction="none")).mean()
    return loss


# ── Args / scheduler / save ──────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train ViewTransferDiT with online VAE encoding")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_root", type=str, default=None,
                     help="walk <data_root>/<scene>/<location>/")
    src.add_argument("--locations_file", type=str, default=None,
                     help="newline-delimited list of location dirs")
    p.add_argument("--pretrained_dit", type=str, required=True)
    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--model_size", type=str, default="14B", choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--output_dir", type=str, default="./runs/view_transfer")
    p.add_argument("--lora_rank", type=int, default=64, help="0 = full finetune")
    p.add_argument("--lora_alpha", type=float, default=64.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from a checkpoint dir written by accelerator.save_state(). "
                        "Requires the same world size and ZeRO stage as the saved run.")
    p.add_argument("--init_trainable", type=str, default=None,
                   help="Warm-start: load LoRA + new-module weights from a flat .pt file "
                        "(e.g. checkpoint-N/trainable_params.pt). Step counter and optimizer "
                        "state restart from zero. Use this to change world size or LoRA rank.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--gradient_checkpointing", action="store_true")
    # v2 CFG dropout: per-stream + joint. See PLAN.md §10.9.
    p.add_argument("--cfg_drop_prob", type=float, default=0.05,
                   help="Per-stream independent CFG drop probability (v2 default 0.05).")
    p.add_argument("--cfg_joint_drop_prob", type=float, default=0.10,
                   help="Joint-drop probability — zeroes ALL conditioning streams jointly. "
                        "Trains the all-zero unconditional case so inference CFG is in-distribution.")
    p.add_argument("--skip_step0_invariant_check", action="store_true",
                   help="Skip the assert_step0_invariant() check after model init. "
                        "Use only when warm-starting from a non-trivially-trained checkpoint.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed. Omit (or pass empty / negative) to draw a fresh seed from os.urandom; "
                        "the chosen seed is logged so a run can be reproduced.")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--num_video_frames", type=int, default=81)
    p.add_argument("--pers_h", type=int, default=480)
    p.add_argument("--pers_w", type=int, default=832)
    p.add_argument("--same_orientation", action="store_true")
    p.add_argument("--min_overlap", type=float, default=0.25,
                help="minimum first-frame frustum overlap fraction between src and tgt "
                    "perspective trajectories (0 = unconstrained; default 0.25)")
    p.add_argument("--use_mesh", action="store_true",
                   help="Lift the static panorama as a cubemap mesh and rasterize via "
                        "nvdiffrast (with backface culling) instead of point-cloud scatter. "
                        "Off by default; requires `pip install nvdiffrast`.")
    p.add_argument("--train_dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"],
                   help="Dtype used for model weights, batch tensors, loss, and grads. "
                        "We do NOT use Accelerator autocast (Accelerator(mixed_precision='no')); "
                        "this knob is the single source of truth for training precision.")
    p.add_argument("--data_dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"],
                   help="Dtype for the heaviest preprocess stage (VAE encode). equi2pers / "
                        "lift+render stay fp32 (z-buffer needs depth precision). bf16 ~halves "
                        "VAE encode time and is safe for the Wan VAE.")
    p.add_argument("--prefetch_depth", type=int, default=2,
                   help="Number of in-flight batches kept on the side CUDA stream by "
                        "CUDAStreamPrefetcher. depth=1 = original behaviour; depth=2-3 hides "
                        "longer per-batch GPU preprocess (VAE encode + lift+render) behind "
                        "compute when preprocess time > compute time. Each extra slot costs "
                        "~150 MB GPU memory.")
    p.add_argument("--log_video_every", type=int, default=0,
                   help="0 disables; otherwise dump perspective videos + trajectory JSON every N steps")
    p.add_argument("--log_video_fps", type=int, default=24)
    p.add_argument("--profile_steps", type=int, default=0,
                   help="0 disables; otherwise run torch.profiler for this many active steps "
                        "(after 1 wait + 2 warmup) and dump a TB trace under <output_dir>/profiler_trace/")
    p.add_argument("--profile_shapes", action="store_true",
                   help="record op input shapes (heavier, larger trace, but useful for shape-related issues)")
    return p.parse_args()


class _EMA:
    """Exponential moving average for displayed metrics — keeps tqdm postfix readable."""
    __slots__ = ("alpha", "value")
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value: float | None = None
    def update(self, x: float) -> float:
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x
        return self.value


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _write_mp4(video_chw_uint8, path: str, fps: int = 24) -> None:
    """video_chw_uint8: torch / numpy uint8 [C, T, H, W] (C in {1, 3}). Writes RGB MP4.

    Uses imageio (ffmpeg backend) with macro_block_size=4 so non-multiple-of-16 frame
    dimensions don't get silently rescaled.
    """
    import imageio
    import numpy as np
    if isinstance(video_chw_uint8, torch.Tensor):
        video_chw_uint8 = video_chw_uint8.detach().cpu().numpy()
    C, T, H, W = video_chw_uint8.shape
    if C == 1:
        video_chw_uint8 = np.repeat(video_chw_uint8, 3, axis=0)
    frames = np.transpose(video_chw_uint8, (1, 2, 3, 0))  # [T, H, W, 3]
    imageio.mimsave(path, frames, fps=fps, macro_block_size=4, codec="libx264")


def log_training_artifacts(step: int, batch: dict, output_dir: str, fps: int = 24) -> None:
    """Dump perspective videos + visibility mask + trajectory metadata for one step.

    Layout:
        ${output_dir}/logs/videos/step_{step:06d}/
            [b{i}_]videos_src.mp4
            [b{i}_]videos_tgt.mp4   (only if target video was provided)
            [b{i}_]videos_rendered.mp4
            [b{i}_]videos_blob.mp4
            [b{i}_]mask_vis.mp4
            [b{i}_]metadata.json
    """
    import json
    if "videos_src" not in batch:
        return  # logging not enabled in preprocess
    save_dir = os.path.join(output_dir, "logs", "videos", f"step_{step:06d}")
    os.makedirs(save_dir, exist_ok=True)

    B = batch["videos_src"].shape[0]
    video_keys = [k for k in ("videos_src", "videos_tgt", "videos_rendered", "videos_blob", "mask_vis")
                  if k in batch]
    for b in range(B):
        prefix = f"b{b}_" if B > 1 else ""
        for k in video_keys:
            _write_mp4(batch[k][b], os.path.join(save_dir, f"{prefix}{k}.mp4"), fps=fps)

        def _scalar(v):
            return v[b].item() if isinstance(v, torch.Tensor) else v[b]
        def _list(v):
            return v[b].float().cpu().tolist() if isinstance(v, torch.Tensor) else v[b]

        meta = {
            "step": step,
            "location_dir": _scalar(batch["location_dir"]) if "location_dir" in batch else None,
            "src_idx": _scalar(batch["src_idx"]) if "src_idx" in batch else None,
            "tgt_idx": _scalar(batch["tgt_idx"]) if "tgt_idx" in batch else None,
            "t_offset": int(_scalar(batch["t_offset"])) if "t_offset" in batch else None,
            "src_fov_h_deg": float(_scalar(batch["src_fov_h_deg"])),
            "tgt_fov_h_deg": float(_scalar(batch["tgt_fov_h_deg"])),
            "src_yaw_deg": _list(batch["src_yaw_deg"]),
            "src_pitch_deg": _list(batch["src_pitch_deg"]),
            "src_roll_deg": _list(batch["src_roll_deg"]),
            "tgt_yaw_deg": _list(batch["tgt_yaw_deg"]),
            "tgt_pitch_deg": _list(batch["tgt_pitch_deg"]),
            "tgt_roll_deg": _list(batch["tgt_roll_deg"]),
        }
        with open(os.path.join(save_dir, f"{prefix}metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


_TRAINABLE_KEY_PREFIXES = (
    "lora",
    "plucker_encoder",   # matches both plucker_encoder and plucker_encoder_src
    "patch_embed_source",
    "geoada_",           # geoada_patch_embedding + geoada_blocks (incl. before/after_proj)
    "cross_attn_src",
)


def save_trainable(model, save_path, full: bool = False):
    """Write a single .pt for inference.

    full=False (LoRA mode, default): filter to LoRA + new modules only.
    full=True  (full finetune):      dump the entire state_dict. Required when
                                     args.lora_rank == 0, otherwise the main DiT
                                     blocks' trained weights would be silently dropped.

    Under ZeRO-2 model params are NOT partitioned across ranks (only optimizer state is),
    so unwrapping the model and reading state_dict on rank 0 returns full bf16 params.
    Under ZeRO-3 you'd need a `with deepspeed.zero.GatheredParameters(...)` block; we
    only ship the ZeRO-2 path today.
    """
    state = model.state_dict()
    if full:
        out = {k: v.detach().to("cpu") for k, v in state.items()}
    else:
        out = {k: v.detach().to("cpu")
               for k, v in state.items()
               if any(t in k for t in _TRAINABLE_KEY_PREFIXES)}
    torch.save(out, save_path)


def _save_step_marker(save_dir: str, step: int) -> None:
    import json
    with open(os.path.join(save_dir, "step.json"), "w") as f:
        json.dump({"global_step": step}, f)


def _load_step_marker(save_dir: str) -> int:
    import json
    path = os.path.join(save_dir, "step.json")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return int(json.load(f).get("global_step", 0))


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # Default to a random seed so each run shuffles the dataset differently.
    # The chosen seed is printed below so a run can be reproduced via --seed N.
    if args.seed is None or args.seed < 0:
        args.seed = int.from_bytes(os.urandom(4), "little")
    set_seed(args.seed)

    # All distributed config (mixed_precision, deepspeed_plugin, num_processes, …) is
    # routed through accelerate's --config_file YAML, which references our DS JSON.
    # We still cast the input batch to bf16 manually inside the loop because
    # FlashAttention's custom op is not covered by autocast.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    accelerator.init_trackers("tensorboard")
    using_ds = accelerator.state.deepspeed_plugin is not None

    device = accelerator.device
    if accelerator.is_main_process:
        # Direct invocation (without scripts/train.sh) may not have created the dir.
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
        print(f"Random seed = {args.seed}  (use --seed {args.seed} to reproduce)")

    # 1. Load DiT (LoRA or full) ------------------------------------------------
    config = MODEL_CONFIGS[args.model_size]()
    if accelerator.is_main_process:
        print(f"Loading pretrained Wan2.1-T2V-{args.model_size}: {args.pretrained_dit}")
    state_dict = load_dit_state_dict(args.pretrained_dit)
    model = ViewTransferDiT.from_pretrained(state_dict, config)
    del state_dict

    if args.lora_rank > 0:
        model.freeze_base()
        apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        if accelerator.is_main_process:
            print(f"LoRA rank={args.lora_rank}, trainable: {model.trainable_param_count():,}")
    else:
        model.unfreeze_all()
        if accelerator.is_main_process:
            print(f"Full finetune, trainable: {model.trainable_param_count():,}")

    # Warm-start LoRA + new-module weights from a flat checkpoint (e.g. produced by
    # save_trainable on a previous run at a different world size / ZeRO stage).
    if args.init_trainable:
        if accelerator.is_main_process:
            print(f"Loading trainable warm-start: {args.init_trainable}")
        sd = torch.load(args.init_trainable, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if accelerator.is_main_process:
            n_loaded = sum(1 for k in sd if k not in unexpected)
            print(f"  loaded {n_loaded} trainable tensors  "
                  f"(unexpected={len(unexpected)}; missing={len(missing)} — "
                  f"missing should be ~all base/frozen params, fine)")

    # FlashAttention requires fp16/bf16. Under DeepSpeed the bf16 wrapper handles
    # the model dtype after `accelerator.prepare()`; under non-DS we cast manually.
    train_dtype = parse_dtype(args.train_dtype)
    if not using_ds:
        model.to(dtype=train_dtype)

    # 2. Load VAE for online encoding ------------------------------------------
    data_dtype = parse_dtype(args.data_dtype)
    if accelerator.is_main_process:
        print(f"Loading Wan2.1 VAE: {args.vae_ckpt}  (dtype={args.data_dtype})")
    vae = load_wan_vae(args.vae_ckpt, device=device, dtype=data_dtype)

    # 3. Scheduler --------------------------------------------------------------
    scheduler = FlowMatchScheduler(template="Wan")
    scheduler.set_timesteps(num_inference_steps=1000, training=True)

    # 4. Dataset + DataLoader ---------------------------------------------------
    dataset = ViewTransferDataset(
        data_root=args.data_root,
        locations_file=args.locations_file,
        num_video_frames=args.num_video_frames,
        same_orientation=args.same_orientation,
        seed=args.seed,
        min_overlap=args.min_overlap,
    )
    if accelerator.is_main_process:
        print(f"Found {len(dataset)} sample entries.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_view_transfer,
        persistent_workers=(args.num_workers > 0),
    )

    # 5. Optimizer + scheduler --------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if using_ds:
        # The real AdamW + WarmupDecayLR are constructed by DeepSpeed from the JSON
        # (the "auto" placeholders are filled from these CLI flags by accelerate).
        optimizer = DummyOptim(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = DummyScheduler(
            optimizer,
            total_num_steps=args.max_steps,
            warmup_num_steps=args.warmup_steps,
        )
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # v2 step-0 invariant check: verify that with the new architecture and zero-inits,
    # the model would reproduce pretrained Wan exactly at step 0. Only meaningful on a
    # fresh init (no resume / no warm-start), and only worth running on the main process.
    if (
        not args.resume
        and not args.init_trainable
        and not args.skip_step0_invariant_check
        and accelerator.is_main_process
    ):
        try:
            accelerator.unwrap_model(model).assert_step0_invariant()
            print("[init] step-0 invariant OK — model is equivalent to pretrained Wan at step 0.")
        except AssertionError as e:
            print(f"[init] step-0 invariant FAILED: {e}")
            raise

    # Resume from a previously-saved checkpoint dir (ZeRO-sharded).
    if args.resume:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume}")
        accelerator.load_state(args.resume)

    # 6. Prefetcher with online preprocess --------------------------------------
    log_videos = args.log_video_every > 0
    def _preprocess(cpu_batch):
        return gpu_preprocess(
            cpu_batch, vae=vae, device=device,
            pers_h=args.pers_h, pers_w=args.pers_w,
            return_videos=log_videos,
            use_mesh=args.use_mesh,
        )

    # 7. Training loop ----------------------------------------------------------
    global_step = _load_step_marker(args.resume) if args.resume else 0
    if accelerator.is_main_process and global_step:
        print(f"Resuming at global_step = {global_step}")
    model.train()

    # Bottleneck-discovery instrumentation:
    #   t_wait    = wall time the consumer is blocked on the prefetcher
    #               (zero if data pipeline is fully hiding behind compute).
    #   t_compute = wall time of fwd + bwd + optimizer.step (cuda-synced).
    #   t_iter    = total wall time of the iteration.
    # Interpretation:
    #   t_wait ≈ 0           → compute-bound (prefetcher hiding data prep).
    #   t_wait > 0           → data-bound; raw data prep ≈ t_wait + t_compute.
    #   t_wait ≈ t_compute   → balanced; consider --no_prefetch / sub-stage timing.
    is_cuda = device.type == "cuda"
    ema_wait = _EMA()
    ema_compute = _EMA()
    ema_iter = _EMA()
    pbar = tqdm(
        total=args.max_steps, initial=global_step,
        desc=f"train[{args.model_size}]", dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    # ── Optional torch.profiler burst ──────────────────────────────────────────
    # Records a Chrome/TB trace over `profile_steps` active iters after a 1-step wait
    # and 2-step warmup, then stops. View under `${OUTPUT_DIR}/profiler_trace/`
    # in TensorBoard (with `pip install torch_tb_profiler`) or chrome://tracing.
    prof = None
    profile_total_steps = 0
    if args.profile_steps > 0 and accelerator.is_main_process:
        trace_dir = os.path.join(args.output_dir, "profiler_trace")
        os.makedirs(trace_dir, exist_ok=True)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=args.profile_steps, repeat=1),
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            record_shapes=args.profile_shapes,
            with_stack=False,
            profile_memory=False,
        )
        prof.start()
        profile_total_steps = 1 + 2 + args.profile_steps
        tqdm.write(
            f"[profiler] capturing {args.profile_steps} active steps "
            f"(after wait=1, warmup=2) → {trace_dir}"
        )

    while global_step < args.max_steps:
        prefetcher = CUDAStreamPrefetcher(
            dataloader, _preprocess, device=device, depth=args.prefetch_depth,
        )
        iter_p = iter(prefetcher)
        while global_step < args.max_steps:
            iter_t0 = time.perf_counter()

            # ── Time the wait on the prefetcher ────────────────────────────────
            # Sync first so prior consumer work doesn't bleed into this measurement;
            # then sync after next() to flush the producer-side wait_stream.
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with record_function("wait_prefetcher"):
                try:
                    batch = next(iter_p)
                except StopIteration:
                    break
            if is_cuda:
                torch.cuda.synchronize()
            t_wait = time.perf_counter() - t0

            with record_function("cfg_dropout_and_cast"):
                batch = apply_cfg_dropout(
                    batch,
                    per_stream_prob=args.cfg_drop_prob,
                    joint_prob=args.cfg_joint_drop_prob,
                )
                batch = {
                    k: (v.to(train_dtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v)
                    for k, v in batch.items()
                }

            # ── Time fwd + bwd + optim ─────────────────────────────────────────
            t0 = time.perf_counter()
            with accelerator.accumulate(model):
                with record_function("model_fwd"):
                    # Use the wrapped model so DDP's forward (and prepare_for_backward,
                    # which initializes the bucket reducer) runs. Unwrapping here would
                    # silently disable gradient allreduce on multi-GPU.
                    loss = training_step(
                        model,
                        batch,
                        scheduler,
                        gradient_checkpointing=args.gradient_checkpointing,
                    )
                with record_function("model_bwd"):
                    accelerator.backward(loss)
                with record_function("optim_step"):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            if is_cuda:
                torch.cuda.synchronize()
            t_compute = time.perf_counter() - t0

            if prof is not None:
                prof.step()
                profile_total_steps -= 1
                if profile_total_steps <= 0:
                    prof.stop()
                    prof = None
                    tqdm.write(f"[profiler] trace written to {os.path.join(args.output_dir, 'profiler_trace')}")

            global_step += 1
            t_iter = time.perf_counter() - iter_t0
            ema_wait.update(t_wait)
            ema_compute.update(t_compute)
            ema_iter.update(t_iter)

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.1e}",
                wait=f"{ema_wait.value*1000:.0f}ms",
                comp=f"{ema_compute.value*1000:.0f}ms",
                iter=f"{ema_iter.value*1000:.0f}ms",
            )

            if global_step % args.log_every == 0:
                accelerator.log(
                    {
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "time/wait_ms": ema_wait.value * 1000,
                        "time/compute_ms": ema_compute.value * 1000,
                        "time/iter_ms": ema_iter.value * 1000,
                    },
                    step=global_step,
                )

            if (
                args.log_video_every > 0
                and global_step % args.log_video_every == 1
                and accelerator.is_main_process
            ):
                log_training_artifacts(
                    global_step, batch, args.output_dir, fps=args.log_video_fps,
                )
                tqdm.write(f"[step {global_step}] logged training artifacts")

            if global_step % args.save_every == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                # ZeRO-sharded resume state — every rank participates.
                accelerator.save_state(save_dir)
                if accelerator.is_main_process:
                    # Single-file inference checkpoint (LoRA + new modules).
                    save_trainable(
                        accelerator.unwrap_model(model),
                        os.path.join(save_dir, "trainable_params.pt"),
                        full=(args.lora_rank == 0),
                    )
                    _save_step_marker(save_dir, global_step)
                    tqdm.write(f"[step {global_step}] saved checkpoint → {save_dir}")

    pbar.close()
    if prof is not None:
        prof.stop()

    # Free large allocations BEFORE NCCL teardown.
    import gc
    try:
        del prefetcher, iter_p, batch
    except (NameError, UnboundLocalError):
        pass
    del model, optimizer, lr_scheduler, vae
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    accelerator.end_training()
    if accelerator.is_main_process:
        print("Training complete.")


if __name__ == "__main__":
    main()
