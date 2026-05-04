"""End-to-end smoke test on a real UE location.

Runs one training step (online VAE encode + lift+render + plücker + forward + backward)
and one inference step (sampling loop with CFG + VAE decode), using the smaller
Wan2.1-T2V-1.3B backbone for fast turnaround. Verifies the entire pipeline is wired
correctly on real data.

Usage:
    python -m view_transfer_via_query.tests.smoke_test \\
        --data_root /share/.../outputs_non_arranged_cars_v2 \\
        --dit_ckpt /home/.../Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \\
        --vae_ckpt /home/.../Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
"""

from __future__ import annotations

import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from safetensors.torch import load_file as load_safetensors

from view_transfer_via_query.model import (
    MODEL_CONFIGS, ViewTransferDiT, apply_lora,
)
from view_transfer_via_query.dataset import (
    ViewTransferDataset, collate_view_transfer,
)
from view_transfer_via_query.gpu_preprocess import gpu_preprocess
from view_transfer_via_query.train import apply_cfg_dropout, training_step
from view_transfer_via_query.pipeline import ViewTransferPipeline
from view_transfer_via_query.prepare_data.encode_latents import load_wan_vae


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="parent dir whose subdirs are scenes; we discover locations underneath")
    p.add_argument("--dit_ckpt", required=True,
                   help="Wan2.1-T2V-1.3B safetensors")
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--pers_h", type=int, default=128)
    p.add_argument("--pers_w", type=int, default=224)
    p.add_argument("--num_video_frames", type=int, default=17)
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=2)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_dit_state_dict(ckpt_path: str) -> dict:
    if ckpt_path.endswith(".safetensors"):
        return load_safetensors(ckpt_path)
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    section("1. Build model from Wan2.1-T2V-1.3B")
    state_dict = load_dit_state_dict(args.dit_ckpt)
    print(f"  DiT state dict: {len(state_dict)} keys")
    model = ViewTransferDiT.from_pretrained(state_dict, MODEL_CONFIGS["1.3B"]())
    del state_dict
    model.freeze_base()
    apply_lora(model, rank=args.lora_rank, alpha=float(args.lora_rank))
    print(f"  Trainable params: {model.trainable_param_count():,}")
    model.to(device=device, dtype=torch.bfloat16)

    section("2. Load VAE")
    vae = load_wan_vae(args.vae_ckpt, device=str(device), dtype=torch.float32)
    print(f"  VAE loaded ({sum(p.numel() for p in vae.model.parameters()):,} params)")

    section("3. Build dataset + dataloader")
    dataset = ViewTransferDataset(
        data_root=args.data_root,
        num_video_frames=args.num_video_frames,
        seed=0,
    )
    print(f"  Dataset entries: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        pin_memory=True, collate_fn=collate_view_transfer,
    )

    section("4. Pull one batch (CPU side)")
    t0 = time.time()
    cpu_batch = next(iter(loader))
    print(f"  Batch loaded in {time.time()-t0:.2f}s")
    print(f"  rgb_src_360: {tuple(cpu_batch['rgb_src_360'].shape)}")
    print(f"  static_depth_t0 range: [{cpu_batch['static_depth_t0'].min():.2f},"
          f" {cpu_batch['static_depth_t0'].max():.2f}]  (meters)")
    print(f"  pano_c2w_src[0,0]:\n{cpu_batch['pano_c2w_src'][0, 0]}")

    section("5. GPU preprocess (online: equi2pers + lift+render + VAE + plücker + mask)")
    t0 = time.time()
    cond = gpu_preprocess(
        cpu_batch, vae=vae, device=device,
        pers_h=args.pers_h, pers_w=args.pers_w,
    )
    torch.cuda.synchronize()
    print(f"  Preprocess: {time.time()-t0:.2f}s")
    for k, v in cond.items():
        print(f"    {k}: {tuple(v.shape)}  dtype={v.dtype}")

    section("6. Training step (forward + backward)")
    scheduler = FlowMatchScheduler(template="Wan")
    scheduler.set_timesteps(num_inference_steps=1000, training=True)

    # Cast model inputs to BF16 to match the model's parameter dtype
    cond_train = {k: v.clone() for k, v in cond.items()}
    cond_train = apply_cfg_dropout(cond_train, drop_prob=0.1)
    cond_train = {
        k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
        for k, v in cond_train.items()
    }

    model.train()
    t0 = time.time()
    loss = training_step(model, cond_train, scheduler, gradient_checkpointing=True)
    print(f"  Forward: {time.time()-t0:.2f}s, loss={loss.item():.4f}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    print(f"  Backward: {time.time()-t0:.2f}s")

    # Verify LoRA grads exist
    grads_with_data = sum(
        1 for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"  Trainable params with non-zero grad: {grads_with_data}")
    assert grads_with_data > 0

    section("7. Inference: 2-step sampling + VAE decode")
    model.eval()
    pipe = ViewTransferPipeline(
        model=model, vae=vae, scheduler=scheduler, device=device,
        pers_h=args.pers_h, pers_w=args.pers_w,
    )
    cpu_batch_no_target = {k: v for k, v in cpu_batch.items() if k != "rgb_tgt_360"}
    t0 = time.time()
    videos = pipe.generate(
        cpu_batch_no_target,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=2.0,
    )
    torch.cuda.synchronize()
    print(f"  Generate: {time.time()-t0:.2f}s")
    print(f"  Output video: {tuple(videos.shape)}, dtype={videos.dtype}, "
          f"range=[{videos.min().item()}, {videos.max().item()}]")

    # Wan VAE is causal: T_lat = (T_video + 3) // 4, so decode of 5 latents → 17 frames
    expected_T = args.num_video_frames
    assert videos.shape == (1, 3, expected_T, args.pers_h, args.pers_w), \
        f"Unexpected shape: {videos.shape}, expected (1, 3, {expected_T}, {args.pers_h}, {args.pers_w})"

    section("ALL SMOKE TEST STEPS PASSED")


if __name__ == "__main__":
    main()
