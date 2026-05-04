"""Agent-mask utility — NOT the model's mask channel.

The model's `mask_packed` input is the **visibility mask** from the lift-and-render
step (which pixels of the warped panorama are valid vs unknown). That mask is
produced by the user's offline lift-and-render pipeline, not here.

This file produces an *agent-detection* mask (per-pixel difference between dynamic
and static panoramas) which can be useful as a hint to the user's blob-video
generator — it tells you where moving agents are in equirect space at each frame.
"""

import torch
import torch.nn.functional as F


def compute_agent_mask_360(
    rgb_dynamic: torch.Tensor,    # [T, 3, H, W] float in [0, 1]
    rgb_static: torch.Tensor,     # [T, 3, H, W] float in [0, 1]
    threshold: float = 0.06,
    dilate_iters: int = 2,
) -> torch.Tensor:
    """Per-pixel binary agent mask in equirect space.

    Args:
        rgb_dynamic, rgb_static: same temporal length and resolution.
        threshold: mean per-pixel L1 difference (over 3 channels) above which a pixel
                   is considered "moving". Range ~0.04–0.10 works for UE renders.
        dilate_iters: number of 3x3 max-pool dilations to grow the mask, covering
                     anti-aliasing edges and shadows.

    Returns:
        mask: [T, 1, H, W] float in {0, 1}
    """
    assert rgb_dynamic.shape == rgb_static.shape, (
        f"shape mismatch: {rgb_dynamic.shape} vs {rgb_static.shape}"
    )
    diff = (rgb_dynamic - rgb_static).abs().mean(dim=1, keepdim=True)  # [T, 1, H, W]
    mask = (diff > threshold).float()
    if dilate_iters > 0:
        for _ in range(dilate_iters):
            mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return mask


def project_mask_to_perspective(
    mask_360: torch.Tensor,        # [T, 1, He, We]
    R_crop_cv: torch.Tensor,       # [T, 3, 3]
    fov_h_deg: float,
    pers_h: int = 480,
    pers_w: int = 832,
) -> torch.Tensor:
    """Apply equi2pers projection to a 1-channel mask. Returns [T, 1, pers_h, pers_w]."""
    # Reuse equi_to_perspective_video by repeating to 3 channels
    from .extract_perspectives import equi_to_perspective_video
    mask3 = mask_360.expand(-1, 3, -1, -1)
    pers3 = equi_to_perspective_video(
        mask3, R_crop_cv, fov_h_deg=fov_h_deg, pers_h=pers_h, pers_w=pers_w
    )
    return pers3[:, :1]
