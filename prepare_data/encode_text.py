"""T5 text encoding for the prep pipeline.

Loads Wan2.1's UMT5-XXL text encoder once and produces [L, 4096] embeddings.
Supports an empty-prompt path for samples without captions.
"""

import torch
from typing import Iterable, List, Optional, Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from diffsynth.models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer


def load_wan_text_encoder(
    encoder_ckpt: str,
    tokenizer_name_or_path: str = "google/umt5-xxl",
    seq_len: int = 512,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[WanTextEncoder, HuggingfaceTokenizer]:
    """Load WanTextEncoder + tokenizer.

    Args:
        encoder_ckpt:        path to e.g. models_t5_umt5-xxl-enc-bf16.pth
        tokenizer_name_or_path: HF model id or local path to tokenizer files
    """
    encoder = WanTextEncoder()
    if encoder_ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(encoder_ckpt, device="cpu")
    else:
        # Wan T5 .pth uses non-tensor metadata; weights_only=True trips on it.
        # The checkpoint is from a trusted source (Wan-AI HF release).
        sd = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    encoder.load_state_dict(sd, strict=False)
    encoder.eval().requires_grad_(False)
    encoder.to(device=device, dtype=dtype)

    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_name_or_path, seq_len=seq_len, clean="whitespace"
    )
    return encoder, tokenizer


@torch.no_grad()
def encode_text(
    encoder: WanTextEncoder,
    tokenizer: HuggingfaceTokenizer,
    prompt: str,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode one prompt → [L, 4096] CPU tensor (CPU-side bf16/fp32 per encoder dtype).

    For an empty prompt, returns the encoding of the empty string (used for the CFG
    null condition).
    """
    ids, mask = tokenizer(prompt, return_mask=True)
    ids = ids.to(device)
    mask = mask.to(device)
    # Mask the inputs for self-attention; UMT5 uses additive mask
    seq_lens = mask.sum(dim=1).long()
    out = encoder(ids, mask=mask)  # [1, L, 4096]
    # Truncate at the actual prompt length to avoid carrying padded slots
    L_eff = int(seq_lens[0].item())
    return out[0, :L_eff].float().cpu()


@torch.no_grad()
def encode_prompts(
    encoder: WanTextEncoder,
    tokenizer: HuggingfaceTokenizer,
    prompts: Iterable[str],
    device: str = "cuda",
) -> List[torch.Tensor]:
    return [encode_text(encoder, tokenizer, p, device=device) for p in prompts]
