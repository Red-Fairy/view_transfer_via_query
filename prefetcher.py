"""CUDA-stream prefetcher.

Overlaps GPU preprocessing of batch N+1 (equi2pers + VAE encode + plucker)
with the model fwd/bwd of batch N. Workers handle CPU-side disk I/O via
PyTorch's DataLoader; this prefetcher handles the GPU-side preprocessing.

Usage:
    loader = DataLoader(dataset, batch_size=B, num_workers=2, pin_memory=True,
                        collate_fn=collate_view_transfer)
    prefetcher = CUDAStreamPrefetcher(
        loader, preprocess_fn=lambda b: gpu_preprocess(b, vae=vae, device=dev),
        device=dev,
    )
    for batch in prefetcher:
        loss = training_step(model, batch, scheduler)
        loss.backward()
        ...
"""

from __future__ import annotations

import torch
from typing import Callable, Iterator, Optional


class CUDAStreamPrefetcher:
    """Async prefetcher that runs `preprocess_fn` on a side CUDA stream.

    The preprocessing kernels for batch N+1 are launched on `self.stream`
    while the model is still computing on the default stream for batch N.
    Before returning batch N+1, we synchronise the consumer stream against
    `self.stream` so all preprocess output is visible.
    """

    def __init__(
        self,
        loader,
        preprocess_fn: Callable[[dict], dict],
        device: torch.device,
    ):
        self.loader = loader
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._iter: Optional[Iterator] = None
        self._next_batch: Optional[dict] = None

    def _preload(self) -> None:
        try:
            cpu_batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self._next_batch = self.preprocess_fn(cpu_batch)
        else:
            self._next_batch = self.preprocess_fn(cpu_batch)

    def __iter__(self) -> "CUDAStreamPrefetcher":
        self._iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self) -> dict:
        if self._next_batch is None:
            raise StopIteration
        # Make sure all preprocess kernels for this batch finished before consumer reads
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next_batch
        self._preload()
        return batch

    def __len__(self) -> int:
        return len(self.loader)
