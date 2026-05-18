"""CUDA-stream prefetcher.

Overlaps GPU preprocessing of batch N+1 (equi2pers + VAE encode + plucker)
with the model fwd/bwd of batch N. Workers handle CPU-side disk I/O via
PyTorch's DataLoader; this prefetcher handles the GPU-side preprocessing.

`depth` controls how many batches are kept in flight on the side CUDA stream
at any time. depth=1 (the original behaviour) hides exactly one preprocess pass
behind the model step — fine when preprocess time ≤ compute time. When preprocess
time exceeds compute time, raise depth (typically 2-3) so several batches are
queued ahead and the consumer never has to wait. Each extra slot costs a few
hundred MB of GPU memory (the encoded conditioning dict; the heavy VAE
intermediate buffers are released at the end of each preprocess so they don't
accumulate across slots).

Usage:
    loader = DataLoader(dataset, batch_size=B, num_workers=2, pin_memory=True,
                        collate_fn=collate_view_transfer)
    prefetcher = CUDAStreamPrefetcher(
        loader, preprocess_fn=lambda b: gpu_preprocess(b, vae=vae, device=dev),
        device=dev, depth=2,
    )
    for batch in prefetcher:
        loss = training_step(model, batch, scheduler)
        loss.backward()
        ...
"""

from __future__ import annotations

import torch
from collections import deque
from typing import Callable, Iterator, Optional


class CUDAStreamPrefetcher:
    """Async prefetcher that runs `preprocess_fn` on a side CUDA stream.

    The preprocessing kernels for the next `depth` batches are queued on
    `self.stream` while the model is still computing on the default stream
    for the current batch. Before returning each batch, we synchronise the
    consumer stream against `self.stream` so all preprocess output is visible.

    Args:
        loader: any iterable of CPU batches (DataLoader).
        preprocess_fn: function (cpu_batch -> gpu_batch). Runs under the side stream.
        device: target CUDA device (or CPU; falls back to synchronous on CPU).
        depth: how many batches to keep in flight on the side stream. Default 2.
    """

    def __init__(
        self,
        loader,
        preprocess_fn: Callable[[dict], dict],
        device: torch.device,
        depth: int = 2,
    ):
        assert depth >= 1, f"depth must be >= 1, got {depth}"
        self.loader = loader
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.depth = depth
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._iter: Optional[Iterator] = None
        self._queue: deque = deque()
        self._exhausted = False

    def _try_fill(self) -> None:
        """Top up the queue until full or the underlying loader is exhausted."""
        while not self._exhausted and len(self._queue) < self.depth:
            try:
                cpu_batch = next(self._iter)
            except StopIteration:
                self._exhausted = True
                return
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    self._queue.append(self.preprocess_fn(cpu_batch))
            else:
                self._queue.append(self.preprocess_fn(cpu_batch))

    def __iter__(self) -> "CUDAStreamPrefetcher":
        self._iter = iter(self.loader)
        self._exhausted = False
        self._queue.clear()
        self._try_fill()
        return self

    def _record_consumer_stream(self, batch: dict) -> None:
        """Tag every CUDA tensor in `batch` with the consumer (current) stream.

        The batch tensors were allocated on `self.stream` (the side/preprocess
        stream) but are *read* on the consumer (default) stream. PyTorch's
        caching allocator only tracks a block's *allocation* stream: when the
        consumer later drops its references, the allocator defers block reuse
        only until the SIDE stream is clear — it has no record that the
        consumer stream touched the memory. So the very next `_try_fill()`
        can hand that block to a new preprocess while the model's fwd/bwd is
        still reading it → intermittent, input-dependent silent corruption.

        `record_stream(consumer)` tells the allocator the consumer stream also
        used these blocks, so it will additionally wait for the consumer
        stream to drain before reusing them. This is the standard NVIDIA
        data-prefetcher pattern; `wait_stream` (below) only orders
        producer→consumer and does not cover this consumer→producer reuse.
        """
        consumer = torch.cuda.current_stream(self.device)
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                v.record_stream(consumer)

    def __next__(self) -> dict:
        if not self._queue:
            raise StopIteration
        # Conservative sync: wait until everything queued on the side stream is done.
        # Cheaper per-batch sync would use a CUDA event per slot, but for depth ≤ 3
        # the side stream is rarely backed up enough to matter.
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._queue.popleft()
        # Memory-safety: mark the batch as consumed on the default stream BEFORE
        # the next `_try_fill()` can reuse its (side-stream-allocated) blocks.
        if self.stream is not None:
            self._record_consumer_stream(batch)
        self._try_fill()                     # refill immediately while consumer runs
        return batch

    def __len__(self) -> int:
        return len(self.loader)
