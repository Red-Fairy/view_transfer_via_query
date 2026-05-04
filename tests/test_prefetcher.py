"""Unit tests for CUDAStreamPrefetcher."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pytest

from view_transfer_via_query.prefetcher import CUDAStreamPrefetcher


def _make_loader(n: int):
    """Yields n trivial dicts."""
    return [{"x": torch.tensor([i], dtype=torch.float32)} for i in range(n)]


def test_iterates_all_items_cpu():
    """On CPU (no CUDA stream), prefetcher should iterate all items in order."""
    items = _make_loader(5)
    pre = CUDAStreamPrefetcher(
        items,
        preprocess_fn=lambda b: {"x": b["x"] * 2.0},
        device=torch.device("cpu"),
    )
    out = [b["x"].item() for b in pre]
    assert out == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_empty_loader():
    pre = CUDAStreamPrefetcher(
        [], preprocess_fn=lambda b: b, device=torch.device("cpu")
    )
    out = list(pre)
    assert out == []


def test_preprocess_called_per_batch():
    calls = []

    def preprocess(b):
        calls.append(int(b["x"].item()))
        return b

    items = _make_loader(3)
    pre = CUDAStreamPrefetcher(items, preprocess_fn=preprocess, device=torch.device("cpu"))
    list(pre)
    assert calls == [0, 1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_iterates_on_cuda_stream():
    """On CUDA, prefetcher should still produce all items in order."""
    items = _make_loader(4)

    def preprocess(b):
        return {"x": b["x"].cuda() * 3.0}

    pre = CUDAStreamPrefetcher(items, preprocess_fn=preprocess, device=torch.device("cuda"))
    out = [b["x"].item() for b in pre]
    assert out == [0.0, 3.0, 6.0, 9.0]


def test_len_passthrough():
    pre = CUDAStreamPrefetcher(
        _make_loader(7), preprocess_fn=lambda b: b, device=torch.device("cpu")
    )
    assert len(pre) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
