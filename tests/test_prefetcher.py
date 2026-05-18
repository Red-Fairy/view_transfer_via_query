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


def test_cpu_heterogeneous_batch_passthrough():
    """CPU path (stream is None): record_stream is never invoked; batches with
    mixed tensor / non-tensor values pass through unchanged and in order."""
    items = [{"x": torch.tensor([float(i)]), "name": f"loc{i}", "k": i,
              "list": [i, i]} for i in range(3)]
    pre = CUDAStreamPrefetcher(
        items, preprocess_fn=lambda b: b, device=torch.device("cpu"),
    )
    out = list(pre)
    assert [b["x"].item() for b in out] == [0.0, 1.0, 2.0]
    assert [b["name"] for b in out] == ["loc0", "loc1", "loc2"]
    assert [b["k"] for b in out] == [0, 1, 2]
    assert [b["list"] for b in out] == [[0, 0], [1, 1], [2, 2]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_record_stream_called_on_cuda_tensors(monkeypatch):
    """Bug #2 fix: every CUDA tensor handed to the consumer must be tagged with
    the consumer (current) stream via record_stream, so the allocator won't
    reuse its side-stream-allocated block until the consumer is done. CPU
    tensors / non-tensors must NOT be recorded. depth=2 to exercise the
    refill-while-consuming path."""
    dev = torch.device("cuda")
    consumer = torch.cuda.current_stream(dev)

    recorded = []  # (data_ptr, stream)
    orig = torch.Tensor.record_stream

    def spy(self, s):
        recorded.append((self.data_ptr(), s))
        return orig(self, s)

    monkeypatch.setattr(torch.Tensor, "record_stream", spy, raising=True)

    # Each batch: one CUDA tensor (must be recorded), one CPU tensor + one
    # non-tensor (must NOT be recorded).
    items = [{"x": torch.tensor([float(i)]), "cpu": torch.tensor([i]),
              "name": f"n{i}"} for i in range(4)]

    def preprocess(b):
        return {"x": b["x"].cuda() * 2.0, "cpu": b["cpu"], "name": b["name"]}

    pre = CUDAStreamPrefetcher(items, preprocess_fn=preprocess, device=dev, depth=2)
    out = list(pre)

    assert [b["x"].item() for b in out] == [0.0, 2.0, 4.0, 6.0]
    # Exactly one record_stream per batch (the CUDA "x"), none for cpu/name.
    assert len(recorded) == 4, f"expected 4 record_stream calls, got {len(recorded)}"
    assert all(s == consumer for _, s in recorded), "recorded on wrong stream"
    cuda_ptrs = {b["x"].data_ptr() for b in out}
    # (data_ptr may be recycled across freed batches; just assert every recorded
    # ptr is a CUDA tensor ptr we saw, and none are the CPU tensors'.)
    cpu_ptrs = {b["cpu"].data_ptr() for b in out}
    assert all(ptr not in cpu_ptrs for ptr, _ in recorded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
