"""pytest auto-loaded config.

Ensures DIFFSYNTH_ROOT (the parent of `view_transfer_via_query/`) is on sys.path
so both `view_transfer_via_query.X` and `diffsynth.X` imports resolve regardless
of the CWD pytest was invoked from. Lets `pytest tests/` work from inside
`view_transfer_via_query/` without per-file `sys.path.insert(...)` boilerplate.

Also disables FlashAttention's CUDA-only kernels so model tests can run on CPU.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))   # view_transfer_via_query/
DIFFSYNTH_ROOT = os.path.dirname(PROJECT_ROOT)               # DiffSynth-Studio/

if DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)

# Force CPU-compatible attention path so the test suite passes on machines without
# CUDA / FlashAttention. Tests that explicitly want GPU should opt in themselves.
try:
    import diffsynth.models.wan_video_dit as _wvd
    _wvd.FLASH_ATTN_3_AVAILABLE = False
    _wvd.FLASH_ATTN_2_AVAILABLE = False
    _wvd.SAGE_ATTN_AVAILABLE = False
except Exception:
    # If diffsynth isn't importable yet (path not set up), tests will surface their own errors.
    pass
