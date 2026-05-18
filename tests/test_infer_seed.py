"""make_location_rng: seeded draws must be reproducible & comparable across runs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from view_transfer_via_query.infer import make_location_rng

LOC_A = "/share/.../outputs/SceneA/x-1_y-2_s600"
LOC_B = "/share/.../outputs/SceneB/x-9_y-9_s600"


def _draw(rng):
    return (int(rng.integers(0, 1000)), rng.random(), rng.random())


def test_same_seed_same_location_is_reproducible():
    # Independent "runs" (fresh generator each time) → identical stream.
    assert _draw(make_location_rng(0, LOC_A)) == _draw(make_location_rng(0, LOC_A))


def test_different_seed_differs():
    assert _draw(make_location_rng(0, LOC_A)) != _draw(make_location_rng(1, LOC_A))


def test_different_location_differs():
    # Distinct locations must not collide under the same seed.
    assert _draw(make_location_rng(0, LOC_A)) != _draw(make_location_rng(0, LOC_B))


def test_seed_none_is_nondeterministic():
    # No seed → fresh entropy each call (the original behaviour).
    assert _draw(make_location_rng(None, LOC_A)) != _draw(make_location_rng(None, LOC_A))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
