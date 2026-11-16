import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_axis import load_config, random_directions, role_directions, unit  # noqa: E402


def test_config_has_the_fixed_layer_and_counts():
    cfg = load_config()
    assert cfg["projection"]["target_layer"] == 32
    assert cfg["vectors"]["n_role_vectors"] == 275
    assert cfg["model"]["thinking_mode"] is False


def test_role_directions_are_unit_and_orthogonal_to_axis():
    rng = np.random.default_rng(0)
    roles = rng.standard_normal((20, 64))
    axis = rng.standard_normal(64)
    d = role_directions(roles, axis)
    assert d.shape == (20, 64)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(d @ unit(axis), 0.0, atol=1e-10)


def test_random_directions_are_unit_and_seeded():
    a = random_directions(5, 32, seed=1)
    b = random_directions(5, 32, seed=1)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, b)
