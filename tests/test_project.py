import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from project import collect, directions_by_layer, project_turns  # noqa: E402

HIDDEN, LAYERS, ROLES = 16, [1, 3], 6


@pytest.fixture
def released():
    rng = np.random.default_rng(0)
    return rng.standard_normal((5, HIDDEN)), rng.standard_normal((ROLES, 5, HIDDEN))


def test_directions_are_unit_and_roles_are_orthogonal_to_the_axis(released):
    axis, roles = released
    dirs = directions_by_layer(axis, roles, LAYERS, n_random=7, seed=1)
    assert len(dirs) == len(LAYERS)
    for layer, d in zip(LAYERS, dirs):
        assert np.allclose(d["axis"], axis[layer] / np.linalg.norm(axis[layer]))
        assert d["roles"].shape == (ROLES, HIDDEN) and d["random"].shape == (7, HIDDEN)
        for name in ("roles", "random"):
            assert np.allclose(np.linalg.norm(d[name], axis=1), 1.0)
        assert np.allclose(d["roles"] @ d["axis"], 0.0, atol=1e-9)
    assert np.array_equal(dirs[0]["random"], dirs[1]["random"])  # the same random directions at every layer


def test_projection_uses_each_layer_with_its_own_directions(released):
    axis, roles = released
    dirs = directions_by_layer(axis, roles, LAYERS, n_random=7, seed=1)
    acts = np.zeros((2, len(LAYERS), HIDDEN))
    acts[0, 0] = 3.0 * dirs[0]["axis"]  # turn 1, first layer: three units along that layer's axis
    acts[1, 1] = 2.0 * dirs[1]["roles"][4]  # turn 2, second layer: two units along that layer's role 4
    p = project_turns(acts, dirs)
    assert p["axis"].shape == (2, 2) and p["roles"].shape == (2, 2, ROLES) and p["random"].shape == (2, 2, 7)
    assert np.isclose(p["axis"][0, 0], 3.0) and np.isclose(p["axis"][0, 1], 0.0)
    assert np.isclose(p["roles"][1, 1, 4], 2.0) and np.isclose(p["axis"][1, 1], 0.0, atol=1e-9)
    assert np.allclose(p["roles"][0, 0], 0.0, atol=1e-9)


def test_a_turn_without_activation_is_flagged_and_projects_to_nan(released):
    axis, roles = released
    dirs = directions_by_layer(axis, roles, LAYERS, n_random=3, seed=1)
    convos = [
        {"id": "a", "stratum": "task", "turns": [{"turn": 1}, {"turn": 2}]},
        {"id": "b", "stratum": "meta_reflection", "turns": [{"turn": 1}]},
    ]
    store = {"a": np.ones((2, 2, HIDDEN)), "b": np.ones((1, 2, HIDDEN))}
    store["a"][1] = np.nan
    index, acts = collect(convos, store.__getitem__, n_layers=2, hidden=HIDDEN)
    assert index["conversation"] == ["a", "a", "b"] and index["turn"] == [1, 2, 1]
    assert index["stratum"] == ["task", "task", "meta_reflection"]
    assert index["has_activation"].tolist() == [True, False, True]
    p = project_turns(acts, dirs)
    assert np.isnan(p["axis"][1]).all() and np.isnan(p["roles"][1]).all() and np.isfinite(p["axis"][[0, 2]]).all()


def test_a_file_that_does_not_match_its_conversation_is_an_error():
    convos = [{"id": "a", "stratum": "task", "turns": [{"turn": 1}, {"turn": 2}]}]
    with pytest.raises(ValueError):
        collect(convos, lambda _: np.ones((3, 2, HIDDEN)), n_layers=2, hidden=HIDDEN)
    with pytest.raises(ValueError):
        project_turns(np.ones((1, 3, HIDDEN)), [{}, {}])
