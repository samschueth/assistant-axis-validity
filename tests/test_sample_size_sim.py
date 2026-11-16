import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis"))

from sample_size_sim import clustered_noise, scenario_grid, shift_for_auc  # noqa: E402


def test_shift_gives_the_nominal_auc():
    rng = np.random.default_rng(0)
    shift = shift_for_auc(np.array([0.7]))[0]
    drift = rng.standard_normal(4000) - shift
    non_drift = rng.standard_normal(4000)
    auc = (drift[:, None] < non_drift[None, :]).mean()
    assert abs(auc - 0.7) < 0.01


def test_clustered_noise_has_unit_variance_and_the_stated_clustering():
    rng = np.random.default_rng(1)
    x = clustered_noise(rng, n_conv=400, n_turns=15, n_dir=20, icc=0.3, shared=True)
    assert abs(x.var() - 1.0) < 0.05
    conv_means = x.reshape(400, 15, 20).mean(axis=1)
    # variance of a conversation mean = icc + (1 - icc) / n_turns
    assert abs(conv_means.var() - (0.3 + 0.7 / 15)) < 0.05


def test_grid_has_null_and_exchangeable_scenarios():
    grid = scenario_grid()
    assert any(s["target"] == 0.50 for s in grid)
    assert any(s["target"] == "exchangeable" for s in grid)
