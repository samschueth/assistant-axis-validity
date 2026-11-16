import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gate1_axis_pc1 import cosine, first_component  # noqa: E402


def test_first_component_recovers_a_planted_direction():
    rng = np.random.default_rng(0)
    planted = rng.standard_normal(50)
    planted /= np.linalg.norm(planted)
    rows = np.outer(rng.standard_normal(200) * 5.0, planted) + rng.standard_normal((200, 50)) * 0.1
    rows += 3.0  # an offset the centring must remove
    pc1, share = first_component(rows)
    assert abs(cosine(pc1, planted)) > 0.99
    assert share > 0.9
