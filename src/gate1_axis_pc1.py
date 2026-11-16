"""Gate 1: the released axis against the first principal component of the released role vectors.

The procedure and the pass threshold are fixed in PREREGISTRATION.md section 5.1 and were
committed before this script was first run. One variant only.

    python src/gate1_axis_pc1.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from load_axis import REPO_ROOT, load_config, load_released

RESULT_PATH = REPO_ROOT / "results" / "gate1_axis_pc1.json"
PAPER_MIDDLE_LAYER_FIGURE = 0.71  # reported, not gating


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def first_component(rows: np.ndarray) -> tuple[np.ndarray, float]:
    """First principal component of mean-centred rows, and its share of variance."""
    centred = rows - rows.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0], float(s[0] ** 2 / np.sum(s**2))


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main() -> None:
    cfg = load_config()
    layer = cfg["projection"]["target_layer"]
    threshold = cfg["gate1"]["min_abs_cosine"]
    released = load_released(cfg)  # raises if the integrity checks fail

    n_layers = released.axis.shape[0]
    per_layer = []
    for i in range(n_layers):
        pc1, share = first_component(released.roles[:, i, :])
        per_layer.append((abs(cosine(released.axis[i], pc1)), share))

    abs_cos, share = per_layer[layer]
    contrast = released.default[layer] - released.roles[:, layer, :].mean(axis=0)

    result = {
        "gate": "1: released axis vs first principal component of released role vectors",
        "git": git_state(),
        "vectors_revision": cfg["vectors"]["revision"],
        "n_role_vectors": int(released.roles.shape[0]),
        "tensor_shape": list(released.axis.shape),
        "layer": layer,
        "threshold_min_abs_cosine": threshold,
        "abs_cosine_axis_pc1": round(abs_cos, 4),
        "passed": bool(abs_cos >= threshold),
        "reported_not_gating": {
            "clears_paper_middle_layer_figure_0.71": bool(abs_cos >= PAPER_MIDDLE_LAYER_FIGURE),
            "pc1_variance_share_at_layer": round(share, 4),
            "cosine_axis_vs_default_minus_mean_role": round(
                cosine(released.axis[layer], contrast), 4
            ),
            "abs_cosine_min_over_layers": round(min(c for c, _ in per_layer), 4),
            "abs_cosine_by_layer": [round(c, 4) for c, _ in per_layer],
        },
    }
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "reported_not_gating"}, indent=2))
    print(f"wrote {RESULT_PATH.relative_to(REPO_ROOT)}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
