"""Load the released Assistant Axis, default vector and role vectors.

The files are the authors' release (see DATA_AND_MODELS.md). Nothing is recomputed here.
They are downloaded at a pinned revision into a directory that is not committed, and are
loaded with weights-only loading because downloaded tensor files are untrusted.

    python src/load_axis.py --fetch     # download
    python src/load_axis.py             # print a summary of what is on disk
"""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "study.toml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def vectors_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg["vectors"]["local_dir"] / cfg["vectors"]["subdir"]


def fetch(cfg: dict) -> Path:
    """Download the axis, the default vector and the role vectors at the pinned revision."""
    from huggingface_hub import snapshot_download

    v = cfg["vectors"]
    sub = v["subdir"]
    snapshot_download(
        repo_id=v["repo_id"],
        repo_type=v["repo_type"],
        revision=v["revision"],
        allow_patterns=[
            f"{sub}/assistant_axis.pt",
            f"{sub}/default_vector.pt",
            f"{sub}/role_vectors/*.pt",
        ],
        local_dir=REPO_ROOT / v["local_dir"],
    )
    return vectors_dir(cfg)


def _load_tensor(path: Path) -> np.ndarray:
    """Load one released file as a float64 array of shape (n_layers, hidden_size)."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        tensors = [t for t in obj.values() if isinstance(t, torch.Tensor)]
        if len(tensors) != 1:
            raise ValueError(f"{path.name}: expected one tensor, found {len(tensors)}")
        obj = tensors[0]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{path.name}: expected a tensor, got {type(obj).__name__}")
    return obj.to(torch.float64).numpy()


@dataclass(frozen=True)
class ReleasedVectors:
    axis: np.ndarray  # (n_layers, hidden_size)
    default: np.ndarray  # (n_layers, hidden_size)
    roles: np.ndarray  # (n_roles, n_layers, hidden_size)
    role_names: tuple[str, ...]


def load_released(cfg: dict | None = None) -> ReleasedVectors:
    cfg = cfg or load_config()
    d = vectors_dir(cfg)
    if not d.exists():
        raise FileNotFoundError(f"{d} not found: run `make fetch-vectors` first")
    role_paths = sorted((d / "role_vectors").glob("*.pt"))
    released = ReleasedVectors(
        axis=_load_tensor(d / "assistant_axis.pt"),
        default=_load_tensor(d / "default_vector.pt"),
        roles=np.stack([_load_tensor(p) for p in role_paths]),
        role_names=tuple(p.stem for p in role_paths),
    )
    check_integrity(released, cfg)
    return released


def check_integrity(r: ReleasedVectors, cfg: dict) -> None:
    """Fail loudly if the files are not what the configuration says they are."""
    shape = (cfg["model"]["total_layers"], cfg["model"]["hidden_size"])
    n = cfg["vectors"]["n_role_vectors"]
    if r.axis.shape != shape:
        raise ValueError(f"axis shape {r.axis.shape}, expected {shape}")
    if r.default.shape != shape:
        raise ValueError(f"default vector shape {r.default.shape}, expected {shape}")
    if r.roles.shape != (n, *shape):
        raise ValueError(f"role vectors shape {r.roles.shape}, expected {(n, *shape)}")
    for name, a in (("axis", r.axis), ("default", r.default), ("roles", r.roles)):
        if not np.isfinite(a).all():
            raise ValueError(f"{name}: non-finite values")


def unit(v: np.ndarray) -> np.ndarray:
    """Scale to unit length along the last dimension."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def role_directions(roles_at_layer: np.ndarray, axis_at_layer: np.ndarray) -> np.ndarray:
    """Role-specific off-target directions at one layer.

    Each role vector has the mean role vector subtracted and its component along the axis
    removed, and is then scaled to unit length. Shape (n_roles, hidden_size).
    """
    a = unit(axis_at_layer)
    centred = roles_at_layer - roles_at_layer.mean(axis=0, keepdims=True)
    residual = centred - np.outer(centred @ a, a)
    return unit(residual)


def random_directions(n: int, hidden_size: int, seed: int) -> np.ndarray:
    """Random unit directions, uniform on the sphere. Shape (n, hidden_size)."""
    rng = np.random.default_rng(seed)
    return unit(rng.standard_normal((n, hidden_size)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="download the released files")
    args = parser.parse_args()
    cfg = load_config()
    if args.fetch:
        print(f"downloaded to {fetch(cfg).relative_to(REPO_ROOT)}")
    r = load_released(cfg)
    print(f"axis {r.axis.shape}, default {r.default.shape}, roles {r.roles.shape}")


if __name__ == "__main__":
    main()
