"""Project every turn's saved activation onto the axis and the off-target directions.

    python src/project.py --set main

Runs on the laptop. Reads results/<set>/conversations.jsonl (the turn order) and
data/runs/<set>/<conversation>.npy (the mean response-token activation of each turn at each
captured layer, written by generate.py). Writes one file, results/<set>/projections.npz, and a
short log, results/<set>/projection_log.json.

For each captured layer (PREREGISTRATION.md sections 4.2 and 4.3):

    axis     the released Assistant Axis at that layer, scaled to unit length
    roles    the 275 released role vectors at that layer, each with the mean of the 275
             subtracted and its component along the axis removed, scaled to unit length
    random   1,000 random unit directions, from the seed in configs/study.toml; the same
             1,000 at every layer

A projection is the dot product of the turn's activation with the direction. A turn with no
activation (an empty reply, or a failed capture) gets NaN everywhere and has_activation false.

No rating is read here and nothing is compared with anything: this script produces numbers
for the analysis and makes no claim.

Arrays in projections.npz, one row per turn, in the order of conversations.jsonl:

    conversation (n,) str     turn (n,) int, from 1     stratum (n,) str
    has_activation (n,) bool  layers (L,) int           role_names (275,) str
    axis (n, L) float32       roles (n, L, 275) float32 random (n, L, 1000) float32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_axis import load_config, load_released, random_directions, role_directions, unit  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def directions_by_layer(axis: np.ndarray, roles: np.ndarray, layers: list[int], n_random: int, seed: int) -> list[dict]:
    """For each layer: the unit axis (hidden,), role directions (n_roles, hidden), random (n_random, hidden)."""
    rand = random_directions(n_random, axis.shape[-1], seed)
    return [
        {"axis": unit(axis[layer]), "roles": role_directions(roles[:, layer], axis[layer]), "random": rand}
        for layer in layers
    ]


def project_turns(activations: np.ndarray, directions: list[dict]) -> dict:
    """activations is (n_turns, n_layers, hidden). Returns axis (n, L), roles (n, L, R), random (n, L, K)."""
    acts = np.asarray(activations, dtype=np.float64)
    if acts.shape[1] != len(directions):
        raise ValueError(f"{acts.shape[1]} layers in the activations, {len(directions)} sets of directions")
    out = {}
    for name in ("axis", "roles", "random"):
        per_layer = [acts[:, i] @ np.atleast_2d(d[name]).T for i, d in enumerate(directions)]
        stacked = np.stack(per_layer, axis=1)  # (n, L, n_directions)
        out[name] = stacked[:, :, 0] if name == "axis" else stacked
    return out


def collect(conversations: list[dict], load_activation, n_layers: int, hidden: int) -> tuple[dict, np.ndarray]:
    """The turn index and the stacked activations. The file must have one row per recorded turn."""
    index: dict = {"conversation": [], "turn": [], "stratum": []}
    blocks = []
    for convo in conversations:
        n = len(convo["turns"])
        act = np.asarray(load_activation(convo["id"]))
        if n == 0:
            continue
        if act.shape != (n, n_layers, hidden):
            raise ValueError(f"{convo['id']}: activations {act.shape}, expected {(n, n_layers, hidden)}")
        blocks.append(act)
        for k, t in enumerate(convo["turns"], start=1):
            if t.get("turn", k) != k:
                raise ValueError(f"{convo['id']}: turn numbered {t['turn']} at position {k}")
            index["conversation"].append(convo["id"])
            index["turn"].append(k)
            index["stratum"].append(convo["stratum"])
    acts = np.concatenate(blocks) if blocks else np.zeros((0, n_layers, hidden))
    index["has_activation"] = np.isfinite(acts).all(axis=(1, 2))
    return index, acts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True)
    args = parser.parse_args()

    cfg = load_config()
    layers = sorted([cfg["projection"]["target_layer"], *cfg["projection"]["reliability_layers"]])
    released = load_released(cfg)
    directions = directions_by_layer(
        released.axis, released.roles, layers, cfg["off_target"]["n_random_directions"], cfg["seeds"]["random_directions"]
    )

    out_dir = REPO_ROOT / "results" / args.set
    act_dir = REPO_ROOT / "data" / "runs" / args.set
    conversations = [
        json.loads(line) for line in (out_dir / "conversations.jsonl").read_text().splitlines() if line.strip()
    ]
    index, acts = collect(
        conversations, lambda cid: np.load(act_dir / f"{cid}.npy"), len(layers), cfg["model"]["hidden_size"]
    )
    proj = project_turns(acts, directions)

    np.savez_compressed(
        out_dir / "projections.npz",
        conversation=np.array(index["conversation"]),
        turn=np.array(index["turn"], dtype=np.int32),
        stratum=np.array(index["stratum"]),
        has_activation=index["has_activation"],
        layers=np.array(layers, dtype=np.int32),
        role_names=np.array(released.role_names),
        **{k: v.astype(np.float32) for k, v in proj.items()},
    )
    missing = [
        f"{c} turn {t}" for c, t, ok in zip(index["conversation"], index["turn"], index["has_activation"]) if not ok
    ]
    log = {
        "set": args.set,
        "conversations": len(conversations),
        "turns": len(index["turn"]),
        "turns_without_activation": missing,
        "turns_by_stratum": {s: index["stratum"].count(s) for s in cfg["sample"]["strata"]},
        "layers": layers,
        "target_layer": cfg["projection"]["target_layer"],
        "role_directions": len(released.role_names),
        "random_directions": cfg["off_target"]["n_random_directions"],
        "random_seed": cfg["seeds"]["random_directions"],
        "vectors_revision": cfg["vectors"]["revision"],
    }
    (out_dir / "projection_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
