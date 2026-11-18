"""Judge validation: the judge's ratings against the human labels (PREREGISTRATION.md section 8).

    python src/judge_validation.py --set main
    python src/judge_validation.py --set main --from-item 41    # section 8's second attempt: the other 80

Reads labels/human_labels_<set>.jsonl and results/<set>/ratings.jsonl. Writes
results/<set>/judge_validation.json (judge_validation.from41.json with --from-item 41).

Reported, as section 8 lists them: raw agreement, Cohen's kappa, Gwet's AC1, the 4x4 table of
ratings and the direction of disagreements. The minimum to proceed is Gwet's AC1 on the drift /
non-drift cut (a rating of 2 or 3 against 0 or 1), at least the value in configs/study.toml.

A sampled turn the judge did not rate is left out of every figure and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label import latest_labels  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def cross_table(human: list[int], judge: list[int], n_categories: int) -> np.ndarray:
    """Counts with the human's rating in rows and the judge's in columns."""
    table = np.zeros((n_categories, n_categories), dtype=int)
    for h, j in zip(human, judge):
        table[h, j] += 1
    return table


def agreement(table: np.ndarray) -> dict:
    """Raw agreement, Cohen's kappa and Gwet's AC1 for two raters, from their cross table."""
    n = table.sum()
    if n == 0:
        return {"n": 0, "raw_agreement": None, "cohen_kappa": None, "gwet_ac1": None}
    p = table / n
    observed = float(np.trace(p))
    rows, cols = p.sum(axis=1), p.sum(axis=0)
    chance_kappa = float(rows @ cols)
    mean_share = (rows + cols) / 2
    chance_ac1 = float((mean_share * (1 - mean_share)).sum() / (len(table) - 1))
    kappa = None if np.isclose(chance_kappa, 1.0) else (observed - chance_kappa) / (1 - chance_kappa)
    return {
        "n": int(n),
        "raw_agreement": observed,
        "cohen_kappa": kappa,
        "gwet_ac1": (observed - chance_ac1) / (1 - chance_ac1),
    }


def validate(pairs: list[tuple[int, int]], min_ac1: float) -> dict:
    """`pairs` are (human rating, judge rating) on the 0-3 scale, for turns both have rated."""
    human, judge = [h for h, _ in pairs], [j for _, j in pairs]
    four = cross_table(human, judge, 4)
    cut = cross_table([int(h >= 2) for h in human], [int(j >= 2) for j in judge], 2)
    drift = agreement(cut)
    return {
        "drift_cut": drift
        | {
            "table_rows_human_cols_judge": {"order": ["not drift", "drift"], "counts": cut.tolist()},
            "human_drift_judge_not": int(cut[1, 0]),
            "judge_drift_human_not": int(cut[0, 1]),
        },
        "four_point": agreement(four)
        | {
            "table_rows_human_cols_judge": {"order": [0, 1, 2, 3], "counts": four.tolist()},
            "judge_higher": int(np.triu(four, 1).sum()),
            "judge_lower": int(np.tril(four, -1).sum()),
        },
        "min_ac1": min_ac1,
        "passes": drift["gwet_ac1"] is not None and drift["gwet_ac1"] >= min_ac1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True)
    parser.add_argument("--from-item", type=int, default=1)
    args = parser.parse_args()
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        min_ac1 = tomllib.load(f)["judge_validation"]["min_ac1"]

    sample = json.loads((REPO_ROOT / "labels" / f"sample_{args.set}.json").read_text())["items"]
    labels = latest_labels((REPO_ROOT / "labels" / f"human_labels_{args.set}.jsonl").read_text().splitlines())
    ratings = {
        (r["id"], r["turn"]): r
        for r in map(json.loads, (REPO_ROOT / "results" / args.set / "ratings.jsonl").read_text().splitlines())
    }
    items = [i for i in sample if i["item"] >= args.from_item]
    unlabelled = [i["item"] for i in items if i["item"] not in labels]
    unrated = [i["item"] for i in items if ratings.get((i["id"], i["turn"]), {}).get("rating") is None]
    used = [i for i in items if i["item"] not in unlabelled and i["item"] not in unrated]
    pair = lambda i: (labels[i["item"]]["rating"], ratings[(i["id"], i["turn"])]["rating"])  # noqa: E731

    out = {
        "set": args.set,
        "items": f"{args.from_item} to {len(sample)}",
        "sampled": len(items),
        "not_labelled": unlabelled,
        "not_rated_by_judge": unrated,
        **validate([pair(i) for i in used], min_ac1),
        "by_stratum_drift_cut": {
            s: validate([pair(i) for i in used if i["stratum"] == s], min_ac1)["drift_cut"]
            for s in dict.fromkeys(i["stratum"] for i in sample)
        },
    }
    name = "judge_validation.json" if args.from_item == 1 else f"judge_validation.from{args.from_item}.json"
    (REPO_ROOT / "results" / args.set / name).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
