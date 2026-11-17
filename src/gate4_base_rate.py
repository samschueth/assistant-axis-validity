"""Gate 4: does the pilot show enough judged drift, and enough non-drift, in two strata?

Defined in PREREGISTRATION.md section 5.4. Run on the laptop after the pilot has been generated
and judged:

    python src/gate4_base_rate.py

It reads ratings only. No projection is read or computed here: the pilot's activations are not
set against its ratings at any point.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT = REPO_ROOT / "results" / "pilot"
RESULT_PATH = REPO_ROOT / "results" / "gate4_base_rate.json"


def evaluate(ratings: list[dict], strata: list[str], *, lo: float, hi: float, min_strata: int, max_unrated: float) -> dict:
    out: dict = {"strata": {}}
    in_range = 0
    for s in strata:
        rows = [r for r in ratings if r["stratum"] == s]
        rated = [r for r in rows if r["rating"] is not None]
        drift = sum(1 for r in rated if r["rating"] >= 2)
        unrated_share = 1 - len(rated) / len(rows) if rows else 1.0
        evaluable = bool(rows) and unrated_share <= max_unrated
        rate = drift / len(rated) if rated else None
        ok = bool(evaluable and rate is not None and lo <= rate <= hi)
        in_range += ok
        by_conv: dict = {}
        for r in rows:
            c = by_conv.setdefault(r["id"], {"turns": 0, "rated": 0, "drift": 0})
            c["turns"] += 1
            c["rated"] += r["rating"] is not None
            c["drift"] += r["rating"] is not None and r["rating"] >= 2
        out["strata"][s] = {
            "turns": len(rows),
            "rated": len(rated),
            "drift_turns": drift,
            "drift_rate": None if rate is None else round(rate, 4),
            "ratings_0_to_3": [sum(1 for r in rated if r["rating"] == v) for v in range(4)],
            "evaluable": evaluable,
            "in_range": ok,
            "by_conversation": by_conv,
        }
    out["strata_in_range"] = in_range
    out["passed"] = in_range >= min_strata
    return out


def main() -> None:
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        cfg = tomllib.load(f)
    g = cfg["gate4"]
    ratings = [json.loads(line) for line in (PILOT / "ratings.jsonl").read_text().splitlines() if line.strip()]
    result = {
        "gate": "4: base rate of judged drift in the six-conversation pilot",
        "rule": g,
        "judge": json.loads((PILOT / "judge_log.json").read_text()),
        **evaluate(
            ratings,
            cfg["sample"]["strata"],
            lo=g["min_drift_rate"],
            hi=g["max_drift_rate"],
            min_strata=g["min_strata_in_range"],
            max_unrated=g["max_unrated_share"],
        ),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
