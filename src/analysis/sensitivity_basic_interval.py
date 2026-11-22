"""Sensitivity check, not registered: the basic (centred) bootstrap interval for each gap.

    python src/analysis/sensitivity_basic_interval.py --set main

Written on 2026-11-22 after the registered analyses had been run and seen. The registered
interval is the percentile interval (pre-registration, section 4.6), which the decision of
2026-11-17 kept with its known lean toward the null: redrawing conversations tends to raise the
95th percentile of the off-target scores, so the interval sits below the estimate. The basic
interval is centred on the estimate instead. It is computed from the same bootstrap quantiles,

    basic = [2 * gap - ci_high, 2 * gap - ci_low]

so it needs no new resampling and reads only results/<set>/analysis.json.

Writes results/<set>/sensitivity_basic_interval.json. Nothing here changes a registered figure
or the registered reading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def basic(r: dict) -> dict:
    g, lo, hi = r["gap"], r["ci_low"], r["ci_high"]
    return {
        "gap": g,
        "percentile": [lo, hi],
        "basic": [2 * g - hi, 2 * g - lo],
        "basic_above_zero": (2 * g - hi) > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True)
    args = parser.parse_args()
    root = Path("results") / args.set
    a = json.loads((root / "analysis.json").read_text())
    p = a["primary"]
    out: dict = {"note": "not registered; basic interval from the registered percentile quantiles"}
    for bar in ("role_directions", "random_directions"):
        b = p[bar]
        rows = {"within_type": basic(b["primary"])}
        for k, v in b.get("strata", {}).items():
            if "gap" in v:
                rows[f"stratum:{k}"] = basic(v)
        if "pooled" in b and "gap" in b["pooled"]:
            rows["pooled"] = basic(b["pooled"])
        out[bar] = rows
    out["layers_within_type_role"] = {k: basic(v["primary"]) for k, v in a.get("layers", {}).items()}
    (root / "sensitivity_basic_interval.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v.get("within_type") for k, v in out.items() if isinstance(v, dict) and "within_type" in v}, indent=1))


if __name__ == "__main__":
    main()
