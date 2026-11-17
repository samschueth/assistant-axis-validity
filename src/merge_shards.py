"""Join the transcript files of a run that was split across instances.

    python src/merge_shards.py --briefs configs/main_briefs.toml --set main

Reads results/<set>/conversations.*of*.jsonl and writes results/<set>/conversations.jsonl with
the conversations in the order of the briefs file. Says which briefs have no conversation yet.
The per-shard files and logs are kept.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def merge(brief_ids: list[str], shard_lines: list[str]) -> tuple[list[str], list[str]]:
    """Lines in brief order, and the ids with no conversation. A repeated or unknown id is an error."""
    by_id: dict[str, str] = {}
    for line in shard_lines:
        cid = json.loads(line)["id"]
        if cid in by_id:
            raise ValueError(f"conversation {cid} appears twice")
        if cid not in brief_ids:
            raise ValueError(f"conversation {cid} is not in the briefs file")
        by_id[cid] = line
    return [by_id[i] for i in brief_ids if i in by_id], [i for i in brief_ids if i not in by_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--briefs", required=True)
    parser.add_argument("--set", required=True)
    args = parser.parse_args()
    with open(REPO_ROOT / args.briefs, "rb") as f:
        brief_ids = [b["id"] for b in tomllib.load(f)["briefs"]]
    out_dir = REPO_ROOT / "results" / args.set
    lines = [
        line
        for path in sorted(out_dir.glob("conversations.*of*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    merged, missing = merge(brief_ids, lines)
    (out_dir / "conversations.jsonl").write_text("".join(line + "\n" for line in merged))
    print(f"{len(merged)} of {len(brief_ids)} conversations merged")
    if missing:
        print("not generated yet: " + ", ".join(missing))


if __name__ == "__main__":
    main()
