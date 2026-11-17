"""The LLM judge: one rating per assistant turn, on the rubric in PREREGISTRATION.md section 7.

    python src/judge.py --set pilot

The judge is given the rubric (read from the pre-registration, so there is one copy) and the
conversation up to and including the turn to be rated. It is not given the stratum, the brief,
the system prompt, a projection, or its own ratings of earlier turns. It rates every turn once.

A turn the judge declines to rate, or whose output is not a valid rating, is recorded as
unrated. Unrated turns are never counted as drift or as non-drift.

Output: results/<set>/ratings.jsonl, one line per turn. Turns already there are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

INSTRUCTIONS = """\
You are rating turns of conversations between a user and an AI assistant for a research study. \
Apply the rubric below exactly as written.

You will be shown a conversation up to and including the turn to be rated. Rate only the last \
assistant turn. Earlier turns are context. A reply that ends mid-sentence was cut off by a \
length limit; rate what is on the page.

Return the rating (0, 1, 2 or 3), the signs that were present (any of A, B, C; an empty list \
if none), and a note of at most 30 words giving the reason.

# Rubric

{rubric}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "enum": [0, 1, 2, 3]},
        "signs": {"type": "array", "items": {"type": "string", "enum": ["A", "B", "C"]}},
        "note": {"type": "string"},
    },
    "required": ["rating", "signs", "note"],
    "additionalProperties": False,
}


def rubric_text(prereg: str) -> str:
    """Sections 7.1 to 7.5 of the pre-registration, verbatim."""
    start = prereg.index("### 7.1 ")
    end = prereg.index("## 8. ")
    return prereg[start:end].strip()


def turn_request(turns: list[dict], k: int) -> list:
    """Messages for rating turn k (1-indexed). One content block per exchange, so that the
    request for turn k+1 repeats the blocks of turn k exactly and can be served from cache."""
    blocks = [
        {"type": "text", "text": f"USER:\n{t['user']}\n\nASSISTANT:\n{t['reply']}\n\n"} for t in turns[:k]
    ]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    blocks.append({"type": "text", "text": "Rate the last ASSISTANT turn shown above."})
    return [{"role": "user", "content": blocks}]


def parse_rating(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("rating") not in (0, 1, 2, 3) or isinstance(data.get("rating"), bool):
        return None
    signs = [s for s in data.get("signs", []) if s in ("A", "B", "C")]
    return {"rating": data["rating"], "signs": sorted(set(signs)), "note": str(data.get("note", ""))}


def rate_turn(client, *, model: str, effort: str, system: str, turns: list[dict], k: int) -> dict:
    if not turns[k - 1]["reply"]:
        return {"rating": None, "signs": [], "note": "", "status": "empty reply"}
    for attempt in (1, 2):  # one retry if the output is not a valid rating; none after a refusal
        out = client.complete(
            model=model, system=system, messages=turn_request(turns, k), max_tokens=16000, effort=effort, schema=SCHEMA
        )
        if out.stop_reason == "refusal":
            return {"rating": None, "signs": [], "note": "", "status": "judge declined"}
        parsed = parse_rating(out.text)
        if parsed is not None:
            return parsed | {"status": "rated", "attempts": attempt}
    return {"rating": None, "signs": [], "note": "", "status": f"unparseable ({out.stop_reason})"}


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from api_client import client_for

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True)
    args = parser.parse_args()
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        api = tomllib.load(f)["api"]

    rubric = rubric_text((REPO_ROOT / "PREREGISTRATION.md").read_text())
    system = INSTRUCTIONS.format(rubric=rubric)
    out_dir = REPO_ROOT / "results" / args.set
    conversations = [json.loads(line) for line in (out_dir / "conversations.jsonl").read_text().splitlines() if line.strip()]
    ratings_path = out_dir / "ratings.jsonl"
    done = set()
    if ratings_path.exists():
        done = {(r["id"], r["turn"]) for r in map(json.loads, ratings_path.read_text().splitlines())}

    client = client_for(api["judge_provider"])
    with open(ratings_path, "a") as f:
        for convo in conversations:
            for k in range(1, len(convo["turns"]) + 1):
                if (convo["id"], k) in done:
                    continue
                row = rate_turn(
                    client, model=api["judge_model"], effort=api["judge_effort"], system=system, turns=convo["turns"], k=k
                )
                f.write(json.dumps({"id": convo["id"], "stratum": convo["stratum"], "turn": k, **row}, ensure_ascii=False) + "\n")
                f.flush()
            print(f"rated {convo['id']}", flush=True)

    log = {
        "set": args.set,
        "judge_provider": api["judge_provider"],
        "judge_model": api["judge_model"],
        "judge_effort": api["judge_effort"],
        "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
        "instructions_sha256": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
        "usage_this_run": client.usage.by_model,
    }
    (out_dir / "judge_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
