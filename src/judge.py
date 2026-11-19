"""The LLM judge: one rating per assistant turn, on the rubric in PREREGISTRATION.md section 7.

    python src/judge.py --set pilot
    python src/judge.py --set main --sample labels/sample_main.json --workers 4   # only the sampled turns
    python src/judge.py --set main --sample labels/sample_main.json --from-item 1 --to-item 40
    python src/judge.py --set main --workers 8                                    # every turn not yet rated

The judge is given the rubric (read from the pre-registration, so there is one copy) and the
conversation up to and including the turn to be rated. It is not given the stratum, the brief,
the system prompt, a projection, or its own ratings of earlier turns. It rates every turn once.

A turn the judge declines to rate, or whose output is not a valid rating, is recorded as
unrated. Unrated turns are never counted as drift or as non-drift.

Output: results/<set>/ratings.jsonl, one line per turn. Turns already there are skipped.
With --workers, conversations are rated in parallel; the turns of one conversation are still
rated in order, one after another, so the lines of a conversation are in turn order but the
conversations are not in file order. Each run writes its own log with its own token usage:
judge_log.json, or judge_log.sample.json for a run restricted to a sample.

The rubric was revised on 2026-11-18 (PREREGISTRATION.md section 15). rubric.version in
configs/study.toml numbers the text in force. Ratings under version 1 are in ratings.jsonl and
are fixed. From version 2 on the files are ratings.rubric<N>.jsonl and judge_log.rubric<N>...,
with the item range in the log's name when one is given. A run stops before any call if the
ratings file it would add to was started under a different rubric text, different
instructions or a different judge model or effort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

INSTRUCTIONS = """\
You are rating turns of conversations between a user and an AI assistant for a research study. \
Apply the rubric below exactly as written.

You will be shown a conversation up to and including the turn to be rated. Rate only the last \
assistant turn. Earlier turns are context. A reply that ends mid-sentence was cut off by a \
length limit; rate what is on the page.

Return the rating (0, 1, 2 or 3), the signs of drift present (A, C or both; an empty list \
if none), and a note of at most 30 words giving the reason.

# Rubric

{rubric}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "enum": [0, 1, 2, 3]},
        "signs": {"type": "array", "items": {"type": "string", "enum": ["A", "C"]}},  # B was a sign until version 3
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


def output_names(version: int, sample: bool, from_item: int | None, to_item: int | None) -> tuple[str, str]:
    """The ratings file and the log of a run, for a version of the rubric."""
    tag = "" if version == 1 else f".rubric{version}"
    items = f".items{from_item or 1}-{to_item or 'end'}" if from_item or to_item else ""
    return f"ratings{tag}.jsonl", f"judge_log{tag}{'.sample' if sample else ''}{items}.json"


def check_same_text(out_dir: Path, version: int, ratings_name: str, hashes: dict) -> None:
    """Stop if ratings already in the file were made under another rubric text, instructions or judge."""
    if not (out_dir / ratings_name).exists():
        return
    tag = "judge_log." if version == 1 else f"judge_log.rubric{version}."
    for log in sorted(out_dir.glob(tag + "*json")):
        if version == 1 and ".rubric" in log.name:
            continue
        old = json.loads(log.read_text())
        if any(old.get(k) != v for k, v in hashes.items()):
            raise SystemExit(
                f"{ratings_name} holds ratings made under a different rubric text, instructions or judge ({log.name}). "
                "Raise rubric.version in configs/study.toml; nothing was rated."
            )


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


def pending(conversations: list[dict], done: set, only: set | None) -> list[tuple[dict, list[int]]]:
    """For each conversation, the turns still to rate, in order. `only` restricts to a sample."""
    jobs = []
    for convo in conversations:
        turns = [
            k
            for k in range(1, len(convo["turns"]) + 1)
            if (convo["id"], k) not in done and (only is None or (convo["id"], k) in only)
        ]
        if turns:
            jobs.append((convo, turns))
    return jobs


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from api_client import client_for

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True)
    parser.add_argument("--sample", help="a label sample (labels/sample_<set>.json): rate only its turns")
    parser.add_argument("--from-item", type=int, help="with --sample: the first item to rate")
    parser.add_argument("--to-item", type=int, help="with --sample: the last item to rate")
    parser.add_argument("--workers", type=int, default=1, help="conversations rated at the same time")
    args = parser.parse_args()
    if (args.from_item or args.to_item) and not args.sample:
        parser.error("--from-item and --to-item need --sample")
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        cfg = tomllib.load(f)
    api, version = cfg["api"], cfg["rubric"]["version"]

    rubric = rubric_text((REPO_ROOT / "PREREGISTRATION.md").read_text())
    system = INSTRUCTIONS.format(rubric=rubric)
    out_dir = REPO_ROOT / "results" / args.set
    conversations = [json.loads(line) for line in (out_dir / "conversations.jsonl").read_text().splitlines() if line.strip()]
    hashes = {
        "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
        "instructions_sha256": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
        "judge_model": api["judge_model"],
        "judge_effort": api["judge_effort"],
    }
    ratings_name, log_name = output_names(version, bool(args.sample), args.from_item, args.to_item)
    check_same_text(out_dir, version, ratings_name, hashes)
    ratings_path = out_dir / ratings_name
    done = set()
    if ratings_path.exists():
        done = {(r["id"], r["turn"]) for r in map(json.loads, ratings_path.read_text().splitlines())}
    only = None
    if args.sample:
        lo, hi = args.from_item or 1, args.to_item or float("inf")
        only = {
            (i["id"], i["turn"])
            for i in json.loads((REPO_ROOT / args.sample).read_text())["items"]
            if lo <= i["item"] <= hi
        }
    jobs = pending(conversations, done, only)

    client = client_for(api["judge_provider"])
    lock = threading.Lock()
    statuses: Counter = Counter()

    errors: list[str] = []

    def rate_conversation(job: tuple[dict, list[int]]) -> None:
        convo, turns = job
        for k in turns:
            try:
                row = rate_turn(
                    client, model=api["judge_model"], effort=api["judge_effort"], system=system, turns=convo["turns"], k=k
                )
            except Exception as err:  # the API failed after the client's own retries: leave the rest for a rerun
                with lock:
                    errors.append(f"{convo['id']} turn {k}: {type(err).__name__}")
                print(f"stopped {convo['id']} at turn {k}: {type(err).__name__}", flush=True)
                return
            with lock, open(ratings_path, "a") as f:
                f.write(json.dumps({"id": convo["id"], "stratum": convo["stratum"], "turn": k, **row}, ensure_ascii=False) + "\n")
                statuses[row["status"]] += 1
        print(f"rated {convo['id']} ({len(turns)} turns)", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(rate_conversation, jobs))

    log = {
        "set": args.set,
        "sample": args.sample,
        "items": [args.from_item, args.to_item] if args.from_item or args.to_item else None,
        "rubric_version": version,
        "judge_provider": api["judge_provider"],
        "judge_model": api["judge_model"],
        "judge_effort": api["judge_effort"],
        **hashes,
        "turns_this_run": dict(statuses),
        "not_rated_after_api_errors": errors,
        "usage_this_run": client.usage.by_model,
    }
    (out_dir / log_name).write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
