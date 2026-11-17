"""Length probe: how long are the model's replies when the length cap is out of the way?

    python src/length_probe.py

Run on the GPU instance, after the pilot. In the pilot, 58 of 90 replies ran into the cap of
384 new tokens. This script regenerates 30 pilot turns (five turn positions in each of the six
pilot conversations) from their pilot contexts with a ceiling of 2,048 new tokens, and records
how long each reply is. It then measures peak memory for the main run's two calls (the capture
pass and a generation) at longer contexts than gate 2 tested.

What this is not: study data. No activation is saved, no projection is computed, no reply is
rated, and no API is called. It informs one setting, max_new_tokens.

A limit to keep in mind when reading the result: from turn 2 on, the context holds the pilot's
earlier replies, most of them cut off at 384 tokens. Only turn 1 is free of that. Lengths are
therefore reported by turn position.

Outputs
  results/length_probe/replies.jsonl   one line per regenerated turn, written as it finishes
  results/length_probe/summary.json    settings, length table, memory table
"""

from __future__ import annotations

import json
import os
import sys
import time
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "length_probe"

CEILING = 2048
TURN_POSITIONS = [1, 8, 15, 4, 12]  # in running order, so a run stopped early still spans the conversation
SEED_OFFSET = 1000  # added to the generation seed, so no probe draw repeats a pilot draw
CUTS = [384, 512, 768, 1024, 1536, 2048]
MEMORY_CONTEXTS = [12000, 16000, 20000, 24000]  # tokens; gate 2 tested 8,151


def probe_items(conversations: list[dict], positions: list[int]) -> list[dict]:
    """The turns to regenerate, each with the messages the pilot gave the model at that turn."""
    items = []
    for k in positions:
        for convo in conversations:
            turns = convo["turns"]
            if k > len(turns):
                continue
            messages = []
            for t in turns[: k - 1]:
                messages.append({"role": "user", "content": t["user"]})
                messages.append({"role": "assistant", "content": t["reply"]})
            messages.append({"role": "user", "content": turns[k - 1]["user"]})
            items.append(
                {
                    "id": convo["id"],
                    "stratum": convo["stratum"],
                    "turn": k,
                    "messages": messages,
                    "pilot_new_tokens": turns[k - 1]["new_tokens"],
                    "pilot_hit_cap": turns[k - 1]["hit_cap"],
                }
            )
    return items


def length_table(rows: list[dict], key: str) -> dict:
    """For each value of `key`: how many replies, their lengths, and the share at or over each cut."""
    table: dict = {}
    for value in sorted({r[key] for r in rows}, key=str):
        lens = sorted(r["new_tokens"] for r in rows if r[key] == value)
        table[str(value)] = {
            "n": len(lens),
            "lengths": lens,
            "median": lens[len(lens) // 2],
            "share_reaching": {str(c): round(sum(n >= c for n in lens) / len(lens), 3) for c in CUTS},
        }
    return table


def memory_messages(conversations: list[dict]) -> list[dict]:
    """A long alternating conversation built by cycling the pilot turns. Filler for the memory test."""
    messages = []
    for _ in range(40):
        for convo in conversations:
            for t in convo["turns"]:
                messages.append({"role": "user", "content": t["user"]})
                messages.append({"role": "assistant", "content": t["reply"]})
    return messages


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gate2_memory import gb, render

    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        cfg = tomllib.load(f)
    gen = cfg["generation"]
    layers = sorted([cfg["projection"]["target_layer"], *cfg["projection"]["reliability_layers"]])
    conversations = [
        json.loads(line)
        for line in (REPO_ROOT / "results" / "pilot" / "conversations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    items = probe_items(conversations, TURN_POSITIONS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    replies_path = OUT_DIR / "replies.jsonl"
    rows = [json.loads(line) for line in replies_path.read_text().splitlines()] if replies_path.exists() else []
    done = {(r["id"], r["turn"]) for r in rows}

    name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    end_ids = model.generation_config.eos_token_id
    end_ids = set(end_ids if isinstance(end_ids, list) else [end_ids])
    memory_after_load = torch.cuda.max_memory_allocated()

    # The same hooks as generate.py, so that memory is measured with them in place.
    store: dict = {}
    for layer in layers:
        model.model.layers[layer].register_forward_hook(
            lambda _m, _i, o, layer=layer: store.__setitem__(layer, (o[0] if isinstance(o, tuple) else o)[0])
        )

    t_start = time.time()
    with open(replies_path, "a") as f:
        for index, item in enumerate(items):
            if (item["id"], item["turn"]) in done:
                continue
            torch.manual_seed(cfg["seeds"]["generation"] + SEED_OFFSET + index)
            torch.cuda.reset_peak_memory_stats()
            ids = render(tok, item["messages"], True)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([ids], device=model.device),
                    do_sample=True,
                    temperature=gen["temperature"],
                    top_p=gen["top_p"],
                    max_new_tokens=CEILING,
                )
            store.clear()
            new = out[0, len(ids) :].tolist()
            row = {
                "id": item["id"],
                "stratum": item["stratum"],
                "turn": item["turn"],
                "context_tokens": len(ids),
                "new_tokens": len(new),
                "hit_ceiling": not (new and new[-1] in end_ids),
                "seconds": round(time.time() - t0, 1),
                "peak_memory_gb": gb(torch.cuda.max_memory_allocated()),
                "pilot_new_tokens": item["pilot_new_tokens"],
                "pilot_hit_cap": item["pilot_hit_cap"],
                "reply": tok.decode(new, skip_special_tokens=True).strip(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(
                f"{row['id']} turn {row['turn']}: {row['new_tokens']} tokens in {row['seconds']} s"
                + (" (ceiling)" if row["hit_ceiling"] else ""),
                flush=True,
            )
    length_seconds = round(time.time() - t_start)

    # Memory at longer contexts: the capture pass exactly as generate.py runs it, then a short
    # generation from the same context. Smallest first; stop at the first one that does not fit.
    filler = render(tok, memory_messages(conversations), False)
    memory = []
    for n in MEMORY_CONTEXTS:
        ids = torch.tensor([filler[:n]], device=model.device)
        entry = {"context_tokens": n}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model(ids)
            store.clear()
            entry["peak_capture_gb"] = gb(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model.generate(ids, do_sample=True, temperature=gen["temperature"], top_p=gen["top_p"], max_new_tokens=16)
            store.clear()
            entry["peak_generation_gb"] = gb(torch.cuda.max_memory_allocated())
            entry["out_of_memory"] = False
        except torch.cuda.OutOfMemoryError:
            store.clear()
            entry["out_of_memory"] = True
        memory.append(entry)
        print(json.dumps(entry), flush=True)
        if entry["out_of_memory"]:
            break

    total_tokens = sum(r["new_tokens"] for r in rows)
    total_seconds = sum(r["seconds"] for r in rows)
    summary = {
        "what": "length probe: pilot turns regenerated with a high ceiling; not study data",
        "git_commit": os.environ.get("STUDY_COMMIT", "unknown"),
        "model": name,
        "thinking_mode": False,
        "generation": {"temperature": gen["temperature"], "top_p": gen["top_p"], "max_new_tokens": CEILING},
        "seed_base": cfg["seeds"]["generation"] + SEED_OFFSET,
        "turn_positions": sorted(TURN_POSITIONS),
        "turns": len(rows),
        "turns_hit_ceiling": sum(r["hit_ceiling"] for r in rows),
        "new_tokens_per_second": round(total_tokens / total_seconds, 2) if total_seconds else None,
        "by_turn": length_table(rows, "turn"),
        "by_stratum": length_table(rows, "stratum"),
        "all": length_table([r | {"all": "all"} for r in rows], "all")["all"],
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "tokenizer_model_max_length": tok.model_max_length,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": gb(torch.cuda.get_device_properties(0).total_memory),
        "memory_after_load_gb": gb(memory_after_load),
        "memory_at_context": memory,
        "length_seconds_this_run": length_seconds,
        "versions": {"torch": torch.__version__, "transformers": transformers_version},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_turn", "by_stratum")}, indent=2))


if __name__ == "__main__":
    main()
