"""Generate conversations and capture activations. Run on the GPU instance.

    python src/generate.py --briefs configs/pilot_briefs.toml --set pilot

For each brief: the simulated user (an API model) and the model under study alternate for the
configured number of assistant turns. After each assistant turn, one forward pass over the
conversation up to that turn records the mean activation over the reply's tokens at the target
layer and the reliability layers (see capture.py for why the pass stops at the turn).

Outputs
  results/<set>/conversations.jsonl   transcripts, one conversation per line (committed)
  results/<set>/generation_log.json   settings, model versions, API token usage (committed)
  data/runs/<set>/<id>.npy            activations, shape (turns, layers, hidden) (not committed)

No projection is computed here. A conversation already in conversations.jsonl is skipped, so an
interrupted run can be restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import AnthropicClient  # noqa: E402
from capture import last_reply_span  # noqa: E402
from conversation import SimulatedUser, run_conversation  # noqa: E402
from gate2_memory import render  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--briefs", required=True, help="TOML file of scenario briefs")
    parser.add_argument("--set", required=True, help="name of this set of conversations, e.g. pilot")
    args = parser.parse_args()

    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        cfg = tomllib.load(f)
    with open(REPO_ROOT / args.briefs, "rb") as f:
        briefs = tomllib.load(f)["briefs"]
    gen, api = cfg["generation"], cfg["api"]
    n_turns = cfg["sample"]["assistant_turns"]
    layers = sorted([cfg["projection"]["target_layer"], *cfg["projection"]["reliability_layers"]])

    out_dir = REPO_ROOT / "results" / args.set
    act_dir = REPO_ROOT / "data" / "runs" / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    transcripts = out_dir / "conversations.jsonl"
    done = set()
    if transcripts.exists():
        done = {json.loads(line)["id"] for line in transcripts.read_text().splitlines() if line.strip()}

    name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    blocks = model.model.layers
    assert len(blocks) == cfg["model"]["total_layers"]
    end_ids = model.generation_config.eos_token_id
    end_ids = set(end_ids if isinstance(end_ids, list) else [end_ids])

    store: dict[int, torch.Tensor] = {}
    for layer in layers:
        blocks[layer].register_forward_hook(
            lambda _m, _i, o, layer=layer: store.__setitem__(layer, (o[0] if isinstance(o, tuple) else o)[0])
        )

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    def rend(messages: list, add_generation_prompt: bool) -> list[int]:
        return render(tok, messages, add_generation_prompt)

    client = AnthropicClient()
    t_start = time.time()

    for index, brief in enumerate(briefs):
        if brief["id"] in done:
            print(f"skip {brief['id']}: already generated", flush=True)
            continue
        torch.manual_seed(cfg["seeds"]["generation"] + index)
        activations: list[np.ndarray] = []

        def next_assistant(messages: list) -> dict:
            ids = rend(messages, True)
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([ids], device=model.device),
                    do_sample=True,
                    temperature=gen["temperature"],
                    top_p=gen["top_p"],
                    max_new_tokens=gen["max_new_tokens"],
                )
            new = out[0, len(ids) :].tolist()
            reply = tok.decode(new, skip_special_tokens=True).strip()
            turn = {"reply": reply, "new_tokens": len(new), "hit_cap": not (new and new[-1] in end_ids)}
            if not reply:
                activations.append(np.full((len(layers), cfg["model"]["hidden_size"]), np.nan, dtype=np.float32))
                return turn | {"reply_tokens": 0, "matches_generation": None}
            span = last_reply_span(rend, encode, [*messages, {"role": "assistant", "content": reply}])
            with torch.no_grad():
                model(torch.tensor([span["ids"]], device=model.device))
            a, b = span["start"], span["end"]
            activations.append(
                np.stack([store[layer][a:b].float().mean(dim=0).cpu().numpy() for layer in layers]).astype(np.float32)
            )
            store.clear()
            return turn | {"reply_tokens": b - a, "matches_generation": span["matches_generation"]}

        user = SimulatedUser(
            client, model=api["user_model"], effort=api["user_effort"], brief=brief["brief"], n_turns=n_turns
        )
        t0 = time.time()
        convo = run_conversation(
            system_prompt=gen["system_prompt"], n_turns=n_turns, next_user=user.next_message, next_assistant=next_assistant
        )
        np.save(act_dir / f"{brief['id']}.npy", np.stack(activations) if activations else np.zeros((0, len(layers), 0)))
        record = {
            "id": brief["id"],
            "stratum": brief["stratum"],
            "set": args.set,
            "complete": convo["complete"],
            "stopped": convo["stopped"],
            "seconds": round(time.time() - t0),
            "turns": convo["turns"],
        }
        with open(transcripts, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{brief['id']}: {len(convo['turns'])} turns in {record['seconds']} s", flush=True)

    log = {
        "set": args.set,
        "git_commit": os.environ.get("STUDY_COMMIT", "unknown"),
        "model": name,
        "thinking_mode": False,
        "system_prompt": gen["system_prompt"],
        "generation": {k: gen[k] for k in ("temperature", "top_p", "max_new_tokens")},
        "assistant_turns": n_turns,
        "layers": layers,
        "seed_base": cfg["seeds"]["generation"],
        "user_model": api["user_model"],
        "user_effort": api["user_effort"],
        "user_model_usage": client.usage.by_model,
        "versions": {"torch": torch.__version__, "transformers": transformers_version},
        "gpu": torch.cuda.get_device_name(0),
        "seconds": round(time.time() - t_start),
    }
    (out_dir / "generation_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
