"""Gate 3: does the released axis separate default-assistant replies from persona replies?

Defined in PREREGISTRATION.md section 5.3. Run on the GPU instance:

    python src/gate3_separation.py

Twenty questions, each answered once under the neutral system prompt and once under a role
instruction (configs/gate3_prompts.toml). Each reply's activation is the mean over the reply's
tokens at the target layer, in a forward pass over the conversation. The statistic is the AUC
for default replies projecting HIGHER on the unit-length axis than persona replies.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate2_memory import render, response_spans  # noqa: E402
from load_axis import fetch, load_released, unit  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = REPO_ROOT / "results" / "gate3_separation.json"
REPLIES_PATH = REPO_ROOT / "results" / "gate3_replies.jsonl"
BATCH = 10


def auc_higher(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg), ties as one half."""
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def main() -> None:
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        cfg = tomllib.load(f)
    with open(REPO_ROOT / "configs" / "gate3_prompts.toml", "rb") as f:
        prompts = tomllib.load(f)
    questions, roles = prompts["questions"], prompts["roles"]
    assert len(questions) == len(roles) == cfg["gate3"]["n_per_group"]
    target = cfg["projection"]["target_layer"]
    layers = sorted([target, *cfg["projection"]["reliability_layers"]])
    gen = cfg["generation"]

    fetch(cfg)
    released = load_released(cfg)
    axes = {layer: torch.tensor(unit(released.axis[layer]), dtype=torch.float32) for layer in layers}

    name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    blocks = model.model.layers

    items = []
    for q, role in zip(questions, roles):
        items.append({"group": "default", "role": None, "system": prompts["neutral_system_prompt"], "question": q})
        items.append({"group": "persona", "role": role["name"], "system": role["instruction"], "question": q})

    # 1. Generate.
    torch.manual_seed(cfg["seeds"]["generation"])
    for start in range(0, len(items), BATCH):
        batch = items[start : start + BATCH]
        ids = [
            render(tok, [{"role": "system", "content": it["system"]}, {"role": "user", "content": it["question"]}], True)
            for it in batch
        ]
        width = max(len(x) for x in ids)
        input_ids = torch.tensor([[tok.pad_token_id] * (width - len(x)) + x for x in ids], device=model.device)
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=mask,
                do_sample=True,
                temperature=gen["temperature"],
                top_p=gen["top_p"],
                max_new_tokens=gen["max_new_tokens"],
            )
        for it, row in zip(batch, out):
            it["reply"] = tok.decode(row[width:], skip_special_tokens=True).strip()
        print(f"generated {start + len(batch)} of {len(items)}", flush=True)

    # 2. Capture and project.
    store: dict[int, torch.Tensor] = {}
    handles = [
        blocks[layer].register_forward_hook(
            lambda _m, _i, o, layer=layer: store.__setitem__(layer, (o[0] if isinstance(o, tuple) else o)[0])
        )
        for layer in layers
    ]
    for it in items:
        convo = [
            {"role": "system", "content": it["system"]},
            {"role": "user", "content": it["question"]},
            {"role": "assistant", "content": it["reply"]},
        ]
        full, spans = response_spans(tok, convo)
        (a, b), = spans
        with torch.no_grad():
            model(torch.tensor([full], device=model.device))
        it["reply_tokens"] = b - a
        it["projection"] = {
            str(layer): round(float(store[layer][a:b].float().mean(dim=0).cpu() @ axes[layer]), 4) for layer in layers
        }
    for h in handles:
        h.remove()

    def auc_at(layer: int) -> float:
        d = np.array([it["projection"][str(layer)] for it in items if it["group"] == "default"])
        p = np.array([it["projection"][str(layer)] for it in items if it["group"] == "persona"])
        return round(auc_higher(d, p), 4)

    auc = auc_at(target)
    threshold = cfg["gate3"]["min_auc"]
    d = [it["projection"][str(target)] for it in items if it["group"] == "default"]
    p = [it["projection"][str(target)] for it in items if it["group"] == "persona"]
    result = {
        "gate": "3: the released axis separates default-assistant replies from persona replies",
        "git_commit": os.environ.get("STUDY_COMMIT", "unknown"),
        "model": name,
        "thinking_mode": False,
        "layer": target,
        "n_default": len(d),
        "n_persona": len(p),
        "threshold_min_auc": threshold,
        "auc_default_higher": auc,
        "passed": bool(auc >= threshold),
        "reported_not_gating": {
            "mean_projection_default": round(float(np.mean(d)), 4),
            "mean_projection_persona": round(float(np.mean(p)), 4),
            "min_default": min(d),
            "max_persona": max(p),
            "auc_by_layer": {str(layer): auc_at(layer) for layer in layers},
            "empty_replies": sum(1 for it in items if it["reply_tokens"] == 0),
        },
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    with open(REPLIES_PATH, "w") as f:
        for it in items:
            f.write(json.dumps({k: it[k] for k in ("group", "role", "question", "reply", "reply_tokens", "projection")}) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
