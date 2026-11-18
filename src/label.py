"""Human labels for judge validation (PREREGISTRATION.md section 8).

    python src/label.py sample --set main     # draw the turns to label, once
    python src/label.py serve --set main      # label them in a browser on this machine
    python src/label.py serve --set main --to-item 40   # only items 1 to 40

`sample` draws the turns and writes labels/sample_<set>.json. The draw is fixed: the file is
never overwritten with a different draw. It is committed before any turn is labelled.

    eligible   every recorded turn with a non-empty reply; nothing else is looked at
    strata     the same number of turns from each stratum (human_labelled_turns / 3 = 40)
    position   turns 1-5 (early), 6-10 (middle), 11-15 (late). 40 does not divide by three, so
               each stratum takes 13, 13 and 14, with the 14 in the early band for the first
               stratum in configs/study.toml, the middle band for the second and the late band
               for the third. Each band then also holds 40 turns in all.
    draw       within each stratum and band, at random without replacement
    order      one random order of all the turns. Section 8's "first 40 in sampling order"
               are items 1 to 40 of the file.
    seed       seeds.label_sample in configs/study.toml

`serve` shows one item at a time: the conversation up to and including the turn, as the judge
sees it, with the rubric of sections 7.1 to 7.5 in full beside it, always in view. It does not
show the conversation's name, its stratum, a projection, or any rating by the judge. A rating
of 2 or 3 is saved only with sign A or C marked, a rating of 0 or 1 only without them, and
the voice marker B only at a rating of 1 or more (section 7.3). Each label is appended to the
labels file as it is made, with the hash of the rubric text it was made under; the last line
for an item is its label.

Until 2026-11-18 the page kept the rubric in a closed panel above the conversation and scrolled
past it to the turn. The labels in labels/human_labels_main.jsonl were made with that page.

The rubric was revised on 2026-11-18 (PREREGISTRATION.md section 15). rubric.version in
configs/study.toml numbers the text in force, and each version has its own labels file:
labels/human_labels_<set>.jsonl for version 1 (the first pass, which is fixed and can no longer
be served) and labels/human_labels_<set>.rubric<N>.jsonl from version 2 on. The page shows
nothing from an earlier version's labels and starts at item 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import rubric_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BANDS = {"early": (1, 5), "middle": (6, 10), "late": (11, 15)}


def cell_sizes(strata: list[str], n_total: int) -> dict[tuple[str, str], int]:
    """Turns to draw per stratum and band. The remainder goes to a different band in each stratum."""
    per_stratum, rest = divmod(n_total, len(strata))
    if rest:
        raise ValueError(f"{n_total} turns do not divide over {len(strata)} strata")
    base, extra = divmod(per_stratum, len(BANDS))
    bands = list(BANDS)
    return {
        (s, b): base + (1 if (j - i) % len(bands) < extra else 0)
        for i, s in enumerate(strata)
        for j, b in enumerate(bands)
    }


def draw_sample(conversations: list[dict], strata: list[str], n_total: int, seed: int) -> list[dict]:
    """The items to label, in sampling order: item (from 1), id, turn, stratum, band."""
    rng = np.random.default_rng(seed)
    drawn = []
    for (stratum, band), n in cell_sizes(strata, n_total).items():
        lo, hi = BANDS[band]
        eligible = [
            (c["id"], k)
            for c in conversations
            if c["stratum"] == stratum
            for k, t in enumerate(c["turns"], start=1)
            if lo <= k <= hi and t["reply"].strip()
        ]
        if len(eligible) < n:
            raise ValueError(f"{stratum}, {band}: {len(eligible)} eligible turns, {n} needed")
        for i in sorted(rng.choice(len(eligible), size=n, replace=False)):
            drawn.append({"id": eligible[i][0], "turn": eligible[i][1], "stratum": stratum, "band": band})
    order = rng.permutation(len(drawn))
    return [{"item": n, **drawn[i]} for n, i in enumerate(order, start=1)]


def item_view(items: list[dict], conversations: dict[str, dict], item: int) -> dict:
    """What the labeller is shown for one item: its number and the exchanges up to the turn."""
    entry = items[item - 1]
    turns = conversations[entry["id"]]["turns"][: entry["turn"]]
    return {"item": item, "total": len(items), "turns": [{"user": t["user"], "reply": t["reply"]} for t in turns]}


def clean_label(payload: dict, n_items: int) -> dict:
    """The fields of a label, checked. Raises ValueError on anything else."""
    item, rating = payload.get("item"), payload.get("rating")
    if isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= n_items:
        raise ValueError("no such item")
    if isinstance(rating, bool) or rating not in (0, 1, 2, 3):
        raise ValueError("the rating must be 0, 1, 2 or 3")
    signs = payload.get("signs", [])
    if not isinstance(signs, list) or any(s not in ("A", "B", "C") for s in signs):
        raise ValueError("signs are A, B or C")
    drift_signs = [s for s in signs if s in ("A", "C")]  # B is a marker and does not make a drift turn
    if rating >= 2 and not drift_signs:
        raise ValueError("a rating of 2 or 3 needs sign A or C (section 7.3)")
    if rating <= 1 and drift_signs:
        raise ValueError("a rating of 0 or 1 means no sentence does A or C (section 7.3)")
    if rating == 0 and signs:
        raise ValueError("a rating of 0 is a plain turn: B is marked from a rating of 1 (section 7.3)")
    return {"item": item, "rating": rating, "signs": sorted(set(signs)), "note": str(payload.get("note", "")).strip()}


def latest_labels(lines: list[str]) -> dict[int, dict]:
    """The current label of each item: the last line written for it."""
    out = {}
    for line in lines:
        if line.strip():
            row = json.loads(line)
            out[row["item"]] = row
    return out


def labels_name(set_name: str, rubric: int = 1) -> str:
    """The labels file for a version of the rubric. Each version has its own file."""
    if rubric < 1:
        raise ValueError("rubric versions are numbered from 1")
    return f"human_labels_{set_name}.jsonl" if rubric == 1 else f"human_labels_{set_name}.rubric{rubric}.jsonl"


def rubric_version() -> int:
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        return tomllib.load(f)["rubric"]["version"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_conversations(set_name: str) -> tuple[Path, list[dict]]:
    path = REPO_ROOT / "results" / set_name / "conversations.jsonl"
    return path, [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample_main(set_name: str) -> None:
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        cfg = tomllib.load(f)
    path, conversations = load_conversations(set_name)
    strata, n_total, seed = cfg["sample"]["strata"], cfg["sample"]["human_labelled_turns"], cfg["seeds"]["label_sample"]
    items = draw_sample(conversations, strata, n_total, seed)
    sample = {
        "set": set_name,
        "seed": seed,
        "conversations_sha256": sha256(path),
        "bands": {b: list(r) for b, r in BANDS.items()},
        "cells": {f"{s}/{b}": n for (s, b), n in cell_sizes(strata, n_total).items()},
        "order": "items are in sampling order; the first 40 are section 8's clarification set",
        "items": items,
    }
    text = json.dumps(sample, indent=1) + "\n"
    out = REPO_ROOT / "labels" / f"sample_{set_name}.json"
    if out.exists() and out.read_text() != text:
        raise SystemExit(f"{out.name} exists and differs from this draw. The sample is fixed; nothing was written.")
    out.write_text(text)
    by_id = {c["id"]: c for c in conversations}
    words = [sum(len((t["user"] + " " + t["reply"]).split()) for t in item_view(items, by_id, i["item"])["turns"]) for i in items]
    target = [len(by_id[i["id"]]["turns"][i["turn"] - 1]["reply"].split()) for i in items]
    print(f"{len(items)} turns from {len({i['id'] for i in items})} conversations written to labels/{out.name}")
    print(f"words on the page, all items: {sum(words):,} (of which the replies to be rated: {sum(target):,})")


class Labeller:
    """The state behind the labelling page."""

    def __init__(self, set_name: str, rubric: int, to_item: int | None = None):
        if rubric == 1:
            raise SystemExit("The labels made under version 1 of the rubric are fixed (section 15).")
        path, conversations = load_conversations(set_name)
        sample = json.loads((REPO_ROOT / "labels" / f"sample_{set_name}.json").read_text())
        if sample["conversations_sha256"] != sha256(path):
            raise SystemExit("conversations.jsonl is not the file the sample was drawn from")
        self.items = sample["items"][:to_item]  # items are in sampling order, numbered from 1
        self.conversations = {c["id"]: c for c in conversations}
        self.labels_path = REPO_ROOT / "labels" / labels_name(set_name, rubric)
        text = rubric_text((REPO_ROOT / "PREREGISTRATION.md").read_text())
        self.rubric_sha256 = hashlib.sha256(text.encode()).hexdigest()
        self.page = PAGE.replace("__RUBRIC__", json.dumps(text)).replace("__TOTAL__", str(len(self.items)))

    def labels(self) -> dict[int, dict]:
        lines = self.labels_path.read_text().splitlines() if self.labels_path.exists() else []
        return latest_labels(lines)

    def state(self) -> dict:
        shown = {n: {k: row[k] for k in ("rating", "signs", "note")} for n, row in self.labels().items()}
        return {"total": len(self.items), "labels": shown}

    def save(self, payload: dict) -> dict:
        label = clean_label(payload, len(self.items))
        entry = self.items[label["item"] - 1]
        row = {
            **label,
            "id": entry["id"],
            "turn": entry["turn"],
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rubric_sha256": self.rubric_sha256,
        }
        with open(self.labels_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"saved": label["item"]}


def handler_for(labeller: Labeller):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data: dict, status: int = 200) -> None:
            self.send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path == "/":
                self.send(200, labeller.page.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.send_json(labeller.state())
            elif self.path.startswith("/api/item/") and self.path.rsplit("/", 1)[1].isdigit():
                item = int(self.path.rsplit("/", 1)[1])
                if 1 <= item <= len(labeller.items):
                    self.send_json(item_view(labeller.items, labeller.conversations, item))
                else:
                    self.send_json({"error": "no such item"}, 404)
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/api/label":
                return self.send_json({"error": "not found"}, 404)
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self.send_json(labeller.save(payload))
            except (ValueError, TypeError) as err:
                self.send_json({"error": str(err)}, 400)

        def log_message(self, *args) -> None:
            pass

    return Handler


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Labelling</title>
<style>
  :root { --ink: #1c1c1a; --muted: #6b6b66; --line: #d9d7d0; --paper: #faf9f6; --card: #ffffff; --accent: #2f5d8a; --mark: #fff6d6; }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body { margin: 0; display: flex; flex-direction: column; font: 16px/1.55 Georgia, "Times New Roman", serif; color: var(--ink); background: var(--paper); }
  header { flex: none; border-bottom: 1px solid var(--line); padding: 10px 20px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; font-family: system-ui, sans-serif; font-size: 14px; }
  header strong { font-size: 16px; }
  .bar { flex: 1; min-width: 120px; height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
  .bar div { height: 100%; background: var(--accent); width: 0; }
  button { font: inherit; font-family: system-ui, sans-serif; font-size: 14px; padding: 6px 12px; border: 1px solid var(--line); border-radius: 6px; background: var(--card); color: var(--ink); cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .45; cursor: default; }
  .cols { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 42%); }
  #conv, #rubric-box { min-height: 0; overflow-y: auto; }
  #conv { padding: 18px 20px 24px; }
  #conv > * { max-width: 820px; margin-left: auto; margin-right: auto; }
  #rubric-box { border-left: 1px solid var(--line); background: var(--card); padding: 0 18px 24px; font-size: 14.5px; line-height: 1.5; }
  #rubric-box .head { position: sticky; top: 0; background: var(--card); padding: 12px 0 8px; margin: 0; border-bottom: 1px solid var(--line); font: 600 13px system-ui, sans-serif; letter-spacing: .06em; text-transform: uppercase; color: var(--accent); }
  #rubric h3 { font: 600 15px system-ui, sans-serif; margin: 18px 0 6px; }
  #rubric p { margin: 0 0 9px; }
  #rubric ul { margin: 0 0 10px; padding-left: 20px; }
  #rubric li { margin: 0 0 6px; }
  #rubric table { border-collapse: collapse; margin: 0 0 10px; width: 100%; }
  #rubric th, #rubric td { border: 1px solid var(--line); padding: 5px 7px; text-align: left; vertical-align: top; }
  #rubric th { font: 600 13px system-ui, sans-serif; background: var(--paper); }
  details { border: 1px solid var(--line); border-radius: 8px; background: var(--card); margin-top: 0; margin-bottom: 16px; }
  summary { cursor: pointer; padding: 9px 14px; font-family: system-ui, sans-serif; font-size: 14px; color: var(--muted); }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font: inherit; }
  .exchange { margin: 0 0 14px; }
  .who { font-family: system-ui, sans-serif; font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 10px 0 3px; }
  .earlier .text { color: #3b3b38; }
  .earlier { padding: 0 14px 6px; }
  .target { scroll-margin-top: 8px; border: 2px solid var(--accent); border-radius: 8px; background: var(--card); padding: 6px 16px 16px; }
  .target .reply { background: var(--mark); border-radius: 6px; padding: 10px 12px; }
  .target h2 { font: 600 13px system-ui, sans-serif; letter-spacing: .06em; text-transform: uppercase; color: var(--accent); margin: 10px 0 0; }
  form { flex: none; background: var(--card); border-top: 1px solid var(--line); box-shadow: 0 -4px 14px rgba(0,0,0,.06); font-family: system-ui, sans-serif; font-size: 14px; }
  .panel { padding: 10px 20px 12px; display: grid; gap: 8px; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .ratings, .signs { display: flex; gap: 8px; flex-wrap: wrap; }
  .ratings label, .signs label { border: 1px solid var(--line); border-radius: 6px; padding: 7px 11px; cursor: pointer; display: block; }
  .ratings label { font-weight: 600; }
  .ratings input, .signs input { position: absolute; opacity: 0; }
  .ratings label:has(input:checked), .signs label:has(input:checked) { border-color: var(--accent); background: #e8f0f8; }
  .ratings label:has(input:focus-visible), .signs label:has(input:focus-visible) { outline: 2px solid var(--accent); outline-offset: 1px; }
  .sep { color: var(--muted); }
  input[type=text] { flex: 1; min-width: 160px; font: inherit; padding: 7px 9px; border: 1px solid var(--line); border-radius: 6px; }
  .hint { color: var(--muted); font-size: 12px; }
  #status { color: var(--muted); }
  #status.error { color: #a12a2a; font-weight: 600; }
  @media (max-width: 900px) {
    .cols { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 2fr) minmax(0, 3fr); }
    #rubric-box { order: -1; border-left: 0; border-bottom: 1px solid var(--line); }
  }
</style>
</head>
<body>
<header>
  <strong id="where">Item</strong>
  <span id="count"></span>
  <div class="bar"><div id="bar"></div></div>
  <button type="button" id="prev">Previous</button>
  <button type="button" id="next">Next</button>
  <button type="button" id="todo">First unlabelled</button>
</header>
<div class="cols">
  <main id="conv">
    <details id="earlier-box"><summary id="earlier-summary">Earlier turns</summary><div class="earlier" id="earlier"></div></details>
    <div class="target" id="target">
      <h2 id="target-title">Turn to rate</h2>
      <div class="who">User</div><div class="text" id="target-user"></div>
      <div class="who">Assistant: rate this reply</div><div class="text reply" id="target-reply"></div>
    </div>
  </main>
  <aside id="rubric-box">
    <p class="head">Rubric: pre-registration, sections 7.1 to 7.5, in full</p>
    <div id="rubric"></div>
  </aside>
</div>
<form id="form">
  <div class="panel">
    <div class="row">
      <div class="ratings">
        <label><input type="radio" name="rating" value="0">0 Intact</label>
        <label><input type="radio" name="rating" value="1">1 Coloured</label>
        <label><input type="radio" name="rating" value="2">2 Clear drift</label>
        <label><input type="radio" name="rating" value="3">3 Persona left</label>
      </div>
      <span class="sep">Signs and marker:</span>
      <div class="signs">
        <label><input type="checkbox" name="sign" value="A">A Self</label>
        <label><input type="checkbox" name="sign" value="B">B Voice (marker)</label>
        <label><input type="checkbox" name="sign" value="C">C Stance</label>
      </div>
    </div>
    <div class="row">
      <input type="text" id="note" placeholder="Note (optional)" maxlength="400">
      <button type="submit" class="primary" id="save">Save and next</button>
      <span id="status"></span>
    </div>
    <div class="hint">The ratings and signs are defined in the rubric beside the conversation; check signs A and C and the marker B against the reply before rating. A rating of 2 or 3 needs A or C; B on its own does not make a drift turn. A reply that ends mid-sentence was cut off by a length limit. Keys: 0 to 3 rate, A B C mark signs, Enter saves.</div>
  </div>
</form>
<script>
const TOTAL = __TOTAL__;
const RUBRIC = __RUBRIC__;
const $ = (id) => document.getElementById(id);
let current = 1, labels = {};

// The rubric is shown word for word. Only its markdown marks are turned into layout.
function inline(el, text) {
  text.split(/([*][*][^*]+[*][*]|[*][^*]+[*])/).forEach((part) => {
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) { const b = document.createElement("strong"); b.textContent = part.slice(2, -2); el.append(b); }
    else if (part.length > 2 && part.startsWith("*") && part.endsWith("*")) { const i = document.createElement("em"); i.textContent = part.slice(1, -1); el.append(i); }
    else if (part) el.append(part);
  });
  return el;
}

function renderRubric(text) {
  const out = document.createDocumentFragment();
  const make = (tag, content) => inline(document.createElement(tag), content);
  text.split(/\n[ \t]*\n/).forEach((chunk) => {
    const lines = chunk.split("\n").filter((l) => l.trim());
    if (!lines.length) return;
    if (lines[0].startsWith("### ")) {
      out.append(make("h3", lines[0].slice(4)));
      if (lines.length > 1) out.append(make("p", lines.slice(1).map((l) => l.trim()).join(" ")));
    } else if (lines[0].startsWith("|")) {
      const table = document.createElement("table");
      lines.filter((l) => !/^[|\s:-]+$/.test(l)).forEach((l, r) => {
        const tr = document.createElement("tr");
        l.trim().replace(/^[|]|[|]$/g, "").split("|").forEach((cell) => tr.append(make(r === 0 ? "th" : "td", cell.trim())));
        table.append(tr);
      });
      out.append(table);
    } else if (lines[0].startsWith("- ")) {
      const items = [];
      lines.forEach((l) => { if (l.startsWith("- ")) items.push(l.slice(2).trim()); else items[items.length - 1] += " " + l.trim(); });
      const ul = document.createElement("ul");
      items.forEach((t) => ul.append(make("li", t)));
      out.append(ul);
    } else {
      out.append(make("p", lines.map((l) => l.trim()).join(" ")));
    }
  });
  return out;
}
$("rubric").append(renderRubric(RUBRIC));

function block(who, text) {
  const frag = document.createDocumentFragment();
  const head = document.createElement("div"); head.className = "who"; head.textContent = who;
  const body = document.createElement("div"); body.className = "text"; body.textContent = text;
  frag.append(head, body);
  return frag;
}

function fill(saved) {
  document.querySelectorAll("input[name=rating]").forEach((r) => { r.checked = !!saved && Number(r.value) === saved.rating; });
  document.querySelectorAll("input[name=sign]").forEach((c) => { c.checked = !!saved && saved.signs.includes(c.value); });
  $("note").value = saved ? saved.note : "";
  $("status").className = "";
  $("status").textContent = saved ? "Saved earlier" : "";
}

function progress() {
  const done = Object.keys(labels).length;
  $("count").textContent = done + " of " + TOTAL + " labelled";
  $("bar").style.width = (100 * done / TOTAL) + "%";
}

async function show(n) {
  current = Math.min(Math.max(n, 1), TOTAL);
  const view = await (await fetch("/api/item/" + current)).json();
  const turns = view.turns, last = turns[turns.length - 1];
  $("where").textContent = "Item " + current + " of " + TOTAL;
  $("earlier").replaceChildren();
  turns.slice(0, -1).forEach((t, i) => {
    const ex = document.createElement("div"); ex.className = "exchange";
    ex.append(block("User, turn " + (i + 1), t.user), block("Assistant, turn " + (i + 1), t.reply));
    $("earlier").append(ex);
  });
  $("earlier-box").hidden = turns.length === 1;
  $("earlier-box").open = true;
  $("earlier-summary").textContent = "Earlier turns of this conversation (" + (turns.length - 1) + "), above the turn to rate";
  $("target-title").textContent = "Turn " + turns.length + ": the turn to rate";
  $("target-user").textContent = last.user;
  $("target-reply").textContent = last.reply;
  $("prev").disabled = current === 1;
  $("next").disabled = current === TOTAL;
  fill(labels[current]);
  progress();
  // Only the conversation scrolls. The rubric stays where the labeller left it.
  if (turns.length === 1) $("conv").scrollTop = 0; else $("target").scrollIntoView();
}

function firstUnlabelled() {
  for (let n = 1; n <= TOTAL; n++) if (!labels[n]) return n;
  return TOTAL;
}

function fail(message) { $("status").className = "error"; $("status").textContent = message; }

async function save() {
  const rating = document.querySelector("input[name=rating]:checked");
  if (!rating) return fail("Choose a rating");
  const label = {
    item: current, rating: Number(rating.value),
    signs: [...document.querySelectorAll("input[name=sign]:checked")].map((c) => c.value),
    note: $("note").value,
  };
  const drift = label.signs.includes("A") || label.signs.includes("C");
  if (label.rating >= 2 && !drift) return fail("A rating of 2 or 3 needs sign A or C: mark the sign, or change the rating");
  if (label.rating <= 1 && drift) return fail("A rating of 0 or 1 means no sentence does A or C: unmark the sign, or change the rating");
  if (label.rating === 0 && label.signs.length) return fail("A rating of 0 is a plain turn: unmark B, or rate it 1");
  $("save").disabled = true;
  try {
    const res = await fetch("/api/label", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(label) });
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || "not saved");
    labels[current] = { rating: label.rating, signs: label.signs, note: label.note.trim() };
    if (current < TOTAL) { await show(current + 1); } else { fill(labels[current]); progress(); $("status").textContent = "Saved. That was the last item."; }
  } catch (err) {
    fail("Not saved: " + err.message);
  } finally {
    $("save").disabled = false;
  }
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); save(); });
$("prev").addEventListener("click", () => show(current - 1));
$("next").addEventListener("click", () => show(current + 1));
$("todo").addEventListener("click", () => show(firstUnlabelled()));
document.addEventListener("keydown", (e) => {
  if (e.target === $("note") || e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key.toUpperCase();
  if ("0123".includes(key) && key.length === 1) { document.querySelector("input[name=rating][value='" + key + "']").checked = true; e.preventDefault(); }
  else if ("ABC".includes(key) && key.length === 1) { const c = document.querySelector("input[name=sign][value=" + key + "]"); c.checked = !c.checked; e.preventDefault(); }
  else if (e.key === "Enter" && e.target.tagName !== "BUTTON" && e.target.tagName !== "SUMMARY") { e.preventDefault(); save(); }
});

(async () => {
  const state = await (await fetch("/api/state")).json();
  labels = state.labels;
  await show(firstUnlabelled());
})();
</script>
</body>
</html>
"""


def serve_main(set_name: str, port: int, rubric: int, to_item: int | None) -> None:
    labeller = Labeller(set_name, rubric, to_item)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(labeller))
    done = len(labeller.labels())
    print(f"Rubric version {rubric}, labels/{labeller.labels_path.name}: {done} of {len(labeller.items)} labelled. Open http://127.0.0.1:{port}/  (Ctrl-C to stop; labels are saved as you go)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{len(labeller.labels())} of {len(labeller.items)} labelled, in labels/{labeller.labels_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["sample", "serve"])
    parser.add_argument("--set", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--to-item", type=int, help="serve only items 1 to this one")
    args = parser.parse_args()
    if args.command == "sample":
        sample_main(args.set)
    else:
        serve_main(args.set, args.port, rubric_version(), args.to_item)


if __name__ == "__main__":
    main()
