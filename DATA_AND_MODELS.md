# Data, models and licences

Every model, vector set and dataset used, with its licence. Nothing enters the study that is not
recorded here first.

## Models

| Model | Role | Licence | Notes |
|---|---|---|---|
| `Qwen/Qwen3-32B` | the only model studied | Apache-2.0 | ungated; 64 layers; thinking mode off for every generation |
| simulated-user model | drives the user side of each conversation | API terms | `claude-sonnet-5-5`, effort low; not yet used |
| judge model | rates drift on every assistant turn | API terms | `gpt-6.1-sol`, reasoning effort high; not yet used |

No second model is run this round.

## Vectors

| Source | Use | Licence | Pinned revision |
|---|---|---|---|
| `lu-christina/assistant-axis-vectors` (Hugging Face dataset), `qwen-3-32b/` | the Assistant Axis, the default vector and 275 role vectors, as released | MIT | `3b3b788432ad33e3a28d9ff08e88a530c0740814` |

The axis and role vectors are the authors' released files. No axis is computed here. The files
are downloaded by `make fetch-vectors` into `data/vectors/` (not committed) and loaded with
weights-only loading.

## Code used as a dependency

| Source | Use | Licence | Pinned commit |
|---|---|---|---|
| `github.com/safety-research/assistant-axis` | read for three conventions, listed below | **none stated** | `a98961956072224eaf244eb289d6c01700b63795` |

That repository states no licence. **No code or transcript from it is copied into this
repository, and it is not installed**: it pulls in a serving stack the study does not need. Its
source was read at the pinned commit for the conventions the released vectors depend on, and the
capture code here is written for this study to follow them:

- the target layer for `Qwen/Qwen3-32B` is 32, of 64;
- layer L means the output of transformer block L, zero-indexed;
- a turn's activation is the mean over the reply's tokens in a forward pass over the
  conversation, with thinking mode off.

Its released transcripts are read for orientation only.

## Datasets and transcripts

| Source | Use | Licence / terms | Notes |
|---|---|---|---|
| Newly generated conversations | the whole study | generated here | prompts and seeds recorded in `configs/` |

No production data, telemetry or user content of any kind enters this repository.

## Seeds

All seeds are in `configs/study.toml` under `[seeds]`.

| Stage | Config key |
|---|---|
| random off-target directions | `random_directions` |
| bootstrap | `bootstrap` |
| sample-size simulation | `simulation` |
| sample of turns for human labelling | `label_sample` |
| conversation generation | `generation` |

## Environment

| Stage | File | State |
|---|---|---|
| laptop: gate 1, simulation, analysis, figures, tests | `requirements.txt`, compiled from `requirements.in` | pinned |
| GPU: generation and capture | `requirements-gpu.txt` on image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` || pinned; full freeze from the gate 2 run in `results/gate2_pip_freeze.txt` |

Python 3.12.

## Excluded artifacts and how to regenerate them

| Artifact | Why excluded | Regenerate with |
|---|---|---|
| released vector files | size; they are the authors' files | `make fetch-vectors` |
| model weights | size | downloaded on the GPU instance by `make generate` |
| activation tensors | size | `make generate` |
| per-turn projections (`results/main/projections.npz`, 25 MB) | size | `make project`, from the activation tensors; `results/main/projection_log.json` records what it held |

## Compute and spend

Budget estimate ≈ $50–65; hard cap $100, confirmed by the owner on 2026-11-16.

| Date | Item | Hours / units | Stage | Cost |
|---|---|---|---|---|
| 2026-11-16 | laptop only | n/a | gate 1, simulation | $0 |
| 2026-11-16 | one A100 SXM 80 GB, community cloud, $1.39/h | 0.22 h (799 s) | gate 2 | $0.31 |
| 2026-11-16 | one A100 PCIe 80 GB, secure cloud, $2.09/h | 0.18 h (649 s) | gate 3 | $0.38 |
| 2026-11-17 | one A100 PCIe 80 GB, secure cloud, $2.09/h | 0.93 h | gate 4 pilot: six conversations | $1.95 |
| 2026-11-17 | simulated-user API, `claude-sonnet-5-5` | 90 calls, plus 2 in a test | gate 4 pilot | about $1.12 |
| 2026-11-17 | judge API, `gpt-6.1-sol` | 90 calls, plus 2 in a test | gate 4 pilot | about $0.91 (list price not verified) |
| 2026-11-17 | one A100 PCIe 80 GB, secure cloud, $2.09/h | 0.51 h | length probe (not study data): 30 pilot turns regenerated, memory at long contexts | $1.06 |
| 2026-11-17 | two A100 PCIe 80 GB, secure cloud, $2.09/h each | 2 × 7.7 h (06:50 to 14:32 UTC) | main run, first part: 37 complete conversations and 2 cut short | about $32.20 |
| 2026-11-17 | simulated-user API, `claude-sonnet-5-5` | 581 logged calls | main run, first part | about $7.25 (from logged token counts; price assumed, not verified) |
| 2026-11-17 | two A100 PCIe 80 GB, secure cloud, $2.09/h each | 4.1 h and 6.0 h (16:32 to 20:38 and to 22:32 UTC) | main run, second part: the remaining 31 conversations | about $21.20 |
| 2026-11-17 | simulated-user API, `claude-sonnet-5-5` | 467 logged calls | main run, second part | about $4.60 (from logged token counts; price assumed, not verified) |
| 2026-11-17 | laptop only | about 2 s | projections of the 1,045 main-run turns | $0 |

**Running total: about $71.** Every instance has been deleted; nothing is billing. The judge
has not been run on the main-run turns; at about $17 to $20 (list price not verified) the study
would end at about $88 to $91 against the cap of $100.

**Main run complete, 22:27 UTC, 2026-11-17.** 70 conversations, 1,045 turns: 68 conversations of
15 turns, and the two that were cut short in the first part and stay as they are (`meta-23`, 14
turns; `emotional-04`, 11 turns). One turn has no activation (`task-06` turn 14, a capture that
ran out of memory before the fix in commit `827418e`). 41 turns reached the ceiling of 1,536
tokens; none was empty. The second part ran at commit `247856f` with no capture failure. The
`generation_log.*of2.json` files were written by the second part: their turn counts cover the
whole shard, but the commit, the running time and the simulated-user usage in them are the
second part's only. The package lists of the two parts are identical.

**Main run stopped at 14:27 and 14:30 UTC, 2026-11-17.** The simulated-user account ran out of
credit. Two conversations were in progress and ended there (`meta-23` after 14 turns,
`emotional-04` after 11); the next conversation on each instance failed on its first message,
which stops the run. 31 conversations remained: 3 meta-reflection and 28 emotional disclosure.
They were generated in the second part, after credit was added.

**What the projection below got wrong.** Replies in the main run are longer than in the length
probe, whose contexts held the pilot's cut-off replies: task conversations averaged about 2,060
seconds each, meta-reflection about 1,110. The first part of the run cost about $39.45 for 39
conversations. At the meta-reflection rate the remaining 31 need about 10 GPU hours (about $21)
and about $6 of simulated-user calls; with the judge at about $20 the study total would be about
$92.

**Projection for the main run (2026-11-17, before it starts).** 70 conversations, 1,050 turns,
ceiling 1,536 tokens, on two A100 80 GB instances in parallel at $2.09/h each. Generation speed
in the length probe was 11.2 new tokens per second. If replies average 520 tokens (the probe's
mean) the GPU time is about 16 h in total, about $33; if they average 700 (the probe's early
turns), about 21 h, about $44. Simulated user about $5 to $7 with the conversation cached; judge
about $14 to $19 at the unverified list price. **Projected study total: about $58 to $76**
against the cap of $100.
