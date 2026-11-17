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

**Running total: about $5.75.** Every instance has been deleted; nothing is billing.

**Projection for the main run (2026-11-17, before it starts).** 70 conversations, 1,050 turns,
ceiling 1,536 tokens, on two A100 80 GB instances in parallel at $2.09/h each. Generation speed
in the length probe was 11.2 new tokens per second. If replies average 520 tokens (the probe's
mean) the GPU time is about 16 h in total, about $33; if they average 700 (the probe's early
turns), about 21 h, about $44. Simulated user about $5 to $7 with the conversation cached; judge
about $14 to $19 at the unverified list price. **Projected study total: about $58 to $76**
against the cap of $100.
