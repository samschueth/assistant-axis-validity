# One command per stage. Seeds are fixed in configs/study.toml; see DATA_AND_MODELS.md.
.PHONY: help setup fetch-vectors gate1 simulate generate project judge analyze figures test safety-check

PY = .venv/bin/python

help:
	@echo "setup         pinned laptop environment"
	@echo "fetch-vectors download the released axis and role vectors (pinned revision)"
	@echo "gate1         gate 1: released axis vs first principal component of the role vectors"
	@echo "simulate      sample-size simulation on synthetic data"
	@echo "generate      GPU: conversation generation and activation capture"
	@echo "project       per-turn projections, target and off-target"
	@echo "judge         LLM judge and its validation against human labels"
	@echo "analyze       AUCs, off-target distributions, gaps"
	@echo "figures       regenerate every figure used in WRITEUP.md"
	@echo "test          unit tests"
	@echo "safety-check  pre-publication scan (needs a local term list, see below)"

NOT_IMPL = @echo "not implemented yet: see PREREGISTRATION.md; this target is a placeholder" && exit 1

# Python 3.12. Uses uv when it is installed (it can fetch 3.12 itself); otherwise python3.12.
setup:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements.txt; \
	else \
		python3.12 -m venv .venv && $(PY) -m pip install -r requirements.txt; \
	fi

fetch-vectors:
	$(PY) src/load_axis.py --fetch

gate1:
	$(PY) src/gate1_axis_pc1.py

simulate:
	$(PY) src/analysis/sample_size_sim.py

generate:  ; $(NOT_IMPL)
project:   ; $(NOT_IMPL)
judge:     ; $(NOT_IMPL)
analyze:   ; $(NOT_IMPL)
figures:   ; $(NOT_IMPL)

test:
	$(PY) -m pytest -q tests/

# Pre-publication scan. The term list is deliberately NOT committed: writing the
# excluded names into this repo would be the very leak the check exists to prevent.
# Put one case-insensitive pattern per line in .safety-terms (gitignored), or point
# SAFETY_TERMS at a file outside the repo.
SAFETY_TERMS ?= .safety-terms
safety-check:
	@test -f "$(SAFETY_TERMS)" || { echo "no term list at $(SAFETY_TERMS): see the comment in this Makefile"; exit 2; }
	@echo "scanning tracked files against $(SAFETY_TERMS)..."
	@git ls-files -z | xargs -0 grep -rinIE -f "$(SAFETY_TERMS)" || echo "clean: no matches"
