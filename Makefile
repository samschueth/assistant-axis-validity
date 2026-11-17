# One command per stage. Seeds are fixed in configs/study.toml; see DATA_AND_MODELS.md.
.PHONY: help setup fetch-vectors gate1 gate2 gate3 pilot judge-pilot gate4 length-probe simulate generate project label-sample label judge analyze figures test safety-check

PY = .venv/bin/python

help:
	@echo "setup         pinned laptop environment"
	@echo "fetch-vectors download the released axis and role vectors (pinned revision)"
	@echo "gate1         gate 1: released axis vs first principal component of the role vectors"
	@echo "gate2         GPU: gate 2, memory for one full-length conversation with capture"
	@echo "gate3         GPU: gate 3, the axis separates default replies from persona replies"
	@echo "pilot         GPU + paid API: generate the six pilot conversations"
	@echo "judge-pilot   paid API: rate every pilot turn on the draft rubric"
	@echo "gate4         gate 4: base rate of judged drift in the pilot"
	@echo "length-probe  GPU: reply lengths with the cap out of the way, and memory at long contexts"
	@echo "simulate      sample-size simulation on synthetic data"
	@echo "generate      GPU + paid API: the main run, conversation generation and activation capture"
	@echo "project       per-turn projections, target and off-target"
	@echo "label-sample  draw the 120 turns for the human labels (fixed once drawn)"
	@echo "label         label the drawn turns in a browser on this machine"
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

gate2:
	python src/gate2_memory.py

gate3:
	python src/gate3_separation.py

pilot:
	python src/generate.py --briefs configs/pilot_briefs.toml --set pilot

judge-pilot:
	$(PY) src/judge.py --set pilot

gate4:
	$(PY) src/gate4_base_rate.py

length-probe:
	python src/length_probe.py

simulate:
	$(PY) src/analysis/sample_size_sim.py

generate:
	python src/generate.py --briefs configs/main_briefs.toml --set main
project:
	$(PY) src/merge_shards.py --briefs configs/main_briefs.toml --set main
	$(PY) src/project.py --set main
label-sample:
	$(PY) src/label.py sample --set main
label:
	$(PY) src/label.py serve --set main
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
