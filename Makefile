# Speech Sentiment Analyzer
# Evaluation targets need NO API key -- fixtures are committed.
# Only the `gen-*` targets require CARTESIA_API_KEY.

.PHONY: help setup test lint data train eval report demo all record gen-probe gen-probe-d1 gen-synthetic clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies (record works out of the box; `generate` needs a key, opt in separately)
	uv venv --python 3.11
	uv pip install -e ".[dev,record]"

test:  ## Run the test suite (invariants)
	uv run pytest -q

lint:  ## Lint and type-check
	uv run ruff check ssa tests scripts
	uv run ruff format --check ssa tests scripts

data:  ## Download CREMA-D and build manifests (~470 MB)
	uv run python scripts/fetch_cremad.py

train:  ## Train the acoustic probe (speaker-disjoint split). BACKEND=permissive|research
	uv run python -m ssa.train --backend $${BACKEND:-permissive}

eval:  ## Evaluate every solution on every dataset -> results/*.json
	uv run python -m ssa.eval --suite all

report:  ## Regenerate report/index.html from results/*.json
	uv run python -m ssa.report

demo: data  ## End-to-end demo on a CREMA-D clip (fetches it first if needed). No API key needed.
	uv run python -m ssa.cli --audio data/cremad/AudioWAV/1001_DFA_ANG_XX.wav

all: data train eval report  ## Full pipeline

record:  ## Teleprompter to record the E3 human set
	uv run python scripts/record_prompts.py

gen-probe:  ## [needs CARTESIA_API_KEY + `pip install -e .[generate]`] D0 emotion-space probe
	uv run python scripts/gen_emotion_probe.py

gen-probe-d1:  ## [needs CARTESIA_API_KEY + `pip install -e .[generate]`] D1 emotion-rendering factorial probe
	uv run python scripts/gen_emotion_probe_d1.py

gen-synthetic:  ## [needs CARTESIA_API_KEY + `pip install -e .[generate]`] E2 incongruence set
	uv run python scripts/gen_synthetic.py

clean:  ## Remove generated artifacts (keeps downloaded data)
	rm -rf results/*.json report/index.html data/cache
