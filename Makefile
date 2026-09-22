# Speech Sentiment Analyzer
# Evaluation targets need NO API key -- fixtures are committed.
# Only the `gen-*` targets require CARTESIA_API_KEY.

.PHONY: help setup test lint data train eval eval-e3 eval-backend-combos listening-sorted prosody-samples probe-hume report overview site demo-web gen-hume-e5 eval-e5 data-zurich eval-zurich eval-research-cremad train-combos train-prosodic eval-prosodic compare-representations normalise-speaker normalise-skew demo all record gen-probe gen-probe-d1 gen-synthetic clean

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

eval-e3:  ## Evaluate A/B/C on both E3 recorded takes + the D1-vs-human control -> results/*.json
	uv run python scripts/eval_e3.py

listening-sorted:  ## report/listening_sorted.html -- D1 + E3 clips ranked by detection score
	uv run python scripts/gen_listening_sorted.py

prosody-samples:  ## Extract VAD/F0 for 20 real samples (12 CREMA-D + 8 E3) -> results/prosody_samples.json
	uv run python scripts/plot_prosody_samples.py

probe-hume:  ## [needs HUME_API_KEY + `pip install hume`] Falsification test of the D1 Cartesia finding
	uv run python scripts/probe_hume.py

eval-backend-combos:  ## WavLM+probe vs audeering across 4 dataset combos (cremad/+e3/+hume) -> results/backend_combo_comparison.json
	uv run python scripts/eval_backend_combos.py

report:  ## Regenerate report/index.html from results/*.json
	uv run python -m ssa.report

overview:  ## Regenerate report/overview.html -- the narrative "read this first" report
	uv run python -m ssa.overview

demo-web:  ## [needs `pip install -e .[demo]`] Live demo at http://127.0.0.1:8000 -- record and classify
	uv run python -m demo.app

gen-hume-e5:  ## [needs HUME_API_KEY] Generate E5: the Hume incongruence dataset (words vs delivery)
	uv run python scripts/gen_hume_dataset.py --voices $${VOICES:-2}

eval-e5:  ## Evaluate A/B/C on E5 -> results/*.json (no API key needed)
	uv run python scripts/eval_e5.py

data-zurich:  ## Build E6: decode the Zurich multi-speaker recordings -> data/zurich/ + manifest
	uv run python scripts/build_zurich.py

eval-zurich:  ## E6: zero-shot cross-corpus eval + retrain B/D on the new speakers
	uv run python scripts/eval_zurich.py

eval-research-cremad:  ## audeering backend on CREMA-D's held-out test split, beside the permissive probe
	uv run python scripts/eval_research_cremad.py

train-combos:  ## Permissive probe trained on E5 / E6 / both / +CREMA-D, scored everywhere
	uv run python scripts/train_combos_e5_e6.py

train-prosodic:  ## Fit Solution D: eGeMAPS+contour features -> transparent classifier
	uv run python scripts/train_prosodic.py

eval-prosodic:  ## Score Solution D on CREMA-D, both E3 takes and E5 -> results/*.json
	uv run python scripts/eval_prosodic.py

compare-representations:  ## Prosodic features vs WavLM with the training domain held constant
	uv run python scripts/compare_representations.py

normalise-speaker:  ## Does per-speaker feature normalisation rescue cross-domain transfer?
	uv run python scripts/normalise_speaker.py

normalise-skew:  ## How much of that gain survives a speaker whose clips are NOT class-balanced?
	uv run python scripts/normalise_skew.py

site:  ## Regenerate /index.html and /architecture.html -- the top-level nav pages
	uv run python -m ssa.site

demo: data  ## End-to-end demo on a CREMA-D clip (fetches it first if needed). No API key needed.
	uv run python -m ssa.cli --audio data/cremad/AudioWAV/1001_DFA_ANG_XX.wav

all: data train eval report overview site  ## Full pipeline

record:  ## Teleprompter to record the E3 human set
	uv run python scripts/record_prompts.py

gen-probe:  ## [needs CARTESIA_API_KEY + `pip install -e .[generate]`] D0 emotion-space probe
	uv run python scripts/gen_emotion_probe.py

gen-probe-d1:  ## [needs CARTESIA_API_KEY + `pip install -e .[generate]`] D1 emotion-rendering factorial probe
	uv run python scripts/gen_emotion_probe_d1.py

gen-synthetic:  ## [needs CARTESIA_API_KEY + `pip install -e .[generate]`] E2 incongruence set
	uv run python scripts/gen_synthetic.py

clean:  ## Remove generated artifacts (keeps downloaded data)
	rm -rf results/*.json report/index.html index.html architecture.html data/cache
