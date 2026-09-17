# CLAUDE.md — Speech Sentiment Analyzer

Guidance for any agent working in this repo. Read this before writing code.

## What this project is

A pipeline that classifies **sentiment (positive / neutral / negative) from raw speech
audio** — not from a handed-over transcript. It is a take-home assignment deliverable for
Domera Labs, who build *Ami*, a voice-first AI companion for seniors.

The point of the project is **not** to maximise accuracy. It is to show sound ML judgment:
honest evaluation, explicit trade-offs, and a clear account of what the system cannot do.
Code that inflates a number at the cost of honesty is a failure, not a win.

## The one idea everything serves

An audio sentiment model can cheat by reading the **words** instead of hearing the **tone**.
Published work confirms this is the normal failure mode (arXiv 2510.10444, arXiv 2510.25054).
Ami has no screen and no camera, so tone is the only signal it has. Therefore this repo is
built to **measure whether a model hears prosody**, via deliberately incongruent speech where
the words say one thing and the delivery says another.

Every design decision below exists to keep that measurement trustworthy.

## Hard rules — do not violate these

1. **Never evaluate on a speaker who appears in training.** All splits are speaker-disjoint.
   `ssa/splits.py` enforces this and a test asserts it. If you add a dataset, add its
   speaker IDs — do not fall back to a random split.
2. **The gold label for evaluation is `prosody_sentiment`, never `text_sentiment`.**
   The whole project is about the tone. Getting this backwards silently inverts every result.
3. **Never silently bucket an ambiguous emotion.** `surprised`, `nostalgic`, `sarcastic`,
   `mysterious`, `determined`, `calm` do not have an agreed sentiment valence. They are
   **excluded** from labelled sets and that exclusion is documented. Do not "fix" this by
   assigning them a label.
4. **Never report a metric without its evaluation condition.** A number without its split
   type and dataset is meaningless here. Every result carries `dataset`, `split_type`.
5. **Do not commit API keys.** Keys come from `.env` (gitignored). Generation scripts are
   optional and key-gated; evaluation must run with no key from committed fixtures.
6. **Do not claim measured what was only reasoned.** Several roadmap claims (presbyphonia
   confound, per-speaker baselining) are arguments from literature, not results from our
   data. They must be labelled as such in every output. This is the single most important
   honesty rule in the project.
7. **Accuracy is not the headline metric.** Use **UAR** (unweighted average recall) because
   the classes are imbalanced. Report macro-F1 alongside. Plain accuracy may appear only
   next to UAR, never alone.
8. **Never train on a dataset and then present within-corpus numbers as generalisation.**
   Cross-corpus and incongruent results are the honest ones and must be shown beside.

## Pinned facts — do not re-derive or "correct" these from memory

These were verified by web research in Sept 2026. If something contradicts them, check
before changing.

| Thing | Value |
|---|---|
| CREMA-D source | HF `myleslinder/crema-d`, `data/crema_d.tar.gz`, 470 MB, ungated, **ODbL** |
| CREMA-D emotions | ANG, DIS, FEA, HAP, NEU, SAD (6; **no surprise** — maps cleanly) |
| CREMA-D speakers | 91 actors, speaker id is the filename prefix (e.g. `1001_DFA_ANG_XX.wav`) |
| CREMA-D text | 12 fixed, emotionally **neutral** sentences → `text_sentiment = NEUTRAL` always |
| Acoustic model (research) | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` → valence/arousal/dominance in ~[0,1]. **CC-BY-NC-SA-4.0, research only** |
| Acoustic model (permissive) | `microsoft/wavlm-base` frozen + our trained probe. MIT |
| ASR | `faster-whisper`, small, int8. MIT. **CTranslate2 has NO Apple MPS support** — CPU or CUDA only |
| TTS | Cartesia. `generation_config.emotion` or SSML `<emotion value="..."/>`. **English only.** No intensity levels |
| Cartesia free tier | 20K credits ≈ 27 min, **API access included** |
| Voice quality | CPPS via `praat-parselmouth` — the one dysphonia measure valid on *continuous* speech |
| SOTA ceiling | <90% on ESD, <78% on IEMOCAP. **If you see >90% UAR speaker-independent, you have a bug**, most likely leakage |

## Architecture in one paragraph

Audio → one of three **solutions** on the lexical↔acoustic axis: **A** lexical-only
(ASR → text sentiment), **B** acoustic-only (frozen encoder + probe; two swappable
backends), **C** fusion of A and B with calibration and abstention. All three implement the
same `Solution` protocol so the evaluation harness treats them identically. The shared
internal representation is the **valence-arousal-dominance** plane: sentiment is a threshold
read-out of valence, which also gives distress quadrants for free and keeps outputs
interpretable. Full contracts in `docs/ARCHITECTURE.md`.

## Code conventions

- Python 3.11+, `uv` for env and running. Type hints everywhere; `from __future__ import annotations`.
- Dataclasses for data, `Protocol` for interfaces. Prefer plain functions over classes with state.
- `pathlib.Path`, never string paths. `numpy` float32 mono at **16 kHz** — resample on load, once.
- Determinism: every random operation takes an explicit `seed`. No global RNG.
- Errors: fail loudly on bad data. Never silently drop clips — count and report them.
- No network calls at evaluation time. Models cache to `~/.cache`; fixtures are committed.
- Logging via `logging`, not `print`, except in CLI output.

## Testing

`uv run pytest`. Tests are not decoration here — they encode the invariants that make the
results trustworthy. At minimum:

- `test_splits.py::test_no_speaker_overlap` — the single most important test in the repo
- `test_mapping.py::test_ambiguous_emotions_excluded` — ambiguous tags raise, never default
- `test_metrics.py::test_psi_*` — PSI on hand-computed fixtures
- `test_manifest.py::test_schema` — every manifest validates against the schema

Prefer small deterministic fixtures over downloading data in tests. Network-dependent tests
are marked `@pytest.mark.network` and skipped by default.

## What "done" looks like

`make all` downloads data, trains, evaluates, and regenerates `report/index.html` — the HTML
dashboard — from the eval JSON. A reviewer clones, runs `make demo`, and it works with no
API key and no contact with the author.
