# Speech Sentiment Analyzer

Sentiment (positive / neutral / negative) from **raw speech audio** — not from a transcript.

The interesting question isn't "what accuracy?" but **"does the model hear the tone, or
is it just reading the words?"** — so the evaluation is built around deliberately
incongruent speech, where what is said and how it is said disagree. See `DESIGN.md`
for the full write-up, or open **[`index.html`](index.html)** for a page that links to
everything (architecture, results dashboard, listening pages, docs).

## Quick start

```bash
make setup     # venv + dependencies
make test      # invariant tests -- no data, model download, or API key needed
make data      # download CREMA-D (~470 MB, ODbL) -- one-time
make train     # fit the acoustic probe(s). BACKEND=permissive (default) | research
make demo      # single real clip, end-to-end. No API key needed.
make eval-e3   # A/B/C on the recorded human set (both takes) + the D1-vs-human control
make listening-sorted  # report/listening_sorted.html -- D1+E3 clips ranked, for spot-checking
make prosody-samples   # extract VAD/F0 for 20 real samples (CREMA-D + E3) -> results/prosody_samples.json
make probe-hume        # [needs HUME_API_KEY] falsification test of the D1 Cartesia finding
make eval-backend-combos  # WavLM+probe vs audeering across 4 training-data combos (needs make data + make probe-hume first)
make report    # regenerate report/index.html from results/*.json
make site      # regenerate /index.html and /architecture.html (the entry point + pipeline page)
```

`make demo` downloads CREMA-D on first run if needed, then runs the recommended
(fusion) solution on one real clip. Expect the first run of any command to be slow —
it downloads pretrained models (WavLM, faster-whisper, the audeering VAD model) from
Hugging Face; they cache locally afterward. Model *loading* dominates a single CLI
invocation's latency (~20s) far more than inference itself does — see `DESIGN.md`'s
limitations for what that means for a real deployment.

Evaluation runs entirely from committed fixtures and the two committed trained
artifacts (`data/cache/probe_permissive.joblib`, `data/cache/thresholds_research.json`)
— no API key needed for any of the above. A key is needed **only** to *regenerate* the
synthetic datasets:

```bash
cp .env.example .env   # fill in CARTESIA_API_KEY
make gen-probe         # D0: Cartesia emotion-space probe
make gen-synthetic      # E2: synthetic incongruence set
```

## CLI

```bash
uv run python -m ssa.cli --audio path/to/clip.wav                    # recommended (fusion)
uv run python -m ssa.cli --audio clip.wav --solution acoustic        # acoustic-only
uv run python -m ssa.cli --audio clip.wav --solution acoustic --backend research  # CC-BY-NC-SA-4.0, research use only
uv run python -m ssa.cli --audio clip.wav --json                     # machine-readable output
```

## Approach

Three solutions spanning the lexical↔acoustic axis, all behind one interface so the
evaluation harness compares them directly:

| | Solution | Reads | License |
|---|---|---|---|
| **A** | Lexical — ASR (faster-whisper) → text sentiment | the words | MIT |
| **B** | Acoustic — frozen encoder + trained probe | the tone | permissive (WavLM, MIT) or research (audeering, CC-BY-NC-SA-4.0) |
| **C** | Fusion — calibrated late fusion + abstention | both | recommended default |

Evaluated on CREMA-D (public benchmark, speaker-disjoint **and** random splits to
quantify leakage), a synthetic incongruence set generated via Cartesia TTS, human
recordings, and an unsupervised probe of Cartesia's emotion-tag space.

**Headline metric — Prosody Sensitivity Index (PSI):** on clips where words and tone
disagree, the fraction of predictions that follow the *tone*. 1.0 = listens, 0.0 = reads
the transcript. See `ssa/eval/metrics.py` for the exact definition.

**Real, measured headline result** (300-clip stratified CREMA-D test subset):

| | UAR | PSI<sub>contested</sub> |
|---|---|---|
| A — Lexical | 0.360 | 0.090 |
| B — Acoustic | **0.797** | 0.920 |
| C — Fusion | 0.797 | 0.891 |

Exactly the shape the design predicts: lexical-only is barely above chance and
structurally can't sense tone (CREMA-D's text is always neutral); acoustic-only, the
only one that can actually hear the emotion, wins by a wide margin. See `DESIGN.md`
and `report/index.html` for the full results, including a real leakage measurement,
a domain-gap finding from the D0/D1 Cartesia probes, and the E3 human-recording
control that shows Solution B doesn't transfer cleanly to a new speaker either.

## Status

Core pipeline complete and tested (267 tests). CREMA-D benchmark results are real,
measured end-to-end. **E3 (human recordings) is recorded and evaluated** — two
independent takes, 27 clips each, `make eval-e3` — and doubles as a control that rules
the measuring instruments out as the explanation for a headline negative finding: **D1
measured Cartesia's emotion tags as not reliably audible on this content** (below-chance
intended-emotion recoverability), replicated with a control showing the same instruments
read real human speech at 2x+ chance. **A same-design Hume Octave probe
(`make probe-hume`) confirms this is vendor-specific, not a synthetic-TTS-wide limit** —
its `description` field produces real differentiation on the identical carrier text.
See `report/listening_sorted.html` (`make listening-sorted`) to listen to the ranked
clips yourself, and `report/index.html` for which model actually extracts
prosody/emotion (with papers) plus a VAD plot over 20 real samples. **A 4-combo
backend comparison** (`make eval-backend-combos`) trains/evaluates WavLM+probe vs
audeering across CREMA-D alone / +E3 / +E3+Hume / +Hume: adding 54–74 non-CREMA-D
clips to 5,235 moves CREMA-D-test UAR by ≤0.003 (noise, not signal) and the
zero-shot research backend is invariant by construction — see `report/index.html`.
Synthetic incongruence
set (E2) generation is **76/90 clips** complete and superseded for the audibility
question by D1's finding — kept as the historical exhibit. See `DESIGN.md` → Known
limitations for the complete, honest accounting, and `report/index.html` for whatever
has landed most recently.

## Documentation

- `report/overview.html` — **start here**: state of the art, what we built, what we found, and
  the prioritised list of what would make it better. `make overview`
- `index.html` — top-level entry point, links to everything below, regenerate with `make site`
- `architecture.html` — pipeline diagram + which model extracts prosody/emotion, `make site`
- `DESIGN.md` — the design write-up: approach, trade-offs, evaluation, limitations, next steps
- `CLAUDE.md` — project rules, invariants, pinned facts (for anyone extending this)
- `docs/ARCHITECTURE.md` — module-by-module contracts and rationale
- `docs/TASKS.md` — the ordered implementation plan this was built against
- `report/index.html` — live results dashboard, regenerate with `make report`

## Data & licensing

| Asset | License | Redistributed here? |
|---|---|---|
| CREMA-D | ODbL v1.0 | No — `make data` fetches it |
| Synthetic (Cartesia) | generated | Yes — 76/90 E2 clips + 45 D1 clips committed |
| Recordings (E3) | authors' own | Yes — 2 takes, 30 clips each, committed |
| `audeering` VAD model | **CC-BY-NC-SA-4.0, research only** | No — flagged at runtime, in the CLI, and in the report |
| WavLM, faster-whisper, text classifier | MIT / Apache-2.0 | No — downloaded, cached locally |

The strongest acoustic model (research backend) is non-commercial. That's surfaced
loudly — CLI warning, README, report — rather than buried: a fully permissive backend
ships alongside it as the deployable default.

## Testing

```bash
make test              # fast, network-free (default)
uv run pytest -m network   # exercises real models -- slower, needs downloads
```
