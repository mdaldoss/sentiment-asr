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
make eval-e5   # A/B/C on E5, the Hume incongruence set (words vs delivery) -> results/*.json
make data-zurich  # E6: decode the 6-speaker Zurich recordings -> data/zurich/ + manifest
make eval-zurich  # E6: zero-shot cross-corpus eval + retrain B/D on the new speakers
make demo-web  # [needs .[demo]] live demo at http://127.0.0.1:8000 -- record your own voice
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

A fourth solution, **D — prosodic**, uses no network at all: 96 named measurements of
the waveform (eGeMAPS + Praat pitch/energy contour) into a transparent classifier.
Nothing in its path can represent a word.

Evaluated on CREMA-D (public benchmark, speaker-disjoint **and** random splits to
quantify leakage), a synthetic incongruence set generated via Cartesia TTS, a Hume
Octave incongruence set (E5), **six-speaker human recordings (E6)**, single-speaker
human recordings (E3), and an unsupervised probe of Cartesia's emotion-tag space.

**Headline metric — Prosody Sensitivity Index (PSI):** on clips where words and tone
disagree, the fraction of predictions that follow the *tone*. 1.0 = listens, 0.0 = reads
the transcript. See `ssa/eval/metrics.py` for the exact definition.

**Measured headline result — read both halves of this table.** Left: the 300-clip
stratified CREMA-D test subset. Right: E6, 160 clips from **eight real speakers** the
models have never heard (`make data-zurich && make eval-zurich`), with 95% bootstrap
intervals. Chance is 0.333 for UAR, 0.500 for PSI.

| | CREMA-D UAR | CREMA-D PSI | **E6 UAR [95% CI]** | **E6 PSI [95% CI]** |
|---|---|---|---|---|
| A — Lexical | 0.360 | 0.090 | 0.314 [0.25, 0.39] | **0.104 [0.05, 0.17]** |
| B — Acoustic (permissive) | **0.797** | 0.920 | 0.396 [0.33, 0.46] | 0.584 [0.47, 0.70] |
| B — Acoustic (research) | — | — | 0.356 [0.29, 0.43] | **0.366 [0.27, 0.47]** |
| C — Fusion | 0.797 | 0.891 | **0.402 [0.34, 0.47]** | 0.557 [0.44, 0.67] |
| D — Prosodic | 0.566 | 0.982 | 0.333 [0.33, 0.33] ⚠ | 0.521 [0.41, 0.64] |

On CREMA-D the result has exactly the shape the design predicts: lexical-only is barely
above chance and structurally can't sense tone (CREMA-D's text is always neutral), and
acoustic-only wins by a wide margin.

**On eight real speakers, only the fusion clears chance on UAR — by 0.003.** The backend
that scores 0.797 on the benchmark scores 0.396 [0.33, 0.46] on laptop-microphone audio.
⚠ Solution D predicts a single class for every E6 clip. This is the project's most
important result and it is a negative one: **nothing fitted on CREMA-D transfers**, and
the earlier one-speaker and synthetic transfer numbers were flattering.

What does survive is sharper than the accuracy numbers. The lexical floor holds — PSI
0.104, an interval entirely below chance, meaning a transcript-only model follows the
transcript (also the check that E6's hand-assigned text labels are sound). And **the
research backend's PSI interval sits entirely below chance too (0.366 [0.27, 0.47])**: an
"acoustic" model that follows the *words*, from audio alone, with no transcript in its
path — replicating on eight real voices what E5 found on synthetic ones, now with a
confidence interval behind it. Note the permissive backend's own PSI interval contains
0.5, so it is not established as prosody-following either; only the research backend's
failure is statistically clean.

See `DESIGN.md` → **E6** for the full account, including a leakage bug caught mid-analysis
(a combo trained on its own validation speaker and returned UAR 1.000) and a claim that
reversed when the dataset grew — training on 80 matched clips appeared to beat 5,235
acted ones on the 6-speaker build and did not survive the 8-speaker one.

## Status

**Live demo**: `make demo-web` → record your own voice at `http://127.0.0.1:8000` and see
both acoustic backends and the lexical control score it, plus where it lands on the
valence–arousal plane. It opens on a bundled clip (positive words, flat delivery) where
they already disagree.

**E5, the sharpest result** (`make gen-hume-e5`, `make eval-e5`): 90 synthetic clips, 60
with words and delivery deliberately contradicting. The lexical control scores PSI 0.000
(follows the words every time — the floor behaving correctly). On *identical audio* the
permissive backend follows the tone (PSI 0.682) while the research backend follows the
**words** (0.211, below chance) despite never seeing a transcript — an empirical
confirmation of the audeering paper's own caveat that its valence performance draws on
implicit linguistic information. "Acoustic" does not automatically mean prosodic.

Core pipeline complete and tested (368 tests). CREMA-D benchmark results are real,
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
| Recordings (E6, Zurich) | participants' own | Yes — 120 clips, 6 speakers, committed (source WebM/m4a + decoded 16 kHz WAV) |
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
