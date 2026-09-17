# Speech Sentiment Analyzer

Sentiment (positive / neutral / negative) from **raw speech audio** — not from a transcript.

Built for a take-home assignment. The interesting question is not "what accuracy?" but
**"does the model hear the tone, or is it just reading the words?"** — so the evaluation is
built around deliberately incongruent speech, where what is said and how it is said disagree.

## Status

🚧 Scaffolding complete, implementation in progress. See `docs/TASKS.md`.

Done: core types, emotion→sentiment mapping, speaker-disjoint splitting, manifest schema,
and the invariant test suite (35 tests).

## Quick start

```bash
make setup     # venv + dependencies
make test      # invariant tests -- no data or API key needed
make data      # download CREMA-D (~470 MB, ODbL)
make all       # train, evaluate, regenerate the HTML report
make demo      # single-clip end-to-end. No API key required.
```

Evaluation runs entirely from committed fixtures. An API key is needed **only** to
regenerate the synthetic sets (`make gen-probe`, `make gen-synthetic`).

## Approach

Three solutions spanning the lexical↔acoustic axis, all behind one interface so the
evaluation harness compares them directly:

| | Solution | Reads |
|---|---|---|
| **A** | Lexical-only — ASR → text sentiment | the words |
| **B** | Acoustic-only — frozen encoder + probe | the tone |
| **C** | Fusion — calibrated late fusion with abstention | both |

Evaluated on four datasets: a public corpus (CREMA-D, speaker-disjoint **and** random splits,
to quantify leakage), a synthetic incongruence set, human recordings, and an unsupervised
probe of the TTS emotion space.

**Headline metric — Prosody Sensitivity Index (PSI):** on clips where words and tone
disagree, the fraction of predictions that follow the *tone*. 1.0 = listens, 0.0 = reads the
transcript.

## Documentation

- `CLAUDE.md` — project rules, invariants, pinned facts
- `docs/ARCHITECTURE.md` — module contracts and the rationale behind each
- `docs/TASKS.md` — ordered implementation plan
- `DESIGN.md` — the design write-up (pending)

## Data & licensing

| Asset | License | Redistributed here? |
|---|---|---|
| CREMA-D | ODbL v1.0 | No — `make data` fetches it |
| Synthetic (Cartesia) | generated, commercial license held | Yes, committed |
| Recordings | authors' own | Yes, committed |
| `audeering` VAD model | **CC-BY-NC-SA-4.0, research only** | No — flagged at runtime |
| WavLM, faster-whisper | MIT | No |

The strongest acoustic model is non-commercial. That is surfaced in the CLI and the report
rather than buried: a permissive backend is provided alongside it.
