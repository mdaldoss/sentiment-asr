# Implementation tasks

Ordered. Each task is self-contained: it states the goal, the files, the acceptance
criteria, and the traps. Do them in order — later tasks assume earlier ones.

**Before any task:** read `CLAUDE.md` (rules + pinned facts) and the relevant section of
`docs/ARCHITECTURE.md` (contracts). Run `uv run pytest` before and after; it must stay green.

**Style reference:** `ssa/types.py`, `ssa/mapping.py`, `ssa/splits.py`, `ssa/manifest.py` are
finished and reviewed. Match their style: module docstring explaining *why*, `from __future__
import annotations`, full type hints, dataclasses for data, explicit exceptions with useful
messages, `logging` not `print`, rationale comments on non-obvious choices.

---

## T1 — Audio loading  ·  `ssa/audio.py`

Load any audio file to a validated `AudioClip`: mono, float32, 16 kHz.

- `load_clip(path: Path, clip_id: str | None = None) -> AudioClip`
- `load_many(paths: Sequence[Path]) -> list[AudioClip]`
- Resample with `librosa.resample` **once**, at load. Downmix stereo by channel mean.
- Peak-normalise to avoid clipping, but **do not** loudness-normalise: loudness is a
  prosodic cue this project measures. Removing it would erase part of the signal.

**Accept:** round-trips a generated sine wav; raises on a missing file; a 44.1 kHz stereo
input comes back mono 16 kHz.

---

## T2 — CREMA-D ingest  ·  `scripts/fetch_cremad.py`

Download, extract, and emit `data/cremad/manifest.csv`.

- Source: `https://huggingface.co/datasets/myleslinder/crema-d/resolve/main/data/crema_d.tar.gz`
  (470 MB, ungated, ODbL). Cache the tarball; skip re-download if present and complete.
- Filename format `1001_DFA_ANG_XX.wav` → `speaker_id="1001"`, sentence code `DFA`,
  emotion `ANG`, intensity `XX`.
- Map the 12 sentence codes to their text (they are fixed and emotionally neutral).
  Therefore `text_sentiment = NEUTRAL` for **every** CREMA-D row.
- `prosody_sentiment = map_emotion(tag, "crema_d")`; `emotion_tag` keeps the raw code.
- Apply `speaker_disjoint_split(seed=0)`. Write a **second** manifest
  `manifest_randomsplit.csv` using `random_split(seed=0)` for the leakage comparison.

**Accept:** 7,442 rows; 91 unique speakers; `validate_manifest` passes; both manifests exist;
`assert_speaker_disjoint` passes on the first and fails on the second.

**Trap:** because all CREMA-D text is neutral, every non-neutral clip is *mildly* incongruent
by our definition. Note it in the report — do not "fix" it.

---

## T3 — Metrics  ·  `ssa/eval/metrics.py`

`uar`, `macro_f1`, `expected_calibration_error`, `psi_contested`, `psi_strict`,
`confusion_matrix` (in `Sentiment.ordered()` order).

- `psi_contested`: over incongruent clips where prediction ∈ {prosody label, text label},
  fraction equal to the prosody label. **Chance = 0.5.** This is the headline.
- `psi_strict`: over *all* incongruent clips, fraction equal to the prosody label.
  **Chance = 1/3.** Reported beside it, because `psi_contested` ignores third-label
  predictions and would flatter a model that mostly predicts the third class.
- Both return `float("nan")` — never `0.0` — when undefined.

**Accept:** hand-computed fixtures in `tests/test_metrics.py`: an all-prosody predictor gives
PSI 1.0, an all-text predictor 0.0, and the empty case gives nan (use `math.isnan`).

---

## T4 — Acoustic solution, permissive backend  ·  `ssa/solutions/acoustic.py`

`microsoft/wavlm-base`, frozen, + a trained probe.

- Extract mean+std pooled hidden states from the last layer. Cache embeddings to
  `data/cache/embeddings_{model}_{sha}.npz` — extraction over 7.4k clips on CPU is slow and
  will be re-run many times.
- Probe: `sklearn` logistic regression (multinomial) first. Only try a shallow MLP if LR
  underperforms, and report both.
- Train on `split=="train"`, tune on `"val"`, never touch `"test"` until the final run.
- **Call `assert_speaker_disjoint` at the top of training.**

**Accept:** trains on CPU in under ~10 min given cached embeddings; speaker-independent UAR
on CREMA-D test lands roughly in 0.45–0.70. **Above 0.90 means a bug** — almost certainly
leakage or test contamination. Investigate, do not celebrate.

---

## T5 — Lexical solution  ·  `ssa/solutions/lexical.py`

`faster-whisper` small int8 → text sentiment → `Prediction`.

- Always populate `transcript`. Empty transcript → abstain with uniform probs.
- Device: `cpu` with `compute_type="int8"`. **CTranslate2 has no Apple MPS support** — do not
  add an `mps` branch, it will fail at runtime.
- Text classifier: a small HF sentiment model is fine; record its id in the `solution` string.

**Accept:** transcribes a known clip correctly; latency recorded; on E2's incongruent clips
this solution should score **near-zero PSI**. That is the expected, desired result — it is the
control condition proving the measurement works.

---

## T6 — Eval harness  ·  `ssa/eval/runner.py`

`evaluate(solution, manifest, *, dataset, split_type) -> EvalResult`, serialised to
`results/{solution}__{dataset}__{split_type}.json`.

- Include: all metrics, confusion matrix, abstention rate, latency p50/p95, per-clip
  predictions, git SHA, timestamp, n_clips.
- **Every number in the report must come from one of these files.** `report.py` renders, it
  never recomputes.

**Accept:** JSON round-trips; re-running with the same seed gives identical numbers.

---

## T7 — Research backend + fusion  ·  `acoustic.py`, `ssa/solutions/fusion.py`

- Research backend: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` → VAD.
  Fit the two valence thresholds on **val only**. Populate `Prediction.vad`.
  **The CLI must print a CC-BY-NC-SA-4.0 research-only notice when this backend loads.**
- Fusion: `softmax(w_a·log p_A + w_b·log p_B)`, weights fitted on val, then temperature
  calibration. Abstain below a val-tuned confidence threshold.
- Keep fusion **late**, not early. Early fusion would score slightly better and would destroy
  the per-branch PSI analysis, which is the point of the project.

**Accept:** fusion ≥ both branches on congruent data; PSI sits between the two branches.

---

## T8 — D0 emotion-space probe  ·  `scripts/gen_emotion_probe.py`

Generate all 58 Cartesia tags × 2 carrier sentences ≈ 116 clips. **Needs `CARTESIA_API_KEY`.**

- Use `all_cartesia_tags()` — ambiguous tags **included**, D0 is unsupervised.
- Embed each clip with the **audeering VAD model** (independent of anything we trained),
  scatter in valence-arousal space, cluster (k-means with silhouette-chosen k).
- Output `results/d0_emotion_space.json`: per-tag VAD, cluster assignment, silhouette.

**Why this matters:** Cartesia's docs call emotion tags *"guidance rather than strict
adjustments"* and name only six as giving best results. This measures how many of the 58 are
actually acoustically distinct. **Run this before T9** — its output decides which tags E2 may
use as prosody labels.

---

## T9 — E2/E4 synthetic sets  ·  `scripts/gen_synthetic.py`

- **E2** (eval): 3 text-sentiments × 3 prosodies × 5 carriers × 2 voices = 90 clips.
  Diagonal 30 congruent, off-diagonal 60 incongruent.
- **E4** (`--augment`, needs Cartesia Pro): ~1–2k clips for the augmentation arm.
- Pick prosody tags from T8's clusters, not arbitrarily.
- **The labelled vocabulary is imbalanced — 24 negative / 15 positive / 5 neutral.** Sample
  balanced per sentiment; do not iterate the tag list uniformly.
- Commit the WAVs and manifest. Key-gate the script so evaluation runs without a key.

**Accept:** manifest validates; exactly 30 congruent / 60 incongruent; costs ≲ 5 min of audio.

---

## T10 — E3 human recordings  ·  `scripts/record_prompts.py`

Teleprompter: prints a carrier sentence and target tone, records on a keypress, writes the
clip and manifest row. ~40 clips, same crossed design, plus a few deliberately whispered or
breathy clips for T11.

**Accept:** a non-technical user can run `make record` and finish without editing anything.

---

## T11 — Voice health  ·  `ssa/voicehealth.py`, `ssa/paralinguistic.py`

CPPS / HNR / jitter / shimmer via `praat-parselmouth`; eGeMAPS via `opensmile`.
Whisper/breathiness detector with thresholds fitted on the E3 whispered clips.

**Build the extraction and the detector. Do NOT build the per-speaker longitudinal baseline
gate** — we have no longitudinal data, and shipping it would imply a validation we do not
have. Document the design in `DESIGN.md` and label it clearly as unvalidated.

---

## T12 — Report + docs  ·  `ssa/report.py`, `DESIGN.md`, `README.md`

**Load the `dataviz` skill before writing any chart code.**

- `report/index.html`, self-contained, no CDN at view time (must open offline).
- Renders only from `results/*.json`.
- Anything reasoned-but-not-measured gets an explicit visual marker plus the words
  "argued from literature, not measured in this study".
- `DESIGN.md` is the 1–2 page doc the brief asks for; the dashboard is its companion, not a
  replacement.

**Accept:** `make all` from a clean clone reproduces every table and the HTML; `make demo`
runs with no API key.
