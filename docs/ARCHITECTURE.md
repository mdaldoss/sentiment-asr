# Architecture & Implementation Spec

Module-by-module contracts. Read `CLAUDE.md` first for the hard rules and pinned facts.

Each section gives: **purpose**, **contract** (signatures), **invariants**, and **rationale**
(why it is this way — so it does not get "simplified" into something wrong).

---

## 0. Core types — `ssa/types.py`

```python
class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL  = "neutral"
    NEGATIVE = "negative"

@dataclass(frozen=True, slots=True)
class AudioClip:
    clip_id: str
    samples: np.ndarray      # float32, mono, shape (n,)
    sr: int                  # ALWAYS 16000 post-load
    path: Path | None = None

@dataclass(frozen=True, slots=True)
class VAD:
    valence: float           # [0,1], higher = more positive
    arousal: float           # [0,1], higher = more activated
    dominance: float         # [0,1]

@dataclass(frozen=True, slots=True)
class Prediction:
    sentiment: Sentiment
    probs: dict[Sentiment, float]   # calibrated, sums to 1.0 ± 1e-6
    confidence: float               # max(probs.values())
    abstained: bool                 # True => sentiment is a fallback, treat as low-trust
    latency_ms: float
    solution: str                   # e.g. "B:audeering"
    vad: VAD | None = None
    transcript: str | None = None
```

**Invariants.** `probs` always has all three keys and sums to 1. `confidence == max(probs)`.
`abstained=True` does not change `sentiment` — callers decide what to do with it.

**Rationale.** `Prediction` is deliberately rich: the eval harness needs `vad` for the
quadrant analysis, `transcript` to diagnose whether solution A's failures are ASR failures,
and `latency_ms` for the CPU benchmark. Do not slim this down.

---

## 1. Manifest schema — `ssa/manifest.py`

Every dataset (CREMA-D, synthetic, recorded) is normalised to one CSV schema. This is the
spine of the project — the eval harness only ever sees manifests.

| Column | Type | Notes |
|---|---|---|
| `clip_id` | str | unique across all datasets |
| `path` | str | relative to repo root |
| `speaker_id` | str | **required**; actor id, TTS voice id, or recorded speaker |
| `source` | str | `crema_d` \| `synthetic` \| `recorded` |
| `text` | str | the spoken words |
| `text_sentiment` | Sentiment | sentiment of the **words** |
| `prosody_sentiment` | Sentiment | sentiment of the **delivery** — **this is the gold label** |
| `emotion_tag` | str \| "" | original label (`ANG`, `dejected`, …) before mapping |
| `voice_id` | str \| "" | TTS voice, empty for human |
| `is_congruent` | bool | `text_sentiment == prosody_sentiment` |
| `split` | str | `train` \| `val` \| `test` \| `""` (eval-only sets) |

```python
def load_manifest(path: Path) -> pd.DataFrame: ...
def validate_manifest(df: pd.DataFrame) -> None:   # raises ManifestError
def write_manifest(df: pd.DataFrame, path: Path) -> None: ...
```

**Invariants.** `validate_manifest` raises if: any column missing; `clip_id` not unique;
`speaker_id` empty; a sentiment value outside the enum; `is_congruent` inconsistent with the
two sentiment columns; a referenced `path` does not exist.

**Rationale.** Separating `text_sentiment` from `prosody_sentiment` is what makes PSI
computable at all. For CREMA-D the 12 carrier sentences are emotionally neutral by design, so
`text_sentiment` is always `NEUTRAL` — which makes every non-neutral CREMA-D clip *mildly
incongruent* and is worth noting in the report.

---

## 2. Emotion → sentiment mapping — `ssa/mapping.py`

```python
CREMA_D_MAP: dict[str, Sentiment] = {
    "ANG": NEGATIVE, "DIS": NEGATIVE, "FEA": NEGATIVE, "SAD": NEGATIVE,
    "HAP": POSITIVE, "NEU": NEUTRAL,
}

CARTESIA_MAP: dict[str, Sentiment] = { ... }   # see below
AMBIGUOUS_TAGS: frozenset[str] = frozenset({
    "surprised", "amazed", "nostalgic", "wistful", "sarcastic", "ironic",
    "mysterious", "determined", "calm", "contemplative", "anticipation",
    "curious", "skeptical", "flirtatious",
})

def map_emotion(tag: str, scheme: str) -> Sentiment:
    """Raises AmbiguousEmotionError for tags in AMBIGUOUS_TAGS."""
```

Cartesia tags, grouped (from the ~58 documented values):

- **POSITIVE** — happy, excited, enthusiastic, elated, euphoric, triumphant, content,
  peaceful, serene, grateful, affectionate, trust, proud, confident, sympathetic
- **NEGATIVE** — angry, mad, outraged, frustrated, agitated, threatened, disgusted, contempt,
  envious, sad, dejected, melancholic, disappointed, hurt, guilty, rejected, anxious,
  panicked, alarmed, scared, insecure, resigned, bored, tired
- **NEUTRAL** — neutral, hesitant, confused, distant, apologetic
- **AMBIGUOUS → excluded** — see `AMBIGUOUS_TAGS` above

**Invariants.** `map_emotion` **raises** on an ambiguous or unknown tag. It never returns a
default. A test asserts this.

**Rationale.** Silently bucketing `surprised` as positive (a common shortcut) injects label
noise into exactly the cells the project is measuring. Excluding them costs a little data and
buys a defensible label set. `calm` is ambiguous on purpose: it is positive in a wellbeing
sense but acoustically near-neutral, and conflating those is how you get a model that calls
every quiet senior "content". These excluded tags are still used in **D0** (unsupervised
emotion-space probe), which needs no labels.

---

## 3. Splits — `ssa/splits.py`

```python
def speaker_disjoint_split(
    df: pd.DataFrame, *, test_frac: float, val_frac: float, seed: int,
) -> pd.DataFrame:   # returns df with `split` column populated

def random_split(df, *, test_frac, val_frac, seed) -> pd.DataFrame:
    """LEAKY BY CONSTRUCTION. Exists only to quantify the leakage gap. Never used to
    train the shipped model."""

def assert_speaker_disjoint(df: pd.DataFrame) -> None:   # raises SplitError
```

**Invariants.** After `speaker_disjoint_split`, the speaker sets of train/val/test are
pairwise disjoint. `assert_speaker_disjoint` is called at the top of every training run.

**Rationale.** `random_split` is deliberately kept, clearly named and documented as leaky,
because the headline table compares the two. Do not delete it as "dead code" and do not let
it become the default.

---

## 4. Solutions — `ssa/solutions/`

```python
class Solution(Protocol):
    name: str
    def predict(self, clip: AudioClip) -> Prediction: ...
    def predict_batch(self, clips: Sequence[AudioClip]) -> list[Prediction]: ...
```

### A — `solutions/lexical.py`
ASR (`faster-whisper` small int8) → text sentiment classifier → `Prediction`.
`transcript` is always populated. If ASR returns empty, abstain with uniform probs.

### B — `solutions/acoustic.py`
Frozen encoder + head. Two backends behind one class, selected by config:

- `permissive` — `microsoft/wavlm-base` frozen; mean+std pooled embeddings; our trained
  logistic-regression / shallow-MLP probe. MIT-licensed end to end.
- `research` — `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` → VAD → valence
  thresholds. **CC-BY-NC-SA-4.0, research only** — the CLI must print a license notice when
  this backend is selected.

`vad` is populated by the `research` backend, and by `permissive` only if a VAD head is trained.

### C — `solutions/fusion.py`
Late fusion: `p = softmax(w_a · log p_A + w_b · log p_B)`, weights fitted on the **validation
split only**, then temperature-calibrated. Abstains when `max(p) < threshold` (configurable,
default from validation to hit a target abstention rate).

**Rationale for late over early fusion.** Late fusion keeps the branches independently
measurable, which is the entire point — PSI per branch is the result. Early/joint fusion would
be marginally stronger and would destroy the analysis. This is a deliberate accuracy sacrifice.

---

## 5. Metrics — `ssa/eval/metrics.py`

```python
def uar(y_true, y_pred) -> float                      # unweighted average recall
def macro_f1(y_true, y_pred) -> float
def expected_calibration_error(probs, y_true, n_bins=10) -> float

def psi_contested(preds, df) -> float:
    """PRIMARY. Over incongruent clips where the prediction equals either the prosody
    label or the text label: fraction equal to the prosody label.
    Head-to-head, so CHANCE = 0.5. Undefined (nan) if no contested clips."""

def psi_strict(preds, df) -> float:
    """Over ALL incongruent clips: fraction equal to the prosody label.
    Equivalent to prosody-accuracy on incongruent data. CHANCE = 1/3."""
```

**Report both.** `psi_contested` is the headline because it isolates the lexical-vs-acoustic
decision; `psi_strict` is reported beside it because `psi_contested` ignores third-label
predictions and could flatter a model that mostly predicts the third class.

**Invariants.** Both return `nan`, never 0.0, when undefined. A test covers the empty case.

---

## 6. Voice-health gate — `ssa/voicehealth.py`

```python
@dataclass(frozen=True)
class VoiceQuality:
    cpps: float          # smoothed cepstral peak prominence, dB
    hnr: float
    jitter_local: float
    shimmer_local: float
    avqi: float | None   # composite, if all six components available

def voice_quality(clip: AudioClip) -> VoiceQuality:   # via praat-parselmouth
def is_whispered(vq: VoiceQuality, *, thresholds: Thresholds) -> bool
```

**Scope.** Extraction and the whisper/breathiness demo are **built**. The per-speaker
longitudinal baseline gate is **designed and documented, not built** — we have no
longitudinal data. Do not implement it and then present it as validated.

**Rationale.** CPPS is the one dysphonia measure with demonstrated validity on *continuous*
speech rather than sustained vowels, which is all Ami would ever have. Vocal-tremor markers
(PF0T/PAT) track ageing specifically and are described in the report as the trait-vs-state
discriminator — the separation is **temporal**, not spectral, which is why it needs a
per-speaker baseline over time and cannot be done from one utterance.

---

## 7. Evaluation harness — `ssa/eval/runner.py`

```python
def evaluate(solution: Solution, manifest: pd.DataFrame, *, dataset: str,
             split_type: str) -> EvalResult
```

`EvalResult` serialises to JSON with: solution name, dataset, split_type, n clips, UAR,
macro-F1, accuracy, ECE, psi_contested, psi_strict, confusion matrix, abstention rate,
per-clip predictions, latency p50/p95, and the git SHA + timestamp.

**Invariant.** Every number that reaches the report comes from a JSON file on disk. The
report generator never recomputes a metric — it only renders. This makes results auditable
and the dashboard reproducible.

---

## 8. Report — `ssa/report.py`

Reads `results/*.json`, emits a single self-contained `report/index.html`.
**Load the `dataviz` skill before writing any chart code.** No CDN dependencies at view
time — inline everything so the reviewer can open the file offline.

Sections: headline recommendation → main table (solutions × datasets × {UAR, macro-F1, PSI})
→ leakage exposé (random vs speaker-disjoint) → D0 emotion-space map → augmentation arm →
confusion matrices → calibration → latency → voice-health demo → limitations.

**Invariant.** Anything reasoned-but-not-measured is rendered with an explicit visual marker
and the words "argued from literature, not measured in this study".

---

## 9. Datasets

| ID | What | Size | Built by |
|---|---|---|---|
| **D0** | Cartesia emotion-space probe — all ~58 tags × 2 carriers | ~116 clips | `scripts/gen_emotion_probe.py` |
| **E1** | CREMA-D, speaker-disjoint **and** random splits | 7,442 clips | `scripts/fetch_cremad.py` |
| **E2** | Synthetic incongruence: 3 text-sent × 3 prosody × 5 carriers × 2 voices | 90 clips | `scripts/gen_synthetic.py` |
| **E3** | Human recordings, same crossed design | ~40 clips | `scripts/record_prompts.py` |
| **E4** | Synthetic augmentation training set (Cartesia Pro) | ~1–2k clips | `scripts/gen_synthetic.py --augment` |

D0 runs **first** and decides which tags E2 is built from: tags that do not separate
acoustically are not usable as prosody labels.

---

## Build order for implementation

1. `types.py`, `manifest.py`, `mapping.py`, `splits.py` + their tests ← **foundations, do first**
2. `scripts/fetch_cremad.py` → E1 manifest; verify `assert_speaker_disjoint`
3. `solutions/acoustic.py` permissive backend + probe training
4. `solutions/lexical.py`
5. `eval/metrics.py`, `eval/runner.py` + tests
6. `solutions/acoustic.py` research backend; `solutions/fusion.py`
7. `scripts/gen_emotion_probe.py` (D0) → decides E2 tag set
8. `scripts/gen_synthetic.py` (E2, E4), `scripts/record_prompts.py` (E3)
9. `voicehealth.py`, `paralinguistic.py`
10. `report.py` (load `dataviz` skill first), `DESIGN.md`

Steps 1–5 need no API key and no budget. Do them before spending a credit.
