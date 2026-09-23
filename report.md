# Speech Sentiment Analyzer

### AI/ML Systems Engineer Take-Home Assignment

## 1. Summary

This project investigates whether sentiment can be inferred from raw speech audio alone, without relying on a transcript. The goal is to build a system that can classify a speaker as positive, neutral, or negative while remaining sensitive to the difference between the words spoken and the prosody used to deliver them.

This matters because the same sentence can convey very different emotional intentions depending on tone. A phrase such as "that's wonderful" can sound warm and sincere, or tired, sarcastic or angry. A useful voice companion should be able to detect when the wording and the delivery disagree, rather than blindly trusting the textual content.

The project compares three system families: a lexical baseline, two acoustic models and an explicit prosody model. The central lesson is not that a model can achieve perfect sentiment classification, but that it must be evaluated speaker-disjoint and on separate datasets that reflect a realistic scenario. In this setting the hardest challenge is not classification accuracy, but generalization to real multi-speaker audio that differs from the training distribution.

### Usage and installation

The code, tests, evaluation scripts, result files, and browser demo are included in this repository. Evaluation uses speaker-disjoint splits throughout.

To see all options, run:

```bash
make help
```

---

## 2. Systems

These models were implemented and evaluated.

0. **Lexical** — Whisper transcription followed by a text sentiment classifier.
1. **Explicit prosody** — handcrafted acoustic features (pitch, loudness, eGeMAPS).
2. **WavLM** — also referred to as the permissive acoustic model.
3. **audEERING** — also referred to as the research acoustic model.

A **late fusion** system combining the lexical and permissive acoustic models, with calibration and abstention, is also evaluated.

### 0 — Lexical model

The lexical model acts as a reference system for measuring whether a model follows prosody or wording. It first transcribes speech with Faster-Whisper Small, then classifies the transcript with CardiffNLP's Twitter RoBERTa sentiment model.

This system is deliberately simple: it provides a lower bound on how much sentiment can be recovered from words alone, and it establishes what "following the text" looks like numerically, so the acoustic models can be compared against it.

### 1 — Explicit prosody model

The explicit prosody model measures named physical quantities of the waveform: 88 openSMILE **eGeMAPS v02** functionals (F0 statistics, loudness, jitter, shimmer, HNR, spectral balance, voiced-segment rates) plus 8 **contour descriptors** computed in Praat — global F0 and energy slope, dynamic ranges, pause structure and CPPS — which the functionals summarise only locally or not at all. 96 features, no embeddings, no pretrained network.

Nothing in this path can represent a word. That is a structural guarantee, not a design aspiration, and it is the point: the audEERING backend was measured following the *transcript* on contradictory clips despite receiving only audio. This model cannot do that by construction.

The features feed a scikit-learn pipeline — median imputation, standardisation, then a classifier. Four candidates are fitted and all four reported, so the results are not the best of four chosen after the fact: logistic regression, a calibrated linear SVM, an RBF SVM (selected on validation UAR, 0.703) and histogram gradient boosting. Because every feature is a named scalar, a prediction can be attributed to measurable quantities rather than to an opaque vector.

Its weakness is generalisation, and it is measured rather than assumed (§5). **Speaker identity is a demonstrated part of the problem**: per-speaker normalisation of the features recovers large amounts of performance (§6.6). Microphone, accent, language and health are plausible further confounds but were **not isolated in this study** — all corpora here are English, so language variation was never tested.

### 2 — WavLM

WavLM is not an emotion classifier by itself. It converts audio into a representation embedding that is used downstream for classification.

`audio → frozen WavLM encoder → embedding → trained classifier → sentiment probabilities`

Key points:

- it extracts a 1,536-value embedding per clip (mean + standard deviation pooling);
- the encoder stays frozen; only a lightweight classifier is trained on top;
- this keeps training inexpensive and focuses the comparison on the information carried by the representation itself;
- it outputs probabilities for negative, neutral and positive;
- MIT-licensed end to end, so it is the only acoustic backend here that could ship commercially.

This configuration tests how much prosodic information survives in a general-purpose speech representation without task-specific fine-tuning.

### 3 — audEERING

The audEERING model is a zero-shot acoustic encoder with validation-calibrated thresholds. Its CC-BY-NC-SA-4.0 licence is research-only and therefore unsuitable for commercial deployment, but it is useful as an analytical reference.

`audio → frozen audEERING model → continuous VAD values → validation-fitted valence thresholds → sentiment`

It outputs three continuous values:

- **Valence**: pleasantness of the emotion;
- **Arousal**: energy level;
- **Dominance**: perceived control or authority in the voice.

Two valence thresholds are fitted on CREMA-D validation speakers, and the calibrated model is then evaluated on a separate speaker-disjoint CREMA-D test set. Note that this is its **only** learned parameter set — two numbers — which is why its behaviour barely changes when the training data changes (§4, Experiment 3).

---

## 3. Data and evaluation

### CREMA-D (E1)

7,442 acted clips from 91 speakers, reading 12 fixed, emotionally neutral sentences. Useful for supervised training, but its wording is not designed to test the conflict between words and delivery: the words are neutral while the emotion is carried entirely by the performance.

The main test split is speaker-disjoint. A separate random split exists only to measure leakage and was never used for a shipped model.

### Synthetic recordings

Two text-to-speech providers were used to generate expressive audio with emotion tags:

- **Cartesia**: did not reliably render the requested emotional tags, especially when those tags conflicted with the sentiment of the words. It was therefore not treated as ground truth (§6.4).
- **Hume (E5)**: 90 clips from two voices, including 60 where wording and delivery deliberately disagree.

### Human recordings (E3)

Two independent takes by one human speaker (me), same prompts, sentiment-laden wording with instructed delivery styles. This tests transfer from acted studio speech to a real voice, but with one speaker it cannot support a generalisation claim.

### Zurich recordings (E6)

**160 clips from eight real speakers** (7 Italian, 1 Turkish; ages 30–40), collected with a custom tool I built for laptop and smartphone microphones ([link](https://voice-emotions-hub.lovable.app)). 35 sentences, **111 incongruent clips (69%)** in which the same wording appears with different intended deliveries.

Average clip duration 3.39 s (median 3.38 s; range 1.61–5.52 s). The split is speaker-disjoint by construction: five speakers train, two validate, one tests.

### Metrics

- **UAR** (unweighted average recall) is the primary classification metric. It averages the three per-class recalls, so it treats the classes equally. Three-class chance is 0.333.
- **Macro-F1** is reported alongside UAR.
- **PSI** (Prosody Sensitivity Index) applies to incongruent clips. 1.0 means the prediction follows the delivery, 0.0 means it follows the words, 0.5 means no systematic preference. *Contested* PSI counts only clips where the model picked one of the two competing labels; *strict* PSI counts all incongruent clips (chance 0.333).
- **Majority baseline** is a model that ignores the audio and always predicts the most frequent class in that set. Its UAR is always exactly 0.333, but its *accuracy* reaches 0.683 on CREMA-D test — which is precisely why accuracy is never reported alone here.
- **Confidence intervals** are percentile bootstrap intervals over clips. They are a floor on uncertainty, not a full account of it, since the same speakers and sentences recur.

---

## 4. Results

Three training regimes were run across all backends (`make model-matrix` → `results/model_matrix.json`).

### Experiment 1 — train CREMA-D → test Hume (E5)

| Backend | n | UAR [95% CI] | PSI contested | PSI strict | Reading |
|---|---:|---|---:|---:|---|
| research (2 thresholds) | 90 | 0.400 [0.31, 0.49] | **0.211** | 0.200 | below 0.5 — follows the words |
| research (VAD → logreg) | 90 | 0.478 [0.42, 0.54] | 0.614 | 0.450 | above chance |
| permissive (WavLM) | 90 | **0.511** [0.42, 0.60] | **0.682** | 0.500 | above chance |
| prosody (eGeMAPS) | 90 | 0.278 [0.20, 0.36] | 0.378 | 0.233 | below the baseline |
| majority baseline | 90 | 0.333 | 0.500 | 0.333 | — |

The Hume clips deliberately contain contradictory wording and delivery. The research model's PSI is 0.211 — below 0.5 — meaning it follows the text more than the voice, on audio only. **An "acoustic" model is not automatically a prosody model.**

### Experiment 2 — train CREMA-D + Hume → test Zurich (E6)

| Backend | n | UAR [95% CI] | macro-F1 | PSI contested | Dominant prediction |
|---|---:|---|---:|---:|---|
| research (2 thresholds) | 160 (8 spk) | 0.356 [0.29, 0.43] | 0.330 | 0.366 | 59% negative |
| research (VAD → logreg) | 160 (8 spk) | 0.398 [0.35, 0.45] | 0.320 | 0.488 | **84% neutral** |
| permissive (WavLM) | 160 (8 spk) | **0.433** [0.36, 0.51] | 0.402 | 0.646 | 65% negative |
| prosody (eGeMAPS) | 160 (8 spk) | 0.372 [0.31, 0.44] | 0.303 | 0.547 | **1% neutral** — collapses to pos/neg |
| majority baseline | 160 (8 spk) | 0.333 | 0.173 | 0.381 | always neutral |

On the single held-out Zurich speaker (20 clips): research-head 0.400, prosody 0.300, thresholds 0.289, permissive 0.222. All intervals are far too wide at n=20 to rank anything.

Every system lands between 0.356 and 0.433 against a 0.333 baseline, with intervals that touch it. The "dominant prediction" column explains why: each model is mostly replaying a class prior rather than reading the speaker. Adding Hume to training did not help — on the held-out Zurich speaker the permissive probe goes from 0.278 zero-shot to 0.222 with Hume folded in.

### Experiment 3 — train CREMA-D + Hume + Zurich → speaker-disjoint split of the union

| Backend | CREMA-D test (1470) | Hume Colton (45) | Zurich test spk (20) |
|---|---|---|---|
| research (2 thresholds) | 0.435 [0.41, 0.46] | 0.422 | 0.289 |
| research (VAD → logreg) | 0.605 [0.58, 0.63] | 0.422 | 0.400 |
| permissive (WavLM) | **0.742** [0.71, 0.77] | **0.600** | 0.289 |
| prosody (eGeMAPS, logreg) | 0.643 [0.61, 0.67] | 0.422 | 0.300 |

Pooling all three corpora improves the **in-domain** CREMA-D result but does not improve transfer to the real human recordings. The pooled "all data" number is not a useful summary, because 1,470 of its 1,535 clips are CREMA-D — it is a CREMA-D number wearing a union's name.

The comparison also isolates what "training" means per backend: giving audEERING a real trainable head instead of two thresholds moves it 0.435 → 0.605 on CREMA-D. Its flatness elsewhere is therefore partly "only two knobs", not only "insensitive encoder".

---

## 5. Cross-corpus comparison

All values are UAR, models trained on CREMA-D and applied cold. CREMA-D is the 300-clip stratified subset; E3 is one speaker (two takes); E5 is synthetic; E6 is 160 clips from eight real speakers.

| System | CREMA-D | E3a | E3b | E5 | **E6 UAR [95% CI]** | **E6 PSI [95% CI]** |
|---|---:|---:|---:|---:|---|---|
| A — Lexical | 0.360 | 0.333 | 0.370 | 0.333 | 0.314 [0.25, 0.39] | **0.104 [0.05, 0.17]** follows words |
| B — WavLM (permissive) | **0.797** | 0.444 | 0.519 | **0.511** | 0.396 [0.33, 0.46] | 0.584 [0.47, 0.70] contains 0.5 |
| B — audEERING (research) | 0.450 | 0.444 | **0.593** | 0.400 | 0.356 [0.29, 0.43] | **0.366 [0.27, 0.47]** follows words |
| C — Late fusion | **0.797** | 0.444 | 0.556 | 0.489 | **0.402 [0.34, 0.47]** | 0.557 [0.44, 0.67] contains 0.5 |
| D — Explicit prosody | 0.566 | 0.333 | 0.333 | 0.333 | 0.333 [0.33, 0.33] ⚠ collapsed | 0.521 [0.41, 0.64] |

**This is the central empirical result.** E6 is the only evaluation with enough real speakers to support a generalization claim, and it does not support a positive one. Only fusion clears chance on UAR, and it clears it by 0.003 (0.402, lower bound 0.336 against chance 0.333). The system that scores 0.797 on the benchmark scores 0.396 on eight people using laptop microphones.

Two reading notes:

- On the **full** CREMA-D speaker-disjoint test split the WavLM system scores **0.744** [0.71, 0.77]. The 0.797 above is the 300-clip comparison subset; the full split is the headline number.
- The **0.744-vs-0.435 gap is not a fair model comparison.** The WavLM probe is trained on CREMA-D's own training split; audEERING is zero-shot, trained on spontaneous podcast speech. On this corpus that is in-domain versus cross-domain. Where neither is in domain — E5 and E6 — they land close together.

### Why UAR and not accuracy

| Backend | accuracy | UAR | negative recall | neutral recall | positive recall |
|---|---:|---:|---:|---:|---:|
| WavLM probe | 0.816 | 0.744 | 0.882 | 0.693 | 0.657 |
| audEERING (zero-shot) | 0.599 | 0.435 | 0.747 | 0.247 | 0.311 |

audEERING predicts negative for 67% of clips on a corpus that is 68% negative, so its accuracy sits 16 points above its UAR. Of 251 genuinely positive clips it labels 99 as negative and only 78 correctly — it inverts a positive clip more often than it identifies one.

---

## 6. Tests in this repository, and what each one teaches

These are the checks that make the numbers above trustworthy. Each is implemented in code and each produced a conclusion.

### 6.1 Speaker-disjoint splits (`tests/test_splits.py`, `ssa/splits.py`)

**Test.** No speaker may appear in both training and evaluation. A function asserts this and is called at the top of every training run, not once when the split is created.

**Result.** All splits pass. A deliberately leaky random split was also measured for comparison: 0.761 UAR versus 0.744 on the speaker-disjoint split.

**What it teaches.** The guard is essential in principle, but on this corpus leakage inflated the score by only ~1.7 points. Random-split leakage is not always dramatic — which makes it *more* dangerous, not less, because a small unexplained gain is easy to accept.

### 6.2 The "too good to be true" ceiling (`ssa/train.py`, all experiment scripts)

**Test.** Any speaker-independent UAR above 0.90 is treated as a bug, since published state of the art is below that on comparable tasks.

**Result.** It fired for real. One E6 retraining run reported **UAR 1.000** because the `split` column was ignored and the model was fitted on the validation speaker it was then scored on. The bug was found, the training path fixed to use `split == "train"` only, and the result discarded.

**What it teaches.** A plausibility ceiling catches leakage that a passing unit test does not. The first sign of a data bug is usually a result that is *better* than it should be.

### 6.3 The words-versus-tone test (PSI, `ssa/eval/metrics.py`)

**Test.** On clips where wording and delivery disagree, does the prediction follow the words or the voice?

**Result.** On E5's 60 contradictory synthetic clips: permissive 0.682, research 0.211, lexical 0.000. On E6's 111 contradictory human clips: permissive 0.584 [0.47, 0.70], research **0.366 [0.27, 0.47]**, lexical **0.104 [0.05, 0.17]**.

**What it teaches.** The lexical baseline follows the words, as designed. The research acoustic model *also* follows the words, with no transcript as input, and its E6 interval lies entirely below chance. This replicates across a synthetic and a human dataset. The permissive model is **not** proven prosodic either: its interval contains 0.5.

### 6.4 Are the synthetic emotion tags real? (`scripts/gen_emotion_probe_d1.py`, `scripts/probe_hume.py`)

**Test.** If a TTS provider is asked for an emotion, can that emotion be recovered from the audio it returns? Measured with a leave-one-carrier-out classifier on descriptive features, plus F0 spread.

**Result.** Cartesia: recoverability **0.175** against 0.200 chance — the tags did not render. Hume with an explicit `description` field: **0.50** against 0.20 chance, F0 span 124 Hz; without it, exactly 0.20 (chance) and F0 span 38 Hz. On neutral wording, where prosody is the only signal, Hume's F0 span was 135 Hz.

**What it teaches.** A vendor's emotion label is not ground truth. One provider's tags were inaudible; another's worked, but only through a specific parameter. Synthetic data is usable as a controlled probe **only after verifying the audio**. This is also why the Cartesia clips were excluded from training.

### 6.5 Is the prosodic signal even reachable? (`scripts/compare_representations.py`)

**Test.** Fit a classifier *inside* the evaluation set (grouped by carrier sentence) on each representation. This is a ceiling measurement, not a deployable score: it asks whether the information exists in the features at all, separately from whether a CREMA-D-trained model can use it.

**Result.** On E3: prosodic 0.870, WavLM 0.759. On E5: prosodic 0.800, WavLM 0.789. The same representations reach 0.278–0.511 when the classifier is trained on CREMA-D instead.

**What it teaches.** This is the strongest evidence in the project that **the training corpus, not the features or the encoder, is the bottleneck**. The signal is present and reachable at ~0.8 UAR from both representations. What fails is transferring a decision boundary learned on acted studio speech.

### 6.6 Per-speaker normalisation (`scripts/normalise_speaker.py`, `scripts/normalise_skew.py`)

**Test.** Subtract each speaker's own feature means before classifying, to remove their baseline voice. Then stress-test it: does the benefit survive when a speaker's clips are *not* class-balanced?

**Result.** Large gains for the prosody model: E5 0.278 → 0.611 (+0.33), E3a +0.19, E3b +0.22, CREMA-D +0.03. Under increasing class imbalance the gain degrades but does not vanish: E5 +0.33 → +0.27, E3b +0.22 → +0.05.

**What it teaches.** A large part of the prosody model's brittleness is **speaker baseline offset**, not missing signal — which is measured, not assumed. But the fix is transductive: it needs a pool of that speaker's clips, so a per-clip API cannot use it. It is a mitigation for a device that knows its user over time, not a fix for domain shift.

### 6.7 Repeatability across two takes (`scripts/eval_e3.py`, `results/d1_vs_e3_control.json`)

**Test.** The same speaker recorded the same prompts twice. How consistent is each model's reading across takes? Correlation over 27 paired clips.

**Result.** audEERING valence correlates at **r = 0.92** across takes. The WavLM probe's equivalent correlates at **r = −0.19**.

**What it teaches.** **Stability is not correctness.** The most repeatable model is the one that follows the words rather than the tone (§6.3), and the less repeatable one is the more prosody-sensitive. High test-retest agreement would have looked like a quality signal; it was measuring something else.

### 6.8 Collapse and baseline detection (`scripts/train_model_matrix.py`, `ssa/eval/runner.py`)

**Test.** Flag any model that predicts one class for ≥95% of clips, and compute a majority-class baseline for every evaluation set.

**Result.** It caught the prosody model predicting a single class for all of E6, which produces UAR exactly 0.333 and PSI exactly 0.500 — numbers that look like ordinary mediocre performance. It also exposed that a do-nothing model scores **PSI contested 1.000** on CREMA-D, because CREMA-D's text is neutral on every clip, so "always negative" can never match the text label.

**What it teaches.** A degenerate model can hide inside plausible-looking metrics. And **PSI is only meaningful where the text label actually varies** — on CREMA-D it should not be read at all.

### 6.9 Ambiguous emotions are excluded, not guessed (`tests/test_mapping.py`)

**Test.** Emotions without an agreed sentiment valence (`surprised`, `nostalgic`, `sarcastic`, `mysterious`, `determined`, `calm`) must raise an error rather than defaulting to a class.

**Result.** Enforced by test; those tags never enter a labelled set.

**What it teaches.** Silently bucketing an ambiguous emotion into positive/negative would inflate the label set with guesses and make every downstream number partly a coding decision. Refusing to label is a result in itself — and it points at the taxonomy problem in §7.

### 6.10 Test suite

`uv run pytest` runs **353 tests** covering the split invariant, the label mapping, the PSI implementation on hand-computed fixtures, the manifest schema and the report renderers. Fixtures are small and deterministic; network-dependent tests are marked and skipped by default.

**What it teaches.** The tests exist to encode the invariants that make results trustworthy, not to reach a coverage figure. The two that matter most — speaker disjointness and ambiguous-label rejection — are the ones that would silently invalidate every number in this report if they broke.

---

## 7. Limitations

**Sample size and population**

- E6's held-out test split is 20 clips from one speaker; the 160-clip, eight-speaker view is the only one worth reading.
- E3 is one human speaker; E5 is two synthetic voices. Neither supports a population claim.
- The target population is older adults, and **no older adults were recorded**. Any claim about age-related voice changes is a hypothesis from the literature, not a measured finding here.
- All corpora are English, so language and accent effects were never isolated.

**Labels and design**

- E6's labels were assigned by me, the analyst. The lexical baseline's below-chance PSI is a useful sanity check but does not remove this limitation.
- Many clips are short (median 3.4 s), and prompts are deliberately contradictory, so even a human listener may not read the intended emotion consistently. This adds label noise and weakens the learning signal.
- Acted and synthetic contradictory delivery may itself be unnatural, distorting the prosodic signal relative to spontaneous speech.

**Withdrawn claims** (kept visible rather than deleted)

- An earlier six-speaker version of E6 suggested that training on 80 matched clips *alone* beat 5,235 acted ones (0.452 versus 0.278). On the eight-speaker version the same configuration gives **0.222 [0.06, 0.40]** — the effect reversed. It was noise on a 20-clip split.
- The E6 retraining run that reported UAR 1.000 was a leak (§6.2), not a result.

**The taxonomy problem**

Three classes may be the wrong label space for this product. "Neutral" absorbs calm, resignation and low-energy speech, which mean different things to a companion device, while sarcasm, disappointment and sadness may deserve separate treatment rather than being folded into "negative". For older adults a more useful taxonomy might be: positive/energetic; calm/relaxed; angry/disappointed/frustrated; sarcastic; depressed/sad; crying; low-energy, trembling or illness-related vocal change.

This is not a side note. Defining which emotional states actually matter to the user population is arguably a larger open problem here than the choice of model.

---

## 8. Next steps

1. Train on naturalistic, multi-speaker, multi-language speech rather than acted CREMA-D.
2. Add more held-out **speakers**, not more clips from the same speakers.
3. Record the older adult population the product targets, in realistic settings.
4. Re-record longer, less ambiguous prompts so emotional intent is humanly interpretable.
5. Design the label taxonomy around the use case instead of assuming three-class sentiment.
6. Investigate a longitudinal per-speaker baseline, which §6.6 shows is where the recoverable performance is — appropriate for a device used by one person over time.
7. Treat tone–word mismatch as a product signal that triggers a clarifying question, rather than forcing a label.

---

## 9. Conclusion

> Models trained on acted CREMA-D speech do not provide practically useful cross-speaker generalization on real-world recordings.

The best cross-speaker result is 0.402 UAR against a chance level of 0.333. A model can score 0.797 on the benchmark and still fail on people speaking into laptop microphones.

The most useful output of this project is not the benchmark score but the diagnostics: an "acoustic" model can follow the wording rather than the delivery (§6.3); a highly repeatable model can be repeatably wrong (§6.7); the prosodic signal is reachable at ~0.8 UAR when the classifier is fitted in-domain, which locates the bottleneck in the training corpus rather than in the features (§6.5).

For a voice-first assistant for older adults, the defensible design is therefore not to force a three-class label, but to detect ambiguity, flag mismatch between wording and tone, and ask a clarifying question when the evidence is weak.

For the full interactive results dashboard and additional evaluation outputs, see [report/index.html](report/index.html).
