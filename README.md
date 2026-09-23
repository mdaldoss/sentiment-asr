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

The explicit prosody model measures named physical quantities of the waveform: 88 openSMILE **eGeMAPS v02** functionals (F0 statistics, loudness, jitter, shimmer, HNR, spectral balance, voiced-segment rates) plus 8 **contour descriptors** computed in Praat: global F0 and energy slope, dynamic ranges, pause structure and CPPS. it provides 96 features, no embeddings, no pretrained network.

Nothing in this path can represent a word. That is a structural guarantee, as it has been measured, on the contrary of other models.

The features feed a scikit-learn pipeline median imputation, standardisation, then a classifier. Four candidates are fitted and reported: logistic regression, a calibrated linear SVM, an RBF SVM (selected on validation UAR, 0.703) and histogram gradient boosting. Every feature is a named scalar, providing a measurable quantities rather than to an opaque vector.

Its weakness is generalisation, as it has been measured (§5). Speaker identity is a demonstrated part of the problem**: per-speaker normalisation of the features recovers large amounts of performance (§6.6). Microphone, accent, language and health are plausible further confounds but were not isolated in this study all corpora here are English, so language variation was not tested.

### 2 — WavLM

WavLM is not an emotion classifier by itself. It converts audio into a representation embedding that is used downstream for classification.

`audio → frozen WavLM encoder → embedding → trained classifier → sentiment probabilities`

Key points:

- it extracts a 1,536-value embedding per clip (mean + standard deviation pooling);
- the encoder stays frozen; only a lightweight classifier is trained on top;
- this keeps training inexpensive and focuses the comparison on the information carried by the representation itself;
- it outputs probabilities for negative, neutral and positive;
- MIT-licensed end to end, so it is an acoustic backend here that could ship commercially.

This configuration tests how much prosodic information survives in a general-purpose speech representation without task-specific fine-tuning.

### 3 — audEERING

The audEERING model is a zero-shot acoustic encoder with validation-calibrated thresholds. Its CC-BY-NC-SA-4.0 licence is research-only and therefore unsuitable for commercial deployment, but it is useful as an analytical reference.

`audio → frozen audEERING model → continuous VAD values → validation-fitted valence thresholds → sentiment`

It outputs three continuous values:

- **Valence**: pleasantness of the emotion;
- **Arousal**: energy level;
- **Dominance**: perceived control or authority in the voice.

Two valence thresholds are fitted on CREMA-D validation speakers, and the calibrated model is then evaluated on a separate speaker-disjoint CREMA-D test set. These are the only two parameters learned, which is why its behaviour barely changes when the training data changes (§4, Experiment 3).

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
- **Majority baseline** it ignores the audio and always predicts the most frequent class in that set. Its UAR is always exactly 0.333, but its *accuracy* reaches 0.683 on CREMA-D test which is precisely why accuracy is not considered as a valuable metric for this project.
- **Confidence intervals** are percentile bootstrap intervals over clips. They represent the boundaries of uncertainty, since the same speakers and sentences recur.

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

The Hume clips deliberately contain contradictory wording and delivery. The research (audEERING) model's PSI is 0.211 — below 0.5 — meaning it follows the text more than the voice, on audio only. -> An "acoustic" model is not automatically a prosody model.

### Experiment 2 — train CREMA-D + Hume → test Zurich (E6)

| Backend | n | UAR [95% CI] | macro-F1 | PSI contested | Dominant prediction |
|---|---:|---|---:|---:|---|
| research (2 thresholds) | 160 (8 spk) | 0.356 [0.29, 0.43] | 0.330 | 0.366 | 59% negative |
| research (VAD → logreg) | 160 (8 spk) | 0.398 [0.35, 0.45] | 0.320 | 0.488 | **84% neutral** |
| permissive (WavLM) | 160 (8 spk) | **0.433** [0.36, 0.51] | 0.402 | 0.646 | 65% negative |
| prosody (eGeMAPS) | 160 (8 spk) | 0.372 [0.31, 0.44] | 0.303 | 0.547 | **1% neutral** — collapses to pos/neg |
| majority baseline | 160 (8 spk) | 0.333 | 0.173 | 0.381 | always neutral |

On the single held-out Zurich speaker (20 clips): research-head 0.400, prosody 0.300, thresholds 0.289, permissive 0.222. All intervals are far too wide at n=20 to rank anything.

Every system lands between 0.356 and 0.433 against a 0.333 baseline, with intervals that touch it. The "dominant prediction" column explains why: each model is mostly replaying a class prior rather than reading the speaker. Adding Hume to training did not help on the held-out Zurich speaker the permissive probe goes from 0.278 zero-shot to 0.222 with Hume included in the probe training.

### Experiment 3 — train CREMA-D + Hume + Zurich → speaker-disjoint split of the union

| Backend | CREMA-D test (1470) | Hume Colton (45) | Zurich test spk (20) |
|---|---|---|---|
| research (2 thresholds) | 0.435 [0.41, 0.46] | 0.422 | 0.289 |
| research (VAD → logreg) | 0.605 [0.58, 0.63] | 0.422 | 0.400 |
| permissive (WavLM) | **0.742** [0.71, 0.77] | **0.600** | 0.289 |
| prosody (eGeMAPS, logreg) | 0.643 [0.61, 0.67] | 0.422 | 0.300 |

Pooling all three corpora improves the **in-domain** CREMA-D result but does not improve transfer to the real human recordings. The pooled "all data" number is actually not a useful summary, because 1,470 of its 1,535 clips are CREMA-D, so CREMA-D represent the majority of the dataset.

In this experiment is also shown an interesting behaviour with audEERING. By providing a logistic regression rather than two fixed threshold the UAR moves from 0.435 → 0.605 on CREMA-D. This open a scanario to consider if the performace bottleneck it is actually in the thresholds selection rather than its encoder.

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

**Test**. To test the limit of these audio features, we ran a grouped $k$-fold cross-validation directly inside the evaluation sets (E3 and E5), keeping carrier sentences separate between folds to prevent lexical memorization. This measures a theoretical "ceiling" rather than a deployable model: it checks whether the necessary information actually exists in the representations, independent of the CREMA-D training pipeline.
**The Results.** When evaluated via in-domain $k$-fold cross-validation, performance improves drastically:On E3: prosodic features reach 0.870, and WavLM reaches 0.759.On E5: prosodic features reach 0.800, and WavLM reaches 0.789.By contrast, using a decision boundary trained on CREMA-D yields only 0.278–0.511 on the exact same features.
**What it teaches**. The sharp jump in performance under $k$-fold cross-validation provides the strongest evidence that the training corpus, not the representations or the encoder, is the bottleneck. The signal is readily reachable at ~0.80 score from both feature sets; the cross-dataset failure is driven entirely by domain shift from the acted studio speech in CREMA-D.


---

## 7. Limitations

**Sample size and population**

- E6's held-out test split is 20 clips from one speaker; the 160-clip, eight-speaker
- E3 is one human speaker; E5 is two synthetic voices. 
- The target population is older adults, but no older adults have been recorded. 
- All corpora are English and language and accent effects were never isolated.

**Labels and design**

- Many clips are short (median 3.4 s), and prompts are deliberately contradictory, even a human listener may not read the intended emotion consistently. This adds label noise and weakens the learning signal.
- Acted and synthetic contradictory delivery may itself be unnatural, distorting the prosodic signal relative to spontaneous speech.


**Taxonomy**

Three classes may be the wrong label space for this product. "Neutral" absorbs calm, resignation and low-energy speech, which mean different things to a companion device, while sarcasm, disappointment and sadness may deserve separate treatment rather than being folded into "negative". For older adults a more useful taxonomy might be: positive/energetic; calm/relaxed; angry/disappointed/frustrated; sarcastic; depressed/sad; crying; low-energy, trembling or illness-related vocal change.

This is not a side note. Defining which emotional states actually matter to the user population is probably a larger open problem than the choosing a model.

---

## 8. Next steps

1. Train on naturalistic, multi-speaker, multi-language speech rather than acted CREMA-D.
2. Add more held-out speakers, not more clips from the same speakers.
3. Record the older adult population the product targets, in realistic settings.
4. Re-record longer, less ambiguous prompts so emotional intent is humanly interpretable.
5. Investigate and design the label taxonomy around the use case and target group instead of assuming a three-class sentiment
6. Treat tone–word mismatch as a product signal that triggers a clarifying question, rather than forcing a label.

---

## 9. Conclusion

A large, representative dataset is critical (§6.5). WaveLM Model have proved to be able to score 0.797 on CREMA-D dataset. This because of the size of the dataset and, arguably, because emotions had been represented with a consistent accent from professional actors.
Moreover, also supported by *arXiv:2604.25776*, trainig and evaluating system on the target (elderly) population has significant value.

On this regard a requirement consideration must be made.
For a voice-first assistant for older adults, a defensible design would therefore not to force a three-class label, but to discriminate different "negative" emotions (such as sadness, sarcastic, angryness) and in ambiguity flag the mismatch between wording and tone and ask a clarifying question when the evidence is weak.

Also state of the art "acoustic" research models can actually still follow the wording rather than the prosody.

For the full interactive results dashboard and additional evaluation outputs, see [report/index.html](report/index.html).


## 10. Data & licensing

| Asset | License | Redistributed here? |
|---|---|---|
| CREMA-D | ODbL v1.0 | No — `make data` fetches it |
| Synthetic (Cartesia) | generated | Yes — 76/90 E2 clips + 45 D1 clips committed |
| Recordings (E3) | authors' own | Yes — 2 takes, 30 clips each, committed |
| Recordings (E6, Zurich) | participants' own | Yes — 120 clips, 6 speakers, committed (source WebM/m4a + decoded 16 kHz WAV) |
| `audeering` VAD model | **CC-BY-NC-SA-4.0, research only** | No |
| WavLM, faster-whisper, text classifier | MIT / Apache-2.0 | No |

