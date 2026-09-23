# Speech Sentiment Analyzer

### AI/ML Systems Engineer Take-Home Assignment

## 1. Summary

This project investigates whether sentiment can be inferred from raw speech audio alone, without relying on a transcript. The goal is to build a system that can classify a speaker as positive, neutral, or negative while remaining sensitive to the difference between the words spoken and the prosody used to deliver them.

This matters because the same sentence can convey very different emotional intentions depending on tone. A phrase such as “that’s wonderful” can sound warm and sincere or tired, sarcastic or angry. A useful voice companion should be able to detect when the wording and the delivery disagree, rather than blindly trusting the textual content.

The project compares three system families: a lexical baseline, an acoustic model and an explicit prosody model. The central lesson is not that a model can achieve perfect sentiment classification, but that it must be evaluated under speaker-disjoint and on separate datasets that reflect the typical real scenario. In this setting, the hardest challenge is not just classification accuracy, but generalization to real multi-speaker audio that differs from the training distribution.

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
1. **WavLM** — also referred to as the permissive acoustic model.
2. **audEERING** — also referred to as the research acoustic model.
3. **Explicit prosody** — handcrafted acoustic features such as pitch, loudness, and eGeMAPS-style measurements.

### 0 — Lexical model

The lexical model acts as a reference system for measuring whether a model follows prosody or wording. It first transcribes speech with Faster-Whisper Small and then classifies the transcript using a text sentiment model, specifically CardiffNLP’s Twitter RoBERTa sentiment model.

This system is deliberately simple: it provides a lower bound on how much sentiment can be recovered from the text alone and it helps establish if a speech model is learning from tone rather than from linguistic content.

### 1 — Explicit prosody model

The explicit prosody model measures named physical quantities of the waveform: 88 openSMILE eGeMAPS v02 functionals (F0 statistics, loudness, jitter, shimmer, HNR, spectral balance, voiced-segment rates) plus 8 contour descriptors computed in Praat — global F0 and energy slope, dynamic ranges, pause structure and CPPS — which the functionals summarise only locally or not at all. 96 features, no embeddings, no pretrained network.

Nothing in this model can represent a word. The deep acoustic backend was measured following the transcript on contradictory clips.

The features feed a scikit-learn pipeline: median imputation, standardisation then a classifier. Four candidates are fitted and reported: logistic regression, a calibrated linear SVM, an RBF SVM (selected on validation UAR, 0.703) and histogram gradient boosting. 

Its weakness, as it was measured, is generalisation. It is known to be affected by speaker identity, microphone, accent, language and health are plausible. 

### 2 — WavLM
WavLM is not an emotion classifier by itself. It converts audio into a representation embedding that can be used downstream for classification.

The pipeline is:

`audio → frozen WavLM encoder → embedding → trained classifier → sentiment probabilities`

Key points:

- it extracts a 1,536-value embedding from each clip;
- the encoder remains frozen;
- only a lightweight classifier is trained on top of the embeddings;
- this keeps the system inexpensive to train and makes the comparison focus on the information carried by the representation itself;
- the system outputs probabilities for negative, neutral, and positive classes.

This configuration is useful because it tests how much useful prosodic information survives in a general-purpose speech representation without task-specific fine-tuning.

### 3 — audEERING

The audEERING model is a zero-shot acoustic encoder with validation-calibrated thresholds. Its CC-BY-NC-SA-4.0 license is research-only and therefore not suitable for commercial deployment, but it is useful as an analytical reference model.

The encoder is pretrained for audio analysis and emotion recognition and remains frozen during use. The pipeline is:

`audio → frozen audEERING model → continuous VAD values → validation-fitted valence thresholds → sentiment`

It provides VAD features:

- Valence: pleasantness of the emotion;
- Arousal: energy level of the emotion;
- Dominance: level of control or perceived authority in the voice.

Two valence thresholds are selected using CREMA-D validation speakers, and the calibrated model is then evaluated on a separate, speaker-disjoint CREMA-D test set.

---

## 3. Data and evaluation

The following datasets were used for training and evaluation.

### CREMA-D (E1)

CREMA-D contains 7,442 acted clips from 91 speakers. The speakers read 12 fixed, emotionally neutral sentences. This dataset is useful for supervised training, but its wording is not designed to test the conflict between words and delivery; the words are neutral while the emotion is conveyed through performance.

The main test split is speaker-disjoint. A separate random split was used only to measure leakage and was never used for the shipped model.

### Synthetic recordings

Two text-to-speech providers were used to generate expressive audio with emotion tags:

- Cartesia: did not reliably render the requested emotional tags, especially when those tags conflicted with the sentiment of the words. Because of this, it was not treated as ground truth without independent audio verification.
- Hume recordings (E5): 90 synthetic clips were generated from two Hume voices, including 60 clips where the wording and delivery deliberately disagree.

### Human recordings (E3)

E3 contains two independent takes by one human speaker (me). The same prompts were recorded with sentiment-laden wording and instructed delivery styles. This dataset tests transfer from acted studio speech to a real voice, but it contains only one speaker and therefore cannot support a generalisation claim beyond that individual.

### Zurich recordings (E6)

The Zurich dataset contains **160 clips from eight real speakers** (7 Italian, 1 Turkish; ages 30–40). The recordings were collected using a custom tool I designed for laptop and smartphone microphones [link](https://voice-emotions-hub.lovable.app). The dataset includes 35 sentences and **111 short incongruent clips (69%)** in which the same wording appears with different intended deliveries.

The average clip duration is 3.39 s (median 3.38 s; range 1.61–5.52 s). The split is speaker-disjoint by construction: five speakers are used for training, two for validation, and one for testing.

### Metrics

- **UAR** (unweighted average recall) is the primary classification metric. It treats the three classes equally and is more appropriate than accuracy when class balance is imperfect. Three-class chance is approximately 0.333.
- **Macro-F1** is reported alongside UAR.
- **PSI** (Prosody Sensitivity Index) is used on incongruent clips. A value of 1.0 means the prediction follows the delivery, 0.0 means it follows the words, and 0.5 means no systematic preference.
- **Confidence intervals** are percentile bootstrap intervals over clips. They provide useful context but cannot fully capture uncertainty when the number of speakers is small.

---

## 4. Results

Several evaluations were performed; the most relevant ones are reported below.

### Experiment 1 — train CREMA-D (E1) → test Hume

Hyperparameters were tuned on CREMA-D and evaluated on the synthetic Hume dataset (E5).

| Backend | n | UAR [95% CI] | PSI | Contested PSI | What it actually predicts |
|---|---:|---|---:|---:|---|
| research (2 thresholds) | 90 | 0.400 [0.31, 0.49] | 0.211 | 0.233 | below 0.5 chance — follows the words |
| research (VAD → logreg) | 90 | 0.478 [0.42, 0.54] | 0.614 | 0.550 | — |
| permissive | 90 | 0.511 [0.42, 0.60] | 0.682 | 0.727 | — |
| prosody | 90 | 0.278 [0.20, 0.36] | 0.378 | 0.421 | — |
| majority baseline | 90 | 0.333 | 0.500 | — | — |

This result is important because the Hume clips intentionally contain contradictory wording and delivery. The research model has the lowest PSI, and the value is below 0.5, which means it follows the text more than the voice. In other words, the acoustic model is not behaving like a prosody-sensitive model in this setting.

### Experiment 2 — train CREMA-D + Hume → test Zurich recordings

In this experiment, hyperparameters were tuned on CREMA-D and the Hume synthetic set, and the evaluation target was the human Zurich dataset (E6).

| Backend | Dataset / split | n | UAR [95% CI] | macro-F1 | PSI | Contested? | What it actually predicts |
|---|---|---:|---|---:|---:|---|---|
| research (2 thresholds) | all | 160 (8 spk) | 0.356 [0.29, 0.43] | 0.330 | 0.366 | 59% negative | — |
| research (VAD → logreg) | all | 160 (8 spk) | 0.398 [0.35, 0.45] | 0.320 | 0.488 | 84% neutral | — |
| permissive (WavLM) | all | 160 (8 spk) | 0.433 [0.36, 0.51] | 0.402 | 0.646 | 65% negative | — |
| prosody (eGeMAPS) | all | 160 (8 spk) | 0.372 [0.31, 0.44] | 0.303 | 0.547 | 1% neutral | collapses to pos/neg |
| majority baseline | all | 160 (8 spk) | 0.333 | 0.173 | 0.381 | — | — |
| permissive (WavLM) | Zurich single held-out speaker | 20 | 0.222 | — | — | — | — |
| research-head | Zurich single held-out speaker | 20 | 0.400 | — | — | — | — |
| prosody | Zurich single held-out speaker | 20 | 0.300 | — | — | — | — |
| thresholds | Zurich single held-out speaker | 20 | 0.289 | — | — | — | — |

This result shows that the prosody model struggles with the neutral class and tends to collapse toward positive or negative predictions. The broader pattern is consistent: once the system is evaluated on real multi-speaker audio, performance degrades substantially from the acted CREMA-D benchmark.

### Experiment 3 — train CREMA-D + Hume + Zurich recordings → speaker-disjoint split of the union

In this setup, CREMA-D, the Hume synthetic set, and the Zurich recordings were split into train, validation, and test partitions using speaker-disjoint logic. The aim was to increase coverage across multiple conditions and evaluate whether a more diverse training set improves generalization.

| Backend | CREMA-D test (1470) | Hume Colton (45) | Zurich test spk (20) |
|---|---|---|---|
| research (2 thresholds) | 0.435 [0.41, 0.46] | 0.422 | 0.289 |
| research (VAD → logreg) | 0.605 [0.58, 0.63] | 0.422 | 0.400 |
| permissive | 0.742 [0.71, 0.77] | 0.600 | 0.289 |
| prosody | 0.643 [0.61, 0.67] | 0.422 | 0.300 |

The key observation is that the union split improves the in-domain CREMA-D performance, but it does not meaningfully improve transfer to the real human recordings. The pooled all-data result is not the right summary when most clips come from CREMA-D, because it masks the much harder cross-corpus transfer problem.


---

### 5.0 Focus on Transfer to real speakers

The table below evaluates models trained on CREMA-D and tested on all 160 E6 clips. Chance is 0.333 for UAR and 0.500 for PSI.

| System | UAR [95% CI] | PSI [95% CI] | Reading |
| --- | ---: | ---: | --- |
| A — Lexical | 0.314 [0.246, 0.386] | **0.104 [0.047, 0.170]** | follows words |
| B — Acoustic, permissive | 0.396 [0.330, 0.464] | 0.584 [0.472, 0.697] | inconclusive |
| B — Acoustic, research | 0.356 [0.287, 0.428] | **0.366 [0.265, 0.472]** | follows words |
| C — Fusion | **0.402 [0.336, 0.467]** | 0.557 [0.443, 0.667] | only just above chance |
| D — Explicit prosody | 0.333 [0.333, 0.333] | 0.521 [0.405, 0.639] | predicted one class |

Only fusion clears chance on UAR by a small margin, and even that result is not practically useful. The permissive acoustic system falls from 0.797 on the CREMA-D subset to 0.396 on E6. This demonstrates that benchmark performance does not transfer to real multi-speaker audio.


## 5.1 Main model comparison

Values are UAR. CREMA-D is the 300-clip stratified test subset; E3 is one speaker
(two takes); E5 is synthetic; **E6 is 160 clips from eight real speakers**.

| Model                |   CREMA-D |   E3a |       E3b |        E5 | **E6 (n=160)** |
| -------------------- | --------: | ----: | --------: | --------: | -------------: |
| Lexical              |     0.360 | 0.333 |     0.370 |     0.333 | 0.314 [0.25, 0.39] |
| Acoustic — WavLM     | **0.797** | 0.444 |     0.519 | **0.511** | 0.396 [0.33, 0.46] |
| Acoustic — audeering |     0.450 | 0.444 | **0.593** |     0.400 | 0.356 [0.29, 0.43] |
| Late fusion          | **0.797** | 0.444 |     0.556 |     0.489 | **0.402 [0.34, 0.47]** |
| Explicit prosody     |     0.566 | 0.333 |     0.333 |     0.333 | 0.333 [0.33, 0.33] |

On the full CREMA-D speaker-disjoint test split the WavLM-based acoustic system achieved
**0.744 UAR** [0.71, 0.77]. The difference from 0.797 is that the latter is measured on
the 300-clip subset used by the comparison harness; the full split is the headline
CREMA-D result.

The audeering column was previously blank because that backend had only ever been run on
CREMA-D's *validation* split — the split its two valence thresholds were fitted on. It now
has a held-out number: **0.435 [0.41, 0.46]** on the full test split. Two points about it:

* **The two thresholds did not overfit.** Validation minus test is +0.019, so the older
  0.454 figure was always a reasonable estimate. What was wrong was presenting it beside
  the permissive backend's held-out 0.744 as though both were measured the same way.
* **The 0.744-vs-0.435 gap is not a fair model comparison.** The WavLM probe is trained
  on CREMA-D's own training split; the audeering model is zero-shot and was trained on
  spontaneous podcast speech. On this corpus that is in-domain versus cross-domain. Where
  neither is in domain — E5 and E6 — the two land close together.

The audeering backend's failure on CREMA-D is concentrated in the minority classes, and
it is a good illustration of why this report uses UAR rather than accuracy:

| Backend | accuracy | UAR | negative recall | neutral recall | positive recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| WavLM probe | 0.816 | 0.744 | 0.882 | 0.693 | 0.657 |
| audeering (zero-shot) | 0.599 | 0.435 | 0.747 | 0.247 | 0.311 |

The audeering model predicts negative for 67% of clips on a corpus that is 68% negative,
so its accuracy sits 17 points above its UAR. Of 251 genuinely positive clips it labels 99
as negative and only 78 correctly — it inverts a positive clip more often than it
identifies one.

The important observation is not the absolute CREMA-D number. It is what happens on E6.

## 5.2 The cross-corpus result

E6 is the only evaluation here with enough real speakers to support a generalization
claim, and it does not support a positive one. With 95% bootstrap intervals, **only the
fusion model clears chance on UAR, and it clears it by 0.003** (0.402, interval lower
bound 0.336 against chance 0.333). The system that reaches 0.797 on the benchmark reaches
0.396 [0.33, 0.46] on eight people recording through laptop microphones.

Explicit prosody (Solution D) is worse than weak on E6: it predicts a single class for
every clip. A collapsed model of this kind produces UAR of exactly 0.333 and PSI of
exactly 0.500, which look like ordinary mediocre scores rather than a failure, so the
evaluation flags the collapse explicitly rather than letting those numbers stand.

I take this as the central empirical result of the project. Acted studio speech with
fixed neutral sentences does not prepare a model for spontaneous, accented, consumer
microphone audio, and every earlier transfer number in this report — measured on one
speaker or on synthesized audio — was more optimistic than reality.

## 5.3 Does training on the new speakers help?

Refitting on E6's 100 training clips, scored on the two held-out validation speakers
(40 clips; the 20-clip test split is too small to separate anything):

| Training set | fit clips | WavLM probe | Prosody (logreg) | Prosody (SVM-RBF) |
| --- | ---: | --- | --- | --- |
| CREMA-D only | 5,235 | 0.330 [0.19, 0.47] | 0.332 [0.26, 0.40] | 0.333 (collapsed) |
| CREMA-D + E6 train | 5,335 | 0.375 [0.22, 0.53] | 0.372 [0.23, 0.53] | 0.370 [0.23, 0.51] |

Every interval overlaps, so **the UAR improvement is not established**. What does not
depend on an interval is that adding E6's training clips stops Solution D collapsing: the
SVM goes from predicting one class for every clip to producing real predictions. That is
a qualitative change, and it points the same way as everything else in this project — the
training corpus, not the model, is the binding constraint.

**A claim I withdraw.** On an earlier six-speaker version of E6, training on the new
clips *alone* gave the WavLM probe 0.452 against CREMA-D's 0.278, and I came close to
reporting that 80 matched clips beat 5,235 acted ones. On the eight-speaker version the
same configuration gives **0.222 [0.06, 0.40]** — the effect reversed. It was noise on a
20-clip test split, and the only reason it was not written up as a finding is that the
bootstrap interval already said it could not be resolved. I record it because it is a
concrete example of the failure mode this report keeps returning to.

---

# 5. What the experiments taught me

## 6.1 Acoustic representations are not necessarily purely prosodic

This is the strongest diagnostic result in the project, and it now holds on two
independent datasets.

### The words-versus-tone test

On E5’s 60 contradictory synthetic clips:

- permissive acoustic PSI: **0.682**;
- research acoustic PSI: **0.211**;
- lexical PSI: **0.000**.

On E6’s 111 contradictory human clips:

- permissive acoustic PSI: 0.584 [0.472, 0.697];
- research acoustic PSI: **0.366 [0.265, 0.472]**;
- lexical PSI: **0.104 [0.047, 0.170]**.

The lexical baseline behaves as expected: it follows the words. The research acoustic model also follows the words, despite receiving no transcript as input. This finding appears in both E5 and E6, and the E6 interval lies entirely below chance.

The permissive acoustic model is not proven to follow tone: its E6 interval contains chance. Its point estimate is above chance, but the evidence is not strong enough to make a confident claim about true prosodic sensitivity.

### Other completed experiments

The headline result is negative but informative: on E6 — 160 clips from eight real speakers recorded on laptop microphones, with 111 incongruent clips — every system trained on CREMA-D collapses under realistic conditions.

| Solution | CREMA-D UAR | E6 UAR [95% CI] | E6 PSI [95% CI] | Interpretation |
| --- | ---: | ---: | ---: | --- |
| A — lexical | 0.360 | 0.314 [0.25, 0.39] | 0.104 [0.05, 0.17] | follows words |
| B — acoustic (permissive) | 0.797 | 0.396 [0.33, 0.46] | 0.584 [0.47, 0.70] | contains 0.5 |
| B — acoustic (research) | 0.450 | 0.356 [0.29, 0.43] | 0.366 [0.27, 0.47] | follows words |
| C — fusion | 0.797 | 0.402 [0.34, 0.47] | 0.557 [0.44, 0.67] | contains 0.5 |
| D — prosodic | 0.566 | 0.333 [0.33, 0.33] ⚠ | 0.521 [0.41, 0.64] | degenerate |

Of the five systems, only one clears chance on UAR — and only by 0.003. Nothing trained on CREMA-D transfers with useful reliability to real multi-speaker audio.

What remains is sharper than the raw accuracy numbers: the audEERING backend’s PSI interval sits entirely below chance, which indicates an “acoustic” model that follows the wording rather than the delivery, even without access to the transcript. E5 found this on synthetic voices; E6 replicates it on eight real speakers.

---

## 6. What the results mean

1. **The data is the main bottleneck.** The gap between the training and evaluation corpora is larger than the differences between the tested model families.
2. **A high benchmark score is not enough.** CREMA-D performance does not transfer to real multi-speaker recordings.
3. **Acoustic does not mean prosodic.** Speech encoders can carry linguistic information even when no transcript is explicitly provided.
4. **Mismatch detection may be more useful than sentiment classification.** On an ambiguous utterance, detecting a conflict between words and delivery can be more valuable than forcing a confident label.
5. **UAR and PSI measure different abilities.** A model can be near chance at naming sentiment while still showing some evidence of detecting delivery mismatch.

---

## 7. Limitations and corrections

- E6 contains only 20 held-out test clips from one speaker in the single-speaker view, which makes it useful but still limited.
- E3 contains only one human speaker, and E5 uses two synthetic voices; neither is enough to support broad population claims.
- The target population is older adults, but no older adults were recorded. Any claims about age-related voice changes remain hypotheses, not measured findings.
- E6’s lexical labels were assigned by the analyst. The lexical baseline’s below-chance PSI is a useful sanity check, but it does not remove this limitation.
- One E6 retraining run produced a UAR of 1.000 because the split column was ignored. The leak was found, the training path was fixed to use only `split == "train"`, and the result was discarded.
- The earlier suggestion that a small E6-only training set had outperformed CREMA-D was withdrawn. That result reversed on the larger eight-speaker version and was based on too few test clips to support the claim.
- The short sentence design is also a limitation: many recordings are too brief for humans to interpret emotional intent reliably, and the same wording can be delivered with contradictory emotions in ways that do not generalize clearly across speakers.
- The three-class label space is not always semantically natural for the product setting. A neutral label may absorb states such as calm, resignation, or low-energy speech, while some emotions, such as sarcasm, disappointment, or sadness, may be better treated as separate signals rather than folded into a generic sentiment label.
- For older adult use cases, categories may need to differ from a standard sentiment taxonomy. A more appropriate taxonomy could include states such as positive/energetic, calm/relaxed, angry/disappointed, sarcastic, depressed/sad, crying, or low-voice/weak/illness-related speech patterns.

This is a core issue in the project: the task is not only to classify sentiment, but to define which emotional states are actually relevant to the target user population and to the product goal. The current three-label setup may be too coarse for real-world interaction, especially when the assistant must distinguish between benign calmness, distress, and irritation in older adult speech.

### Why / What did not work

- The recordings are short and many prompts are intentionally contradictory, which makes it difficult even for humans to interpret the intended emotion consistently. This introduces label noise and weakens the learning signal.
- The annotation scheme itself is not fully aligned with the real product goal. A three-class sentiment system (positive / neutral / negative) may be too coarse for a voice assistant for older adults, where emotional states such as calmness, confusion, fatigue, frustration, or distress carry different practical meanings.
- The average speaker may not be the right target for this problem: acted or synthetic contradictory emotions can add artificial or inconsistent delivery, which may distort the true prosodic signal.
- Mapping emotions to sentiment classes is not obvious; the labels must be designed around the target use case, not assumed to be universal.

Example of a more product-relevant taxonomy for elderly users:

- positive / energetic
- calm / relaxed / neutral
- angry / disappointed / frustrated
- sarcastic
- depressed / sad
- crying
- low-energy voice, trembling, coughing, sickness-related vocal changes

---

## 8. Next steps

1. Train on naturalistic, multi-speaker, multi-language speech rather than relying primarily on acted CREMA-D.
2. Add more held-out speakers, not just more clips from the same speaker.
3. Record and evaluate the older adult population the product is intended to support.
4. Investigate a longitudinal per-speaker baseline for devices used by the same person over time.
5. Treat tone–word mismatch as a separate product signal that can trigger clarification instead of a forced label.
6. Explore a more domain-specific emotion taxonomy that reflects the real interaction needs of the user group, rather than assuming that the standard three-class sentiment schema is sufficient.
7. Re-record longer, more natural sentences to reduce ambiguity and make emotional intent easier to interpret.
8. Use a fusion approach that combines acoustic and lexical information, while remaining explicit about when the model should abstain or ask a clarifying question.
9. Test larger public datasets and compare results against realistic, domain-specific recordings from the intended end users.
10. Collect more recordings from the target age group and real-world settings, so the evaluation matches the product context rather than acted or synthetic speech alone.

The current evidence supports a cautious assistant that expresses uncertainty and asks follow-up questions when words and delivery disagree. It does not support presenting the current classifier as a reliable general-purpose sentiment model.

For the full interactive results dashboard and additional evaluation outputs, see: [report/index.html](report/index.html).

The main conclusion is negative but useful:

> Models trained on acted CREMA-D speech do not provide practically useful cross-speaker generalization on the new real-world recordings.

The best cross-speaker result is only just above three-class chance. A model can score well on CREMA-D and still fail on people speaking into laptop microphones.

The clearest diagnostic result is also a warning: the research acoustic model follows the words rather than the delivery on deliberately contradictory speech. An acoustic model is therefore not automatically a prosody model.

---

## 9. Other tests and validation story

This project also included a set of diagnostic tests that were not designed to produce a headline score, but were essential for understanding what was actually being learned and whether the results were trustworthy.

### 8.1 Speaker-disjoint split enforcement

The single most important rule in this project is that no speaker can appear in both the training and evaluation sets. This is not a minor implementation detail; it is the difference between a real generalization test and a leakage-prone benchmark. The repository enforces this with a dedicated split check, and it is treated as a hard invariant.

This is especially important because acted speech datasets such as CREMA-D contain strong speaker-specific patterns. A random clip-level split can make a model look better than it is by letting it memorize the voice rather than the emotion.

### 8.2 Leak detection and the “too-good-to-be-true” rule

Several early runs showed that a model could reach implausibly high scores if the evaluation split was accidentally mixed into training. One E6 retraining experiment produced a UAR of 1.000 because the split column had been ignored. The project contains a built-in plausibility ceiling: speaker-independent performance above about 0.90 is treated as a bug rather than a success, because published state-of-the-art performance on similar tasks is lower.

The result was not hidden; it was examined, fixed, and discarded. This became part of the project’s QA story: the system is designed to flag suspiciously good results, because those are often the first sign of leakage.

### 8.3 Per-speaker normalization as a robustness probe

The project also tested whether per-speaker feature normalization could stabilize prediction under speaker variation. This was a useful experiment because the acoustic features are strongly influenced by the speaker’s baseline voice. Normalization can reduce class collapse by removing some speaker-level drift.

The result was nuanced: it did help, but the effect was much smaller once the eight-speaker real set was included. In other words, per-speaker normalization is a useful mitigation, not a real solution to domain shift. It is also transductive — it requires a pool of that speaker’s audio — which means it is not a generic single-clip classifier solution.

### 8.4 Hume delivery-description probe

A separate Hume probe tested whether explicit delivery instructions change the acoustic output even when the text stays neutral. This was important because it distinguishes “paper emotion label” from “audible prosody.”

The result was clear: explicit delivery descriptions created measurable prosodic differences. Intended-emotion recoverability rose substantially relative to the no-description condition, and the F0 span increased significantly. This gave evidence that the synthetic delivery mechanism could be used as a controlled probe of prosody, but only when checked against the actual audio.

### 8.5 Cartesia probe and the danger of trusting emotion tags

A Cartesia probe showed the opposite story: some requested emotion tags were not rendered reliably, particularly when the text sentiment and the requested emotion conflicted. This matters because it rules out a simplistic assumption that a TTS vendor’s emotion tag is automatically reliable ground truth. Synthetic data is useful for controlled experiments, but not for end-to-end truth claims without audio verification.

### 8.6 E3 repeatability and the stability-vs-correctness lesson

The E3 recordings were also used to test whether repeated takes of the same prompts produced stable predictions. The audeering backend had a very high correlation across the two takes, while the WavLM probe did not. This is a critical lesson: **stability is not the same as correctness**. A model can be highly consistent and still systematically wrong about the signal of interest. This is precisely why the project uses PSI and cross-corpus evaluation rather than accepting repeated agreement as evidence of success.

### 8.7 Why these tests matter

These tests were not decoration. They are the reason the report is careful about its claims. The project does not merely ask “which model performs best on the benchmark?” It asks whether the model is learning the signal that matters, whether it crosses speaker boundaries, whether it follows wording or delivery, and whether a “good” number is actually trustworthy.

That is the deeper story behind the evaluation: the system was not only measured, it was stress-tested against the exact ways in which a speech sentiment model can cheat. The result is a more honest but also less glamorous conclusion: the real bottleneck is not the architecture; it is the data and the mismatch between training and deployment conditions.

---

## 10. Final conclusion

The honest conclusion of this project is that the current pipeline does not yet provide a reliable general-purpose sentiment model for real speech in the intended use case. The best performance remains concentrated on acted or synthetic data, while the real multi-speaker transfer problem remains difficult.

The most useful result is not the benchmark score, but the diagnostic finding: a model can appear acoustic and still follow the wording, and a model that seems consistent can still be wrong about the signal that matters. For a voice-first assistant for older adults, the safer product design is not to force a three-class sentiment label at all costs, but to detect ambiguity, identify mismatch between wording and tone, and ask a clarifying question when the evidence is weak.

That is the most defensible conclusion supported by the data.

For the full interactive results dashboard and additional evaluation outputs, see: [report/index.html](report/index.html).
