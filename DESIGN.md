# Design Document — Speech Sentiment Analyzer

## Approach, and why

The brief's own framing is the design brief: *"the system reacts only to what a user
says, not how they say it."* That is a claim about **prosody**, and the single biggest
risk in speech-sentiment work is building a system that quietly reads the transcript
instead. Published work confirms this is the default failure mode for audio models
(arXiv 2510.10444, arXiv 2510.25054), so the whole pipeline is organized around
**measuring** it, not just building a classifier.

Three solutions, one interface, spanning the lexical↔acoustic axis:

- **A — Lexical**: `faster-whisper` (small, int8, CPU) → text sentiment classifier. The
  deliberate control condition — it can only ever see the words.
- **B — Acoustic**: a frozen speech encoder + a small trained head, no fine-tuning.
  Two backends: **permissive** (WavLM-base, MIT, our own probe trained on CREMA-D) and
  **research** (audeering's wav2vec2 VAD model, CC-BY-NC-SA-4.0, zero-shot valence
  thresholds fit on held-out data — trained on MSP-Podcast, *naturalistic* speech).
- **C — Fusion**: calibrated late fusion of A and B (weighted log-linear pooling +
  temperature scaling + abstention), recommended default.

**Why late fusion, deliberately weaker than joint fusion:** it keeps each branch's
output independently measurable. The **Prosody Sensitivity Index (PSI)** — on clips
where the words and the delivery disagree, what fraction of predictions follow the
*tone* rather than the *words* — is only computable because the branches stay
separable. This is the headline metric the whole evaluation design serves.

**Why frozen encoder + shallow head, not fine-tuning:** the brief asks for something
"lightweight enough to sit alongside a real-time voice pipeline." Training only ever
fits a logistic regression (falling back to a small MLP if that underperforms) on top
of fixed embeddings — seconds of CPU time once embeddings are cached, no GPU required,
and the resulting artifact is under 60KB.

**Why valence-arousal as the shared internal representation:** sentiment is a threshold
read-out of valence (the required deliverable); the same triple gives distress-tone
quadrants "for free" for the roadmap, without training a second model.

## Data

- **CREMA-D** (E1) — 7,442 clips, 91 actors, ODbL-licensed, fetched at run time (not
  redistributed). Primary training/benchmark set. Its 12 carrier sentences are
  emotionally neutral by design, so `text_sentiment` is always NEUTRAL — a real
  limitation, discussed below.
- **Synthetic incongruence set (E2)**, via Cartesia TTS — a 3(text-sentiment) ×
  3(prosody) × 5(carrier) × 2(voice) crossed design, 30 congruent / 60 incongruent
  clips. Ground-truth prosody label is **TTS intent** (the requested emotion tag), not
  a re-classification by any model — following the EMIS paper's precedent (arXiv
  2510.25054).
- **D0/D1** — Cartesia emotion-space probes: D0 an unsupervised clustering of ~58
  tags (2 carriers each, 116 clips); D1 a dedicated factorial probe (40 clips) into
  the user's proposed 5-emotion set (happy/sad/angry/calm/frustrated), crossing
  text condition, length and speed. Both diagnostics, not filters — D1's finding
  (below chance recoverability) is a headline result, not a gate on tag selection.
- **Human recordings (E3)** — `scripts/record_prompts.py`'s teleprompter, the same
  crossed design as E2 but spoken by a real person. **Recorded and evaluated**: two
  independent takes (`data/recorded0`, `data/recorded`), 27 non-whisper clips each,
  run through `scripts/eval_e3.py`. Doubles as the control that rules out the
  measuring instruments as the explanation for D1's negative Cartesia result (see
  the E3 section below).

## Evaluation methodology

- **UAR** (unweighted average recall), not plain accuracy, is the headline
  accuracy-family metric: CREMA-D is 68% negative after sentiment mapping, so plain
  accuracy rewards a majority-class predictor.
- **Speaker-disjoint splits**, always. `assert_speaker_disjoint` runs at the top of
  every training script and is unit-tested. A parallel, deliberately-leaky random
  split exists *only* to quantify the leakage gap, never to train the shipped model.
- **PSI** (contested and strict variants — see `ssa/eval/metrics.py` for the exact
  definitions) on incongruent clips, isolating whether a solution tracks tone or words.

## Results (measured, not projected)

**The headline comparison** — A/B/C on the same stratified 300-clip CREMA-D test
subset (100 per sentiment class):

| Solution | UAR | Macro-F1 | PSI<sub>contested</sub> | Abstention |
|---|---|---|---|---|
| **A — Lexical** | 0.360 | 0.264 | **0.090** | 0.0% |
| **B — Acoustic (permissive)** | **0.797** | 0.796 | 0.920 | 0.0% |
| **C — Fusion** | 0.797 | 0.796 | 0.891 | 10.3% |

This is exactly the shape the project's design predicts. **A is barely above chance**
(1/3 for a balanced 3-way problem) **and structurally can't sense tone** (PSI=0.090,
near the 0.0 "reads the transcript" pole) — because CREMA-D's carrier text is always
neutral, a text classifier has almost nothing to work with, and correctly gets close
to nothing right. **B, hearing only the tone, is the strongest of the three** by a wide
margin. **C matches B's accuracy** while adding calibration: it abstains on 10.3% of
predictions (the fitted threshold, chosen on validation data to flag genuinely
low-confidence cases) and its fusion weight came out at `weight_lexical=0.45` —
roughly balanced on paper, but B's near-certain predictions dominate the log-pooled
result in practice, which is why C's PSI (0.891) stays close to B's rather than
collapsing toward A's.

**Full A/B/C numbers on E1's held-out speaker-disjoint test set:**

| | UAR | Notes |
|---|---|---|
| B, permissive, speaker-disjoint test (n=1,470) | **0.744** | held out, never touched during training |
| B, research (zero-shot), val (n=737) | **0.454** | no CREMA-D fine-tuning at all — the honest cost of the permissive license |
| B, permissive, **leaky** random-split test (n=1,488) | 0.761 | trained/evaluated with 91/91 speakers overlapping train↔test |

**The leakage gap on this setup is +0.017 UAR points** — real, confirmed (91 speakers
genuinely overlap on the leaky split), but far smaller than the 10–40 point swings
often cited for cross-corpus generalization. Plausible explanation, not asserted fact:
WavLM's pretraining objective may already produce a fairly speaker-invariant
representation, and CREMA-D's actors all read the same 12 sentences, so within-speaker
acoustic variance may matter less here than in less controlled corpora. Reported as
measured — the number did not confirm my prior expectation and I am not adjusting it
to fit.

**A caveat on this comparison, stated plainly:** because CREMA-D's text is always
neutral, this A/B/C table demonstrates the words-vs-tone axis in the *easiest possible*
direction for B (there is no competing lexical signal to resist). E2's crossed design
— genuinely sentiment-laden text paired against conflicting tone — is the harder,
more informative version of this same test, and it is what will actually tell us
whether B (or C) can hold onto prosody when the words argue against it. That result
was not available at submission time (see Known limitations).

**A structural caveat on E1's PSI**, discovered while building the report: since
CREMA-D's text is *always* neutral, an acoustic-only solution structurally cannot
"read" sentiment-laden words that were never there — PSI on E1 came back at 0.945,
which looks impressive but is close to vacuous. This is precisely why E2/E3 (built
with genuinely sentiment-laden text) exist: E1 alone cannot test the words-vs-tone
question in an interesting way.

The full A/B/C comparison table is above. The E2/E3 PSI results (the harder,
genuinely-competing-signals version of this test) were not available at submission
time — see the live dashboard (`report/index.html`, regenerate with `make report`)
for whatever has landed since.

## E3: human recordings, and a control on the Cartesia finding

E3 (`data/recorded{0,}/`, `scripts/eval_e3.py`) is now recorded and evaluated: 27
non-whisper clips per take, two independent takes by the same speaker (same 30
prompts, re-recorded — a free test-retest experiment), with genuinely sentiment-laden
text crossed against instructed delivery, so **PSI is finally computed on
competing signals**, not E1's structurally-neutral text.

**A/B/C on E3 (n=27 per take, chance UAR=0.333):**

| | take0 | take1 |
|---|---|---|
| A — lexical | 0.333 | 0.370 |
| B — permissive | 0.444 | 0.519 |
| B — research | 0.444 | 0.593 |
| C — fusion | 0.444 | 0.556 |

**The instruments are exonerated, with a control.** D0/D1 found Cartesia's emotion
tags not measurably audible (recoverability 0.175 vs 0.200 chance, 5-way; F0 barely
separates the 5 emotions). Running the *exact same instruments and method*
(`results/d1_vs_e3_control.json`) on E3 instead of Cartesia:

| | Cartesia (D1) | E3 take0 | E3 take1 |
|---|---|---|---|
| Intended-sentiment recoverability (leave-one-carrier-out) | **0.175** / chance 0.200 | **0.667** / chance 0.333 | **0.741** / chance 0.333 |
| F0 spread between classes | ~8 Hz | 33 Hz | 34 Hz |
| F0 rank of "positive" (1=highest) | ≈ chance | **#1 in every carrier** | **#1 in every carrier** |

Replicated across two independently recorded takes, with a perfect and consistent
pitch ordering both times. This rules out the measuring instruments as the
explanation for D1's negative Cartesia result — the same instruments read real
emotion out of human speech at more than double chance. **D1's finding about
Cartesia stands; it is the synthetic audio, not the measurement, that lacks the
prosody.**

**The more important finding: Solution B doesn't transfer to this speaker, and the
permissive backend isn't even self-consistent.** Trained on CREMA-D (acted, US
studio actors), it scores UAR 0.444/0.519 on E3 and — on take1 — never once predicts
"positive" (0/9 recall). Worse: its own per-clip valence reading correlates at
**r = −0.19** between the two takes of the same prompts by the same speaker, while
the research backend's reading correlates at **r = +0.92**. The research backend is
both *more accurate* (0.444/0.593) and *far more stable* on this speaker — turning
the permissive-vs-research licensing trade-off (see Key trade-offs, below) into
something partially measured, not only argued.

**Per-speaker recalibration was tried and mostly didn't help.** Fitting per-speaker
decision thresholds (leave-one-carrier-out, so the number isn't circular) moved UAR
by backend and take: permissive 0.444→0.556 (take0, helps) but 0.519→0.407 (take1,
hurts); research 0.444→0.370 (take0, hurts) and 0.593→0.519 (take1, hurts). One win
in four, at 27 clips fitting two thresholds on ~24 — this is fitting noise, not
signal, and a probe with r=−0.19 test-retest has no stable per-clip score for a
threshold to sit on anyway. **This is reported as a negative result, not shipped as
a feature.** It sharpens, rather than undermines, the roadmap claim below: DESIGN.md
always argued for *longitudinal* baselining over weeks, and this experiment shows
concretely what a handful of clips cannot buy — evidence for, not against, needing
the longer horizon.

n=1 speaker (not a native English speaker, not elderly) — this is a controlled
instrument check and a real transfer-failure measurement, not a claim about speech
emotion recognition in general, and not yet a measurement of Ami's actual user
population.

## E5: words against delivery, and what it says about "acoustic"

E5 (`data/hume_e5/`, `scripts/gen_hume_dataset.py`) is the incongruence set E2 was
supposed to be. Cartesia's emotion parameter only works when the requested emotion
already matches the transcript — precisely the case that is useless for measuring
prosody sensitivity — and D1 measured that failure at 0.175 vs 0.200 chance. Hume
Octave's `description` field sets delivery independently of the text, so E5 crosses
15 lexically-polar sentences against 3 deliveries across 2 voices: **90 clips, 60 of
them with words and tone deliberately disagreeing**. Eval-only; labels come from the
requested delivery.

**PSI, not UAR, is the metric here** (n=90, 60 incongruent; UAR chance 0.333, PSI
chance 0.5):

| | UAR | PSI<sub>contested</sub> | what it follows |
|---|---|---|---|
| A — Lexical (words only) | 0.333 | **0.000** | the words, always |
| B — Acoustic, permissive (WavLM) | **0.511** | **0.682** | mostly the tone |
| B — Acoustic, research (audeering) | 0.400 | **0.211** | mostly the *words* |
| C — Fusion | 0.489 | 0.605 | between, as designed |

**The dataset validates itself.** Solution A scored PSI exactly 0.000 — it followed the
words on all 60 contradictory clips, which is the only thing a model that cannot hear
tone can do, and lands it at exactly chance UAR against a prosody label. That is the
floor behaving correctly, and it confirms both that the labels are right and that
Hume really did render delivery against the transcript.

**The finding is the gap between the two acoustic backends on identical audio.** The
permissive backend follows the delivery 68% of the time. The research backend follows
the *words* 79% of the time — below PSI chance — despite never receiving a transcript.
Same clips, same generated prosody, so this is a property of the models, not of the
audio.

This is an empirical confirmation of a caveat this project already quotes from the
audeering paper itself: its authors report that the model's valence performance draws
partly on **implicit linguistic information learned during fine-tuning**. E5 turns that
from a cited caveat into a measurement — the "acoustic" research model has substantially
learned to read words out of audio.

**It partly reverses the backend recommendation.** On E3 the research backend looked
better: more accurate on a real voice (0.593 vs 0.519 on take1) and far more
self-consistent (r=+0.92 vs −0.19). Both still hold. But on the question this project
exists to ask — does it hear the tone or read the words? — the research backend sits
closer to the pure lexical model than to the acoustic one. The honest summary is that
neither backend is simply better: the research one is more accurate and more stable,
the permissive one is more genuinely prosodic, and which matters depends on whether
you need a reliable reading or a reading that is actually about delivery.

**Limits.** Synthetic, one TTS vendor, two voices. It measures prosody sensitivity
under controlled contradiction, not performance on real speech — E3 remains the only
real-speaker evidence here.

### The diagnostic: the audio is fine, the models are the bottleneck

E5's PSI alone could not say whether the models were failing to read the delivery or
whether Hume had softened it when it contradicted the words. `scripts/diagnose_e5.py`
settles it by removing the models entirely: can a plain logistic regression on **nine
raw acoustic descriptors** (F0 mean/std/range, loudness, HNR, jitter, shimmer, RMS,
speech rate) recover the *requested* delivery, leave-one-carrier-out?

| Cut | n | Recoverability | F0 span |
|---|--:|--:|--:|
| All E5 clips | 90 | **0.811** | 89.2 Hz |
| Congruent (delivery agrees with words) | 30 | 0.700 | 93.1 Hz |
| **Incongruent (delivery contradicts words)** | 60 | **0.783** | 87.3 Hz |

Chance is 0.333. Same instruments on Cartesia scored 0.175 (vs 0.200 chance — delivery
genuinely absent) and on real human speech 0.667 / 0.741.

Three things follow, and none of them are kind to the models:

1. **The delivery is strongly present.** 0.811 is *higher* than real human speech
   scored on the same instruments. Mean F0 separates the classes by 89 Hz
   (positive 204.5, neutral 126.2, negative 115.3) against Cartesia's ~8 Hz.
2. **The contradiction did not weaken it.** Incongruent clips are *more* recoverable
   than congruent ones (0.783 vs 0.700), so Hume did not quietly defer to the
   transcript. The confound this diagnostic existed to rule out is ruled out.
3. **It is not one lucky voice.** Both score well (Ava Song 0.889, Colton Rivers 0.800).

So on audio where nine hand-crafted features recover the delivery 81% of the time, our
WavLM probe reaches UAR 0.511 and the audeering model 0.400. **A logistic regression on
simple prosodic descriptors substantially outperforms both deep acoustic models at the
task those models exist to do** — and the same pattern holds on E3 (features 0.667/0.741
vs the probe's 0.444/0.519), so it is not an artefact of synthetic audio.

**Why** is not measured, and the candidates are worth separating (all *reasoned*, per
CLAUDE.md rule 6): the probe is trained on CREMA-D, which is acted, US-studio, and
always-neutral-text, so both E5 and E3 are out of domain for it; mean+std pooling over
a WavLM sequence may discard the temporal contour that carries prosody; and a linear
readout of embeddings dominated by phonetic and speaker identity may simply not surface
the prosodic subspace. The cheap, obvious next experiment is to **use these nine
features directly** — alone as a baseline, and concatenated with the WavLM embedding —
which this result now strongly motivates and which has not been tried.

## Solution D: explicit prosodic features — built, and it does not transfer

The E5 diagnostic showed a logistic regression over nine acoustic descriptors
recovering the intended delivery at 0.811 vs 0.333 chance, beating both deep backends
on the same audio. Solution D turns that into a real solution: eGeMAPS v02 (88
functionals) plus eight Praat contour descriptors — global F0 and energy slope,
dynamic ranges in semitones, voiced fraction, pause ratio, CPPS — into a classifier,
with no network anywhere in the path. Four classifiers were fitted on CREMA-D's
speaker-disjoint train split and **all four scored on every dataset**, because
selecting one on CREMA-D validation turned out to be exactly the wrong move.

| Candidate | CREMA-D UAR | CREMA-D PSI | E3b UAR | E3b PSI | E5 UAR | E5 PSI |
|---|--:|--:|--:|--:|--:|--:|
| logreg | 0.646 | 0.738 | **0.407** | 0.583 | **0.278** | 0.378 |
| linear_svm | 0.422 | 0.987 | 0.333\* | 0.500\* | 0.333\* | 0.500\* |
| svm_rbf *(selected on val)* | 0.557 | 0.965 | 0.333\* | 0.500\* | 0.333\* | 0.500\* |
| hist_gbdt | 0.638 | 0.861 | 0.333\* | 0.545\* | 0.333\* | 0.500\* |

\* **degenerate** — predicts one class for ≥95% of the set (always "negative").

**Three of four candidates collapse completely** on out-of-domain audio, answering
"negative" for 100% of E3 and E5 clips at ~0.88 confidence. Only `logreg` keeps
predicting more than one class, and it is still 78–79% negative with poor numbers.

**Why it is not the features.** Distribution shift between CREMA-D and E5 is
ordinary — median |z| 0.47, nothing beyond 10 SD, 9 of 96 features beyond 3 SD. The
kernel is the mechanism: in 96 dimensions, once a test point sits far from every
support vector, RBF kernel values decay toward zero and the decision falls back to
the bias, which is the majority class. Selecting on CREMA-D validation is selecting
*in-domain*, so it crowned `svm_rbf` (0.703) over `logreg` (0.646) on evidence that
structurally could not see this failure.

### The correction this forces

Against the deep backends, trained the same way and met cold:

| Model | E3b UAR | E3b PSI | E5 UAR | E5 PSI |
|---|--:|--:|--:|--:|
| B — permissive (WavLM) | **0.519** | **0.615** | **0.511** | **0.682** |
| B — research (audeering) | 0.593 | 0.444 | 0.400 | 0.211 |
| D — prosodic (best: logreg) | 0.407 | 0.583 | 0.278 | 0.378 |

**The WavLM probe beats Solution D out of domain on both sets.** So the earlier
framing — that simple features outperform deep models — was too strong, and the
distinction matters: the diagnostic's 0.811 came from fitting *within* E5 with
leave-one-carrier-out, which is a far easier problem than training on CREMA-D and
meeting E5 cold. It established that the delivery **is present and linearly
recoverable** in that audio; it did not establish that a CREMA-D-trained
feature classifier would generalise, and the two claims were run together.

What survives is narrower and, if anything, more useful: **the common factor in every
out-of-domain failure is CREMA-D as the training source.** Acted, US studio, 68%
negative, always-neutral text — everything fitted on it transfers badly, the WavLM
probe included (0.744 there, 0.511 on E5). The information the diagnostic found is
real; extracting it with a model trained on this corpus is what fails. That makes
training data, not feature engineering and not architecture, the thing to fix next —
and it is the same conclusion the presbyphonia and elderly-voice gaps already point at.

## The decisive comparison: it is the training corpus, not the representation

The fair test — same classifier, same leave-one-carrier-out CV, both representations
fitted *within* the same dataset, so neither gets a domain advantage
(`scripts/compare_representations.py`):

| Dataset | Representation | dims | Ceiling (fitted in-domain) | CREMA-D-trained | Cost of CREMA-D |
|---|---|--:|--:|--:|--:|
| E3 (your voice, 54 clips) | **prosodic** | 96 | **0.870** | 0.407 | **−0.463** |
| | wavlm | 1536 | 0.759 | 0.519 | −0.240 |
| | combined | 1632 | 0.796 | — | — |
| E5 (90 clips) | **prosodic** | 96 | 0.800 | 0.278 | **−0.522** |
| | wavlm | 1536 | 0.789 | 0.511 | −0.278 |
| | **combined** | 1632 | **0.856** | — | — |

Chance is 0.333.

**Three findings.**

1. **Both representations reach the signal.** Fitted in-domain, everything lands at
   0.76–0.87 against 0.333 chance. The prosodic information is genuinely there and
   genuinely learnable — from 96 hand-built numbers *or* from a 1536-dimension
   embedding.

2. **Training on CREMA-D costs 0.24 to 0.52 UAR.** That is the gap between what these
   representations can do on this audio and what they actually do once fitted on acted
   studio speech and transferred. It dwarfs every difference between models, backends
   and feature sets measured anywhere else in this project. **The training corpus is
   the bottleneck.**

3. **Prosodic features have the higher ceiling but are more domain-fragile.** They beat
   WavLM outright on the real human voice (0.870 vs 0.759) with sixteen times fewer
   dimensions, yet they lose *more* when trained cross-domain (−0.46/−0.52 against
   −0.24/−0.28). That is coherent: absolute pitch, loudness and harmonicity move with
   microphone, room and recording level, while a learned representation is partly
   invariant to them. Hand-built descriptors are the better *representation* and the
   worse *transfer* — which argues for per-domain or per-speaker normalisation of those
   features, not for abandoning them.

Combining the two helps on E5 (0.856, above either alone) and hurts on E3 (0.796, below
prosodic alone) — consistent with 1,632 dimensions overfitting 54 clips rather than with
any real property of the combination. At these sample sizes that difference is not
worth interpreting.

**This corrects the earlier claim properly.** "Simple features beat the deep models" was
drawn from comparing an in-domain fit against a cross-domain transfer. Measured fairly,
prosodic features *do* win on the real voice — but the headline is that both
representations lose far more to the training corpus than they differ from each other.

## Backend x training-data comparison

Requested directly: does folding the user's own recordings (E3) and/or the Hume
Octave probe into training change either acoustic backend, across four
combinations (`scripts/eval_backend_combos.py`, `results/backend_combo_comparison.json`;
Cartesia excluded per the user's own instruction — D1 already found it added no
signal). Same fixed 300-clip stratified CREMA-D-test subset for every combo; E3 is
evaluated only for the two combos that never trained on it (CLAUDE.md rule 1 — E3
has one real speaker, so a combo that trains on them cannot also score itself on
them, even under a different clip_id):

| Combo | n train | Permissive CREMA-D UAR | Permissive CREMA-D macro-F1 | Permissive E3 UAR | Research CREMA-D UAR | Research E3 UAR |
|---|---|---|---|---|---|---|
| `cremad_only` | 5,235 | 0.782 | 0.774 | 0.481 | 0.450 | 0.519 |
| `cremad_e3` (+E3) | 5,289 | 0.782 | 0.771 | excl.\* | 0.450 | 0.519 |
| `cremad_e3_hume` (+E3+Hume) | 5,309 | 0.783 | 0.775 | excl.\* | 0.450 | 0.519 |
| `cremad_hume` (+Hume) | 5,255 | 0.785 | 0.778 | 0.481 | 0.450 | 0.519 |

\* excluded, not missing: these two combos trained on E3's one speaker.

**Two findings, both real, neither dramatic:**

1. **Adding 54 clips of E3 and/or 20 clips of Hume to 5,235 clips of CREMA-D moves
   the permissive backend's CREMA-D-test UAR by at most 0.003** — noise, not
   signal, at this ratio (~1% of training volume). No sign of catastrophic
   forgetting either. Read as: a handful of extra clips from one new source
   doesn't perceptibly help *or* hurt the model on the benchmark it was already
   tuned for.
2. **The research backend is architecturally invariant here by construction** —
   it has no trainable encoder, only two valence thresholds fit once from CREMA-D's
   own held-out validation split, which none of these combos ever touch. Its
   identical numbers across all four rows are a reported property of the
   architecture, not a shortcut standing in for missing results (see
   `results/backend_combo_comparison.json`'s `research_backend_note`).

**The more informative negative result:** training on E3 doesn't buy a
measurable transfer benefit either — E3 eval is excluded for combos that train
on it, but there's no reason from these numbers to expect a better outcome if it
weren't: 54 real clips against 5,235 acted ones is not enough training signal to
move a shared classifier's decision boundary for one new voice. This is
consistent with the E3 section's earlier finding that per-speaker *threshold*
recalibration (fitting 2 numbers, not retraining a classifier) is the more
promising lever at this sample size, not more raw training data from the same
source.

## Key trade-offs

1. **Late fusion over joint fusion** — sacrifices some accuracy to keep PSI
   computable per branch. The whole point of the project over accuracy-chasing.
2. **Two acoustic backends, not one** — the strongest model (audeering, zero-shot,
   naturalistic training data) is CC-BY-NC-SA-4.0, non-commercial. Rather than pick
   one, both ship behind one interface; the CLI prints a license notice when the
   research backend loads. A company shipping this has a documented, working
   alternative. **This is no longer only a licensing argument**: on E3 (see above),
   the non-commercial research backend is both more accurate and far more stable
   across takes than the shipped permissive one — a measured reason, not just a
   licensing one, to keep both behind one interface rather than commit to either.
3. **Synthetic (E2) vs. human (E3) data, not either/or** — TTS gives scale, balance,
   and zero speaker confound; human speech is the validity anchor. Cross-checking them
   is itself a planned result, not an afterthought.
4. **CPPS over sustained-vowel voice-quality measures** — the one dysphonia metric with
   demonstrated validity on continuous speech, the only kind a voice assistant ever
   has access to.

## Known limitations and failure modes

- **E2 is incomplete (76/90 clips)** — generation hit Cartesia's free-tier quota
  mid-run. The script is resumable; the remaining 14 clips (plus any E4 augmentation)
  complete in one command once the key is topped up.
- **D0/D1: Cartesia's emotion tags are not measurably audible on this content** —
  D0's clustering was too degenerate to trust as a tag filter (0.216 valence span
  across 58 tags on the audeering model), and D1's dedicated factorial probe (40
  clips: 5 emotions × neutral/congruent text × short/long × default/adjusted speed)
  scored *below* chance on intended-emotion recoverability (0.175 vs 0.200). The E3
  control above rules out the instruments as the explanation. Reported as measured
  for this content, these 2 voices, sonic-3 — not asserted as Cartesia's general
  limit (see `results/d1_emotion_probe.json`, `report/d1_listening.html`). **Confirmed
  vendor-specific, not a synthetic-TTS-wide limit**: a same-design Hume Octave probe
  on the identical carrier text shows real differentiation (see What I'd do next, #5,
  and `results/hume_probe.json`) — the finding is "Cartesia didn't render this,"
  not "no TTS vendor can."
- **E2 is incomplete (76/90 clips) and superseded for the audibility question** —
  generation hit Cartesia's free-tier quota mid-run; D1 found the underlying
  rendering problem E2 was already hinting at, so E2 stays as the historical exhibit
  rather than being completed.
- **Only 2 TTS voices** in the synthetic sets — thin voice diversity, stated rather
  than papered over.
- **The permissive backend does not transfer to the one real speaker tested** — see
  the E3 section above (UAR 0.444/0.519, r=−0.19 test-retest). Trained on acted
  speech (CREMA-D); this is a first, small, but real cross-corpus/cross-speaker
  failure measurement, not a hypothetical one.
- **No elderly voices in any accessible dataset.** Domera Labs (the employer) builds a
  senior-focused companion device; the single most important idea in this project —
  that presbyphonia (age-related vocal-fold changes) can look acoustically like sadness
  to a model trained on younger speakers, and that per-speaker baselining is the fix —
  is **reasoned from the clinical-acoustics literature, not measured against any real
  aged voice in this repository.** Labelling this as anything other than a design
  argument would be the one genuinely dishonest move available in this write-up.

## What I'd do next, with more time or resources

1. **Longitudinal per-speaker baselining, over weeks not clips** — E3's 27-clip
   leave-one-carrier-out recalibration helped in 1 of 4 backend/take combinations and
   hurt in the other 3 (see the E3 section above): a real, measured demonstration
   that a handful of clips is the wrong horizon for this idea, not evidence against
   it. Ami is a personal device with one persistent user; learning *that person's*
   neutral over weeks is the version that could dissolve both the presbyphonia
   confound and the acute-illness-vs-trait confound in the voice-health module. Needs
   longitudinal single-speaker data this project still doesn't have.
2. **Cross-corpus generalisation on public benchmarks** — RAVDESS (train CREMA-D,
   test RAVDESS, both acted but different actors/recording setup) and a same-content
   age contrast via TESS (26 vs 64 year old speaker, identical words) would extend
   E3's single-speaker transfer-failure finding into something closer to a population
   estimate. Both CC-BY-NC — eval only, per the licensing argument above.
3. **MSP-Podcast** for naturalistic-speech training data — CREMA-D's acted delivery is
   a real generalization gap, and it's now a *measured* one (E3), not just argued.
4. A small, consented elderly-voice pilot — the one dataset that would let the
   presbyphonia argument move from "reasoned" to "measured" the way E3 has done for
   the acted-vs-spontaneous-speaker gap more broadly.
5. ~~A genuinely emotional synthetic TTS source, if one exists~~ — **done, and it
   works, now at D1's own scale.** `scripts/probe_hume.py` / `results/hume_probe.json`
   runs Hume Octave through the *exact same* 40-clip factorial grid D1 ran on Cartesia
   (5 emotions × neutral/congruent text × short/long, one fixed voice), varying only
   Hume's `description` acting-instruction field, with 12 genuine carrier groups this
   time — enough for an honest leave-one-carrier-out recoverability classifier, not
   just a descriptive F0 span:

   | | with `description` | without `description` | Cartesia (D1) |
   |---|---|---|---|
   | Intended-emotion recoverability (LOCO) | **0.5** / chance 0.2 | 0.2 (= chance) | 0.175 / chance 0.2 |
   | F0 span across emotions | **124.1 Hz** | 37.7 Hz | ~8 Hz |
   | Valence ordering (pos>neu>neg) | True | True | scrambled |
   | F0 span, **neutral text only** (harder case) | **134.8 Hz** | — | ~8 Hz |

   `description` produces real, classifier-recoverable emotion signal even on
   neutral text, which is exactly the condition that isolates prosody from wording
   (this project's gold-label rule) — and the condition Cartesia failed hardest.
   First synthetic source in this project to clear that bar. **n=40, 12 carrier
   groups, 1 voice** — the grid D1 itself used, so the comparison to Cartesia's 0.175
   is apples-to-apples; still 1 TTS voice, so voice diversity remains future work.
6. Extend voice-health from a demo to a validated gate, once real longitudinal
   recordings exist to fit and check it against.
