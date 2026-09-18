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
   works.** A 10-clip Hume Octave probe (`scripts/probe_hume.py`, `results/hume_probe.json`),
   same design as D1 (same 5 emotions, same neutral carrier text, one fixed voice),
   varying only Hume's `description` acting-instruction field: F0 span **126.7 Hz**
   with `description` vs **32.3 Hz** without (Cartesia/D1: ~8 Hz), and the correct
   positive>neutral>negative valence ordering holds *only* with `description`. First
   synthetic source in this project to show a working emotion-rendering signal — see
   `report/index.html`'s Hume section. **n=10, 1 carrier, 1 voice** — real, but a
   fuller grid (matching D1's 8-cell design) is the next step before trusting this for
   a shipped dataset, which remains future work.
6. Extend voice-health from a demo to a validated gate, once real longitudinal
   recordings exist to fit and check it against.
