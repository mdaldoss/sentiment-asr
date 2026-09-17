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
- **D0** — an unsupervised probe of Cartesia's ~58 emotion tags (2 carriers each, 116
  clips), embedded with the audeering VAD model and clustered. Not used to gate E2's
  tag selection (see Limitations) — it's a diagnostic, not a filter.
- **Human recordings (E3)** — a teleprompter script (`scripts/record_prompts.py`)
  records the same crossed design from real speakers, as the validity anchor for E2's
  synthetic data.

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

**A structural caveat on E1's PSI**, discovered while building the report: since
CREMA-D's text is *always* neutral, an acoustic-only solution structurally cannot
"read" sentiment-laden words that were never there — PSI on E1 came back at 0.945,
which looks impressive but is close to vacuous. This is precisely why E2/E3 (built
with genuinely sentiment-laden text) exist: E1 alone cannot test the words-vs-tone
question in an interesting way.

*A full A/B/C comparison table on a matched subset, and the E2/E3 PSI results, were
still completing at submission time — see the live dashboard
(`report/index.html`, regenerate with `make report`) for whatever had landed.*

## Key trade-offs

1. **Late fusion over joint fusion** — sacrifices some accuracy to keep PSI
   computable per branch. The whole point of the project over accuracy-chasing.
2. **Two acoustic backends, not one** — the strongest model (audeering, zero-shot,
   naturalistic training data) is CC-BY-NC-SA-4.0, non-commercial. Rather than pick
   one, both ship behind one interface; the CLI prints a license notice when the
   research backend loads. A company shipping this has a documented, working
   alternative.
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
- **E3 has not been recorded** — requires a human at a microphone; the teleprompter
  script is built and tested (prompt construction, not the live recording loop, which
  needs real hardware) but no clips exist yet.
- **D0's clustering was too degenerate to trust as a tag filter.** The audeering VAD
  model — trained on real speech — showed only a 0.216 valence span across all 58
  Cartesia tags, and many tags' two carrier-instances split across different clusters.
  Verified this is a genuine domain-gap finding, not a plumbing bug (duration and RMS
  energy vary substantially and sensibly by tag), then changed plan: E2's tags are
  drawn directly from the mapping table, not gated by this clustering.
- **Only 2 TTS voices** in the synthetic sets — thin voice diversity, stated rather
  than papered over.
- **The permissive backend is trained on acted speech (CREMA-D)** and has not been
  validated on natural, spontaneous speech at all.
- **No elderly voices in any accessible dataset.** Domera Labs (the employer) builds a
  senior-focused companion device; the single most important idea in this project —
  that presbyphonia (age-related vocal-fold changes) can look acoustically like sadness
  to a model trained on younger speakers, and that per-speaker baselining is the fix —
  is **reasoned from the clinical-acoustics literature, not measured against any real
  aged voice in this repository.** Labelling this as anything other than a design
  argument would be the one genuinely dishonest move available in this write-up.

## What I'd do next, with more time or resources

1. Complete E2/E4 and record E3; run the full PSI comparison across all three solutions.
2. **Per-speaker longitudinal baselining** — the highest-value next step for the actual
   product. Ami is a personal device with one persistent user; it can learn *that
   person's* neutral over weeks, which dissolves both the presbyphonia confound and the
   acute-illness-vs-trait confound described in the voice-health module, using the same
   mechanism. This needs longitudinal single-speaker data this project doesn't have.
2. **MSP-Podcast** for naturalistic-speech training data — CREMA-D's acted delivery is
   a real generalization gap.
3. A small, consented elderly-voice pilot — the one dataset that would let the
   presbyphonia argument above move from "reasoned" to "measured."
4. Extend voice-health from a demo to a validated gate, once E3 (and ideally real
   longitudinal recordings) exist to fit and check it against.
