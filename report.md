# Speech Sentiment Analyzer

### AI/ML Systems Engineer Take-Home Assignment

## 1. Overview

I built a local Python pipeline that takes raw speech audio and predicts **positive, neutral, or negative sentiment**. The main goal was not to maximize a single benchmark score, but to understand a harder practical question:

> **Can sentiment be inferred from how something is said, rather than simply what is said?**

I therefore evaluated several complementary approaches:

* **Lexical:** ASR with Whisper followed by a text sentiment model.
* **Acoustic:** a frozen speech representation with a lightweight classifier.
* **Fusion:** calibrated late fusion of lexical and acoustic predictions, with abstention.
* **Prosodic:** explicit acoustic/prosodic features such as pitch, energy and eGeMAPS.

The pipeline includes speaker-disjoint evaluation, calibration metrics, robustness experiments, and specifically designed **incongruent speech tests**, where the words and delivery express conflicting sentiment.

The main findings were:

1. **Nothing trained on CREMA-D generalizes to real multi-speaker audio.** On 160 clips
   from eight speakers recording through laptop microphones, only one of five systems has
   a UAR confidence interval clearing chance, and it clears it by 0.003. The acoustic
   system that reaches 0.797 on the benchmark reaches 0.396 [0.33, 0.46] there. This is
   the central result and it is a negative one.
2. **The training corpus is a far larger bottleneck than the choice of model,
   representation or feature set.** Fitting within a dataset reaches 0.76–0.87;
   transferring from CREMA-D costs 0.24–0.52 UAR — more than any architectural difference
   measured here.
3. **An "acoustic" model is not necessarily a prosodic one.** One pretrained speech
   representation follows the *words* rather than the delivery on deliberately
   contradictory clips, from audio alone with no transcript in its path. On the
   eight-speaker set its entire confidence interval sits below chance. This replicates
   across two independent datasets and is the sharpest diagnostic finding in the project.
4. **Detecting a tone/words mismatch is easier than naming the sentiment**, and the two
   come apart in the results. For an assistant that should ask rather than assume under
   ambiguity, the mismatch detector may be the more useful and more attainable component.
5. **Per-speaker feature normalization reliably stops models collapsing to a single
   class, but its accuracy benefit shrank by a factor of three to four once eight real
   speakers were available** (+0.05 to +0.09 UAR, against +0.19 to +0.33 measured on one
   speaker and two synthetic voices). It is also transductive, requiring a pool of the
   speaker's audio that a single-clip classifier does not have. It is reported as an
   experiment, not a shipped feature.
6. Synthetic emotional speech is useful as a diagnostic, but not as a substitute for
   diverse real conversational speech — one vendor's emotion tags proved not to be
   reliably audible at all.

Two results in this report were **withdrawn or narrowed when more data arrived**, and
both are described where they occur rather than quietly dropped. Small held-out sets
produce confident-looking numbers that do not survive, which is why every figure on the
multi-speaker set carries a confidence interval.

The current implementation runs locally, with an automated test suite, a command-line
evaluation pipeline, and a browser-based FastAPI demo with microphone capture.

---

# 2. Problem framing and approach

A speech sentiment system can exploit several sources of information:

$$
\text{Audio} \rightarrow
\begin{cases}
\text{linguistic content} \\
\text{prosody/acoustics} \\
\text{speaker characteristics}
\end{cases}
\rightarrow \text{sentiment}
$$

This creates an important ambiguity for the intended product.

If a user says *"That's great"* in an angry voice, a system that predicts positive sentiment purely from the words is not necessarily useful for a conversational assistant. Conversely, a system that relies only on pitch or energy may fail when sentiment is primarily expressed linguistically.

I therefore treated the assignment as both a **classification problem** and a **signal attribution problem**.

### Solution A — ASR + text sentiment

Audio is transcribed using Whisper and passed to a pretrained RoBERTa sentiment classifier.

This provides a useful baseline and establishes how much of the benchmark can be explained by linguistic content alone.

However, it does not satisfy the deeper product requirement by itself: the model effectively sees *what was said*, not *how it was said*.

### Solution B — acoustic representation + lightweight classifier

I extracted frozen speech representations and trained lightweight classifiers on top of them.

Two pretrained backends were evaluated:

* a WavLM-based representation with a permissive license suitable for deployment;
* an audeering speech-emotion representation, used as a research comparison.

Keeping the representation frozen made experimentation inexpensive and allowed the comparison to focus on the information contained in the representation rather than on large end-to-end fine-tuning runs.

### Solution C — calibrated late fusion

I combined lexical and acoustic predictions using late fusion.

The motivation was practical: linguistic and acoustic signals are complementary, and a conversational system may benefit from using both.

I also implemented calibration and an abstention mechanism because a production voice assistant should not necessarily force a sentiment decision when the evidence is ambiguous.

### Solution D — explicit prosodic features

Finally, I extracted interpretable acoustic features, including eGeMAPS/Praat-derived measurements such as pitch-related and energy-related characteristics, and trained conventional classifiers.

This approach is lightweight and interpretable and provides a useful control against relying entirely on large pretrained representations.

---

# 3. Data

I used several datasets for different purposes rather than treating one benchmark as representative of the complete problem.

### CREMA-D

CREMA-D contains 7,442 clips from 91 speakers and provides the main supervised training/evaluation corpus.

I used a **speaker-disjoint split** so that speakers appearing in the training set do not appear in the test set.

This is important because a random clip-level split can allow a model to exploit speaker-specific characteristics rather than learning sentiment.

The full speaker-disjoint test split contains approximately 1,470 clips.

### Human recordings

I additionally created a small controlled recording set (E3) containing two takes of the same 27 evaluation prompts.

This was intended as a test of whether conclusions from CREMA-D transfer to independently recorded speech and whether predictions are stable across repeated takes.

The current limitation is that E3 contains only one human speaker. Every conclusion
drawn from it is therefore about one voice, and this was the single largest gap in the
evaluation until E6 closed it.

### Zurich multi-speaker recordings (E6)

E6 is the evaluation set that makes cross-speaker claims possible at all. It contains
**160 clips from 8 speakers** reading 35 short sentences, where each sentence was
recorded with two or three *different intended deliveries*. Because the same wording
appears under conflicting deliveries, **111 of the 160 clips (69%) are incongruent** —
making this the first *human, multi-speaker* incongruence set in the project. E5 provided
incongruence but synthetically; E3 provided real speech but from one person.

The split is speaker-disjoint by construction: five speakers train, two validate, one
tests. The recordings are laptop-microphone audio from mostly non-native English
speakers, which is a substantially harder and more realistic condition than CREMA-D's
acted American studio speech.

Two properties of E6 require disclosure:

* **The lexical valence is assigned by hand.** The dataset labels intended *delivery*
  only. PSI additionally needs to know what the words say, so all 35 sentences were
  labelled from the text alone, with genuinely two-sided wordings (*"Whatever, it's
  fine."*) left neutral rather than forced to a side. Forcing a side would manufacture
  incongruence the text does not contain, and PSI would then measure the labelling rather
  than the model. This is the one derived column in the dataset.
* **One E6 speaker also recorded E3.** He carries the same speaker identifier in both, so
  the speaker-disjointness assertion can detect the overlap rather than relying on anyone
  remembering it. No model trained on E6 is evaluated on E3.

### Hume recordings

I created E5 using Hume-generated speech, including **60 deliberately incongruent clips**, where linguistic sentiment and vocal delivery were designed to disagree.

This set was designed primarily as a diagnostic rather than as a conventional benchmark.

It allows the following question to be tested:

> When words and delivery disagree, which signal does the model follow?

### Synthetic TTS experiments

I also tested Cartesia and Hume-generated emotional speech.

These experiments revealed that not all synthetic voices reliably express the requested emotional/prosodic condition. I therefore used synthetic data primarily as a controlled probe rather than assuming that an emotion label attached to generated speech represents ground truth.

---

# 4. Evaluation methodology

I used several complementary metrics.

### Unweighted Average Recall (UAR)

UAR averages recall across classes and is less sensitive to class imbalance than raw accuracy.

Chance performance for the three-class problem is approximately:

$$
UAR_{chance}=0.333
$$

### Macro-F1

Macro-F1 provides another class-balanced measure of classification performance.

### Calibration

Expected Calibration Error (ECE) was used to evaluate whether predicted probabilities correspond reasonably to observed correctness.

This matters because a conversational system may use sentiment probabilities to determine how strongly it should adapt its response.

### Prosody-vs-language diagnostic

For the incongruent sets (E5 and E6) I introduced a simple **Prosody Sensitivity Index
(PSI)**:

* 1.0 = prediction follows the intended vocal delivery;
* 0.0 = prediction follows the linguistic sentiment;
* approximately 0.5 = no systematic preference.

This is not intended as a standard ML metric. It is a diagnostic specifically designed to answer the product question of whether a model is actually sensitive to vocal delivery.

### Uncertainty

Held-out speaker sets here are small — E6's test split is 20 clips from one person, so
each class's recall rests on five to nine clips. Every E6 figure therefore carries a 95%
percentile bootstrap interval over clips. Where two intervals overlap, I report the
comparison as unresolved rather than as a result; several comparisons below fall into
that category, and saying so is the point.

The interval is a floor on the uncertainty, not a full account of it: resampling clips
treats them as exchangeable when the same speaker and the same 35 sentences recur, so the
true interval is if anything wider.

---

# 5. Results

## 5.1 Main model comparison

Values are UAR. CREMA-D is the 300-clip stratified test subset; E3 is one speaker
(two takes); E5 is synthetic; **E6 is 160 clips from eight real speakers**.

| Model                |   CREMA-D |   E3a |       E3b |        E5 | **E6 (n=160)** |
| -------------------- | --------: | ----: | --------: | --------: | -------------: |
| Lexical              |     0.360 | 0.333 |     0.370 |     0.333 | 0.314 [0.25, 0.39] |
| Acoustic — WavLM     | **0.797** | 0.444 |     0.519 | **0.511** | 0.396 [0.33, 0.46] |
| Acoustic — audeering |         — | 0.444 | **0.593** |     0.400 | 0.356 [0.29, 0.43] |
| Late fusion          | **0.797** | 0.444 |     0.556 |     0.489 | **0.402 [0.34, 0.47]** |
| Explicit prosody     |     0.566 | 0.333 |     0.333 |     0.333 | 0.333 [0.33, 0.33] |

On the full CREMA-D speaker-disjoint test split the WavLM-based acoustic system achieved
**0.744 UAR**. The difference from 0.797 is that the latter is measured on the 300-clip
subset used by the comparison harness; the full split is the headline CREMA-D result.

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

# 6. What the experiments taught me

## 6.1 Acoustic representations are not necessarily purely prosodic

This is the strongest diagnostic result in the project, and it now holds on two
independent datasets.

On E5's 60 contradictory synthetic clips:

* WavLM achieved PSI = **0.682**
* the audeering representation achieved PSI = **0.211**
* the lexical baseline achieved PSI = **0.000**

On E6's 111 contradictory clips from eight real speakers, with 95% intervals:

* lexical baseline: PSI = **0.104 [0.047, 0.170]** — entirely below chance
* audeering: PSI = **0.366 [0.265, 0.472]** — **entirely below chance**
* WavLM: PSI = 0.584 [0.472, 0.697] — contains chance
* late fusion: PSI = 0.557 [0.443, 0.667] — contains chance

The lexical baseline is the sanity check: on deliberately contradictory examples it
follows the words, as it must. Because E6's lexical labels are the one column I assigned
by hand, this also validates that labelling.

The finding is that **the audeering representation also follows the words**, from audio
alone, with no transcript anywhere in its path — and on E6 its entire confidence interval
sits below chance. Pretrained speech representations can encode linguistic information
implicitly, and calling a model "acoustic" does not establish that it solves the prosodic
problem. For a product whose purpose is detecting vocal affect, this distinction is the
difference between a system that works and one that appears to.

**What I am careful not to claim:** that WavLM is prosody-sensitive. Its E6 interval
contains chance. The *direction* is consistent across E5 and E6, but only the audeering
failure is statistically clean. On the smaller six-speaker build I briefly had this as a
clear WavLM win; the intervals on the full set say the two representations merely touch.

## 6.2 PSI and UAR measure different things, and come apart

On E6 the WavLM probe is at chance on *which* sentiment a clip carries while its PSI
point estimate sits above chance. Recognising that a delivery contradicts the wording is
an easier problem than naming the emotion.

This has a product consequence. For an assistant whose correct behaviour under ambiguity
is to *ask* rather than assume, a reliable "the tone doesn't match the words" detector may
be both more useful and more attainable than a sentiment classifier — and it is the part
of the signal that survives here.

## 6.3 Stability and generalization are different properties

The audeering representation produced highly correlated predictions across the two E3
recordings of the same prompts:

$$
r = 0.921
$$

The corresponding correlation for the WavLM probe was:

$$
r = -0.191
$$

The two representations behave very differently under repeated recording conditions. This
should not be read as audeering being generally superior: the experiment contains one
human speaker, and E5 and E6 both show the same model following words rather than tone. A
model can be highly self-consistent and consistently wrong about the thing you care
about, which is the useful lesson here.

## 6.4 The training corpus dominates everything else

Across the representation comparison, the backend comparison, the four-way prosodic model
comparison and now E6, the same pattern recurs: differences between architectures,
representations and feature sets are small next to the difference between training
corpora. Fitting within a dataset reaches 0.76–0.87; transferring from CREMA-D to
anything else costs 0.24–0.52 UAR. That gap is larger than any modelling choice measured
in this project.

---

# 7. Limitations

I would rather state these precisely than gesture at them, because several of them bound
the conclusions above more tightly than the numbers suggest.

### Sample sizes are small where it matters most

CREMA-D is large (7,442 clips, 91 speakers) and is the only set where a result is
statistically comfortable. Every set that tests the interesting question is small: E3 is
one speaker, E5 is two synthetic voices, and E6 — the largest real-speaker set — has 160
clips with only 20 in the held-out test split. Comparisons between retrained models on
that split cannot be resolved, and I report them as unresolved rather than ranking them.

### One derived label

E6's lexical valence is my own judgment, not the dataset's (see §3). PSI on E6 therefore
depends on that labelling in a way it does not for E5, where the text valence was fixed
by design before any audio was generated. The lexical baseline's PSI acts as a check on
it: a transcript-only model should follow the transcript, and it does.

### The population the product targets is still unmeasured

The intended deployment is a voice companion for seniors. CREMA-D's actors are not
elderly, and E3/E6 are working-age adults, mostly non-native English speakers. The
presbyphonia argument in the design document — that age-related voice changes will shift
the acoustic baseline — remains **reasoned from literature, not measured on our data**.
Nothing here tests it, and I have deliberately not presented it as though it does.

### A leakage bug, and what it says about the rest

During the E6 work, one retraining combination reported UAR 1.000 on its validation
speaker. The cause was that the feature-fitting path ignored the split column and trained
on the evaluation speaker. It was caught only because the project fixes a plausibility
ceiling — speaker-independent results above 0.90 are treated as bugs, since published
state of the art sits below it — and shouts when a number exceeds it.

I record this because it is the honest lesson of the whole exercise: **the failure mode
of this kind of work is a number that looks like good news.** Every other guard in the
repository exists for the same reason, and the leak surfaced as an unusually good result
rather than as an error.

### Synthetic speech is a diagnostic, not a substitute

One TTS vendor's emotion tags proved not to be reliably audible on this content
(below-chance recoverability), while another's produced real differentiation. Generated
audio was therefore used to construct controlled contradictions, never as a stand-in for
diverse real conversational speech. E6 exists precisely because that substitution would
not have been sound.

---

# 8. What I would do next

In the order I would actually do them:

1. **Change the training corpus, not the model.** Every experiment points the same way:
   the gap between corpora dwarfs the gap between architectures, representations and
   feature sets. I would train on spontaneous in-the-wild speech (MSP-Podcast is the
   obvious candidate; it requires an access request) rather than acted studio speech, and
   expect that to move the numbers more than any modelling change tried here.

2. **Collect more held-out speakers, not more clips per speaker.** The binding constraint
   on every cross-speaker claim is the number of *people*, not the number of recordings.
   Twenty speakers with ten clips each would be worth more than the reverse.

3. **Record the actual target population.** Nothing in this project measures elderly
   voices. Until it does, the presbyphonia argument stays a hypothesis, and a system
   shipped to seniors would be extrapolating.

4. **Make per-speaker baselining longitudinal.** Per-speaker feature normalisation was
   the largest single intervention measured, but it is transductive — it needs a pool of
   that speaker's audio, which a single-clip classifier does not have. For a companion
   device used daily by one person, a running baseline accumulated over weeks is the
   natural form of it, and is also the version the design document argues for.

5. **Treat "does the delivery contradict the words?" as its own product signal.** PSI and
   UAR come apart in the results: a model can be at chance on naming the sentiment while
   still beating chance on detecting a mismatch between tone and wording. For an
   assistant whose correct response to ambiguity is to *ask* rather than assume, the
   mismatch detector may be the more useful and more attainable component.
