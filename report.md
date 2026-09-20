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

1. **Acoustic information contains useful sentiment signal**, but performance drops substantially when moving from CREMA-D to independently recorded/generated speech.
2. **The training corpus appears to be a larger bottleneck than the choice between the tested acoustic representations.**
3. **Per-speaker normalization substantially improved cross-speaker performance in the experiments**, although this result is not yet sufficient to claim generalization because the external evaluation currently contains only three speakers/voice identities.
4. The distinction between an **acoustic representation** and a genuinely **prosodic representation** matters: one of the tested pretrained models showed strong sensitivity to lexical content despite receiving only audio.
5. Synthetic emotional speech is useful as a diagnostic tool, but I would not treat it as a substitute for diverse real conversational speech.

The current implementation runs locally and contains 407 automated tests, a command-line evaluation pipeline, and a browser-based FastAPI demo with microphone capture.

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

The current limitation is that E3 contains only one human speaker.

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

For the incongruent E5 set I introduced a simple **Prosodic Signal Index (PSI)**:

* 1.0 = prediction follows the intended vocal delivery;
* 0.0 = prediction follows the linguistic sentiment;
* approximately 0.5 = no systematic preference.

This is not intended as a standard ML metric. It is a diagnostic specifically designed to answer the product question of whether a model is actually sensitive to vocal delivery.

---

# 5. Results

## 5.1 Main model comparison

The main results were:

| Model                |   CREMA-D |   E3a |       E3b |        E5 |
| -------------------- | --------: | ----: | --------: | --------: |
| Lexical              |     0.360 | 0.333 |     0.370 |     0.333 |
| Acoustic — WavLM     | **0.797** | 0.444 |     0.519 | **0.511** |
| Acoustic — audeering |         — | 0.444 | **0.593** |     0.400 |
| Late fusion          | **0.797** | 0.444 |     0.556 |     0.489 |
| Explicit prosody     |     0.566 | 0.333 |     0.333 |     0.333 |

Values are UAR.

On the full CREMA-D speaker-disjoint test split, the WavLM-based acoustic system achieved **0.744 UAR**.

The difference between the 0.797 result and 0.744 is due to the former being measured on a smaller 300-clip subset used in the comparison harness. The full split is the headline CREMA-D result.

The important observation is not the absolute CREMA-D number. It is the substantial reduction when evaluating independently recorded/generated speech.

---

# 6. What the experiments taught me

## 6.1 Acoustic representations are not necessarily purely prosodic

The strongest diagnostic result came from the E5 incongruence set.

On the 60 contradictory clips:

* WavLM achieved PSI = **0.682**.
* The audeering representation achieved PSI = **0.211**.
* The lexical baseline achieved PSI = **0.000**.

The lexical baseline is an important sanity check: on intentionally contradictory examples, it follows the words rather than the vocal delivery.

More surprisingly, the audeering representation also showed strong sensitivity to lexical content despite receiving audio rather than a transcript.

This suggests that pretrained speech representations can encode linguistic information implicitly. Therefore, simply calling a model "acoustic" does not establish that it is solving the desired prosodic problem.

This distinction would matter in a production system where the objective is specifically to detect changes in vocal affect.

---

## 6.2 Stability and generalization are different properties

The audeering representation produced highly correlated predictions across the two E3 recordings of the same prompts:

$$
r = 0.921
$$

The corresponding correlation for the WavLM probe was:

$$
r = -0.191
$$

This suggests that the two representations have different behavior under repeated recording conditions.

However, this should not be interpreted as evidence that the audeering model is generally superior: the experiment contains only one human speaker, and its cross-domain E5 performance was lower than WavLM.

The result instead motivated further investigation into **representation stability versus cross-domain sentiment accuracy**.

---

# 7.
