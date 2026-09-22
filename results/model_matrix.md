# Model matrix: three backends x three data regimes

Generated from `results/model_matrix.json` by `scripts/render_model_matrix.py`.
Gold label is `prosody_sentiment` everywhere. UAR is the headline; accuracy
appears only beside it. Chance UAR is 0.333 for three classes; PSI chance is
0.5 (contested) and 0.333 (strict).

## X1_cremad_hume__zurich

Train CREMA-D + Hume, test on the Zurich human recordings.

- train: 5280 clips / 65 speakers ({'crema_d': 5235, 'hume': 45}), classes {'negative': 3591, 'positive': 909, 'neutral': 780}
- val: 782 clips / 10 speakers (threshold fitting and candidate selection only)

| backend | eval set | n | UAR [95% CI] | macro-F1 | acc | PSI contested | PSI strict | flags |
|---|---|---|---|---|---|---|---|---|
| research (audeering, 2 thresholds) | zurich_all | 160 | 0.356 [0.29, 0.43] | 0.330 | 0.350 | 0.366 [0.27, 0.47] | 0.270 | - |
| research (audeering, 2 thresholds) | zurich_test_speaker | 20 | 0.289 [0.11, 0.48] | 0.200 | 0.250 | 0.364 [0.09, 0.67] | 0.286 | - |
| research (audeering VAD -> logreg) | zurich_all | 160 | 0.398 [0.35, 0.45] | 0.320 | 0.412 | 0.488 [0.39, 0.59] | 0.369 | - |
| research (audeering VAD -> logreg) | zurich_test_speaker | 20 | 0.400 [0.33, 0.56] | 0.326 | 0.500 | 0.500 [0.20, 0.79] | 0.429 | - |
| permissive (WavLM + probe) | zurich_all | 160 | 0.433 [0.36, 0.51] | 0.402 | 0.425 | 0.646 [0.54, 0.75] | 0.459 | - |
| permissive (WavLM + probe) | zurich_test_speaker | 20 | 0.222 [0.08, 0.33] | 0.111 | 0.200 | 0.333 [0.00, 0.67] | 0.214 | - |
| prosody (eGeMAPS+contour, logreg) | zurich_all | 160 | 0.372 [0.31, 0.44] | 0.303 | 0.362 | 0.547 [0.44, 0.67] | 0.369 | - |
| prosody (eGeMAPS+contour, logreg) | zurich_test_speaker | 20 | 0.300 [0.10, 0.52] | 0.221 | 0.250 | 0.400 [0.11, 0.73] | 0.286 | - |
| _majority baseline (neutral)_ | zurich_all | 160 | 0.333 | 0.173 | 0.350 | 0.381 | 0.288 | reference |
| _majority baseline (neutral)_ | zurich_test_speaker | 20 | 0.333 | 0.207 | 0.450 | 0.500 | 0.357 | reference |

Notes:

- Hume's Ava Song trains; Colton Rivers is held out into the validation set so threshold fitting and candidate selection see a second corpus rather than CREMA-D alone. Neither voice is in any test set here.
- All 160 Zurich clips are scored: 8 speakers, none of whom appear in training. zurich_test_speaker is Zurich's own 20-clip held-out speaker, reported separately for comparability with X3 and results/e6_summary.json.

Prosody candidates (val UAR; headline is `logreg`, val-selected would be `hist_gbdt`): logreg 0.606, linear_svm 0.423, svm_rbf 0.658, hist_gbdt 0.662

## X2_cremad__hume

Train CREMA-D only, test on the Hume incongruence set.

- train: 5235 clips / 64 speakers ({'crema_d': 5235}), classes {'negative': 3576, 'positive': 894, 'neutral': 765}
- val: 737 clips / 9 speakers (threshold fitting and candidate selection only)

| backend | eval set | n | UAR [95% CI] | macro-F1 | acc | PSI contested | PSI strict | flags |
|---|---|---|---|---|---|---|---|---|
| research (audeering, 2 thresholds) | hume_all | 90 | 0.400 [0.31, 0.49] | 0.367 | 0.400 | 0.211 [0.11, 0.32] | 0.200 | - |
| research (audeering, 2 thresholds) | hume_ava | 45 | 0.378 [0.25, 0.51] | 0.345 | 0.378 | 0.185 [0.05, 0.33] | 0.167 | - |
| research (audeering, 2 thresholds) | hume_colton | 45 | 0.422 [0.29, 0.55] | 0.388 | 0.422 | 0.233 [0.09, 0.39] | 0.233 | - |
| research (audeering VAD -> logreg) | hume_all | 90 | 0.478 [0.42, 0.54] | 0.394 | 0.478 | 0.614 [0.47, 0.76] | 0.450 | - |
| research (audeering VAD -> logreg) | hume_ava | 45 | 0.556 [0.48, 0.63] | 0.479 | 0.556 | 0.667 [0.48, 0.85] | 0.533 | - |
| research (audeering VAD -> logreg) | hume_colton | 45 | 0.400 [0.33, 0.48] | 0.287 | 0.400 | 0.550 [0.32, 0.75] | 0.367 | - |
| permissive (WavLM + probe) | hume_all | 90 | 0.511 [0.42, 0.60] | 0.493 | 0.511 | 0.682 [0.54, 0.82] | 0.500 | - |
| permissive (WavLM + probe) | hume_ava | 45 | 0.467 [0.35, 0.60] | 0.435 | 0.467 | 0.636 [0.43, 0.83] | 0.467 | - |
| permissive (WavLM + probe) | hume_colton | 45 | 0.556 [0.41, 0.70] | 0.547 | 0.556 | 0.727 [0.53, 0.90] | 0.533 | - |
| prosody (eGeMAPS+contour, logreg) | hume_all | 90 | 0.278 [0.20, 0.36] | 0.214 | 0.278 | 0.378 [0.22, 0.55] | 0.233 | - |
| prosody (eGeMAPS+contour, logreg) | hume_ava | 45 | 0.289 [0.17, 0.41] | 0.241 | 0.289 | 0.333 [0.12, 0.57] | 0.200 | - |
| prosody (eGeMAPS+contour, logreg) | hume_colton | 45 | 0.267 [0.19, 0.33] | 0.140 | 0.267 | 0.421 [0.21, 0.65] | 0.267 | - |
| _majority baseline (positive)_ | hume_all | 90 | 0.333 | 0.167 | 0.333 | 0.500 | 0.333 | reference |
| _majority baseline (positive)_ | hume_ava | 45 | 0.333 | 0.167 | 0.333 | 0.500 | 0.333 | reference |
| _majority baseline (positive)_ | hume_colton | 45 | 0.333 | 0.167 | 0.333 | 0.500 | 0.333 | reference |

Notes:

- No E5 anywhere in the fit, so this is the only configuration whose E5 PSI reading is a clean instrument (DESIGN.md designates E5 eval-only).
- Both voices are scored together and separately: 45 clips each, and a two-voice synthetic corpus is not a population.

Prosody candidates (val UAR; headline is `logreg`, val-selected would be `svm_rbf`): logreg 0.646, linear_svm 0.457, svm_rbf 0.703, hist_gbdt 0.653

## X3_cremad_hume_zurich__disjoint

Train CREMA-D + Hume + Zurich, test on a speaker-disjoint split of the same union.

- train: 5380 clips / 70 speakers ({'crema_d': 5235, 'hume': 45, 'recorded': 100}), classes {'negative': 3623, 'positive': 943, 'neutral': 814}
- val: 777 clips / 11 speakers (threshold fitting and candidate selection only)

| backend | eval set | n | UAR [95% CI] | macro-F1 | acc | PSI contested | PSI strict | flags |
|---|---|---|---|---|---|---|---|---|
| research (audeering, 2 thresholds) | pooled_all | 1535 | 0.434 [0.41, 0.46] | 0.439 | 0.590 | 0.750 [0.72, 0.78] | 0.646 | - |
| research (audeering, 2 thresholds) | pooled_cremad_subset300 | 365 | 0.442 [0.39, 0.50] | 0.448 | 0.570 | 0.672 [0.62, 0.73] | 0.607 | - |
| research (audeering, 2 thresholds) | cremad_test | 1470 | 0.435 [0.41, 0.46] | 0.440 | 0.599 | 0.768 [0.74, 0.79] | 0.660 | - |
| research (audeering, 2 thresholds) | hume_colton | 45 | 0.422 [0.29, 0.55] | 0.388 | 0.422 | 0.233 [0.09, 0.39] | 0.233 | - |
| research (audeering, 2 thresholds) | zurich_test_speaker | 20 | 0.289 [0.11, 0.48] | 0.200 | 0.250 | 0.364 [0.09, 0.67] | 0.286 | - |
| research (audeering VAD -> logreg) | pooled_all | 1535 | 0.601 [0.57, 0.63] | 0.488 | 0.504 | 0.593 [0.56, 0.62] | 0.453 | - |
| research (audeering VAD -> logreg) | pooled_cremad_subset300 | 365 | 0.591 [0.54, 0.64] | 0.482 | 0.488 | 0.571 [0.51, 0.63] | 0.430 | - |
| research (audeering VAD -> logreg) | cremad_test | 1470 | 0.605 [0.58, 0.63] | 0.487 | 0.506 | 0.594 [0.56, 0.62] | 0.454 | - |
| research (audeering VAD -> logreg) | hume_colton | 45 | 0.422 [0.36, 0.50] | 0.319 | 0.422 | 0.600 [0.37, 0.81] | 0.400 | - |
| research (audeering VAD -> logreg) | zurich_test_speaker | 20 | 0.400 [0.33, 0.56] | 0.326 | 0.500 | 0.500 [0.20, 0.79] | 0.429 | - |
| permissive (WavLM + probe) | pooled_all | 1535 | 0.729 [0.70, 0.76] | 0.727 | 0.799 | 0.939 [0.93, 0.95] | 0.821 | - |
| permissive (WavLM + probe) | pooled_cremad_subset300 | 365 | 0.722 [0.66, 0.78] | 0.715 | 0.773 | 0.918 [0.88, 0.95] | 0.787 | - |
| permissive (WavLM + probe) | cremad_test | 1470 | 0.742 [0.71, 0.77] | 0.739 | 0.812 | 0.944 [0.93, 0.96] | 0.833 | - |
| permissive (WavLM + probe) | hume_colton | 45 | 0.600 [0.44, 0.74] | 0.609 | 0.600 | 0.842 [0.65, 1.00] | 0.533 | - |
| permissive (WavLM + probe) | zurich_test_speaker | 20 | 0.289 [0.10, 0.53] | 0.299 | 0.300 | 0.556 [0.22, 0.89] | 0.357 | - |
| prosody (eGeMAPS+contour, logreg) | pooled_all | 1535 | 0.622 [0.59, 0.65] | 0.553 | 0.595 | 0.738 [0.71, 0.76] | 0.574 | - |
| prosody (eGeMAPS+contour, logreg) | pooled_cremad_subset300 | 365 | 0.542 [0.48, 0.60] | 0.515 | 0.564 | 0.746 [0.69, 0.80] | 0.567 | - |
| prosody (eGeMAPS+contour, logreg) | cremad_test | 1470 | 0.643 [0.61, 0.67] | 0.561 | 0.605 | 0.743 [0.72, 0.77] | 0.582 | - |
| prosody (eGeMAPS+contour, logreg) | hume_colton | 45 | 0.422 [0.28, 0.58] | 0.445 | 0.422 | 0.579 [0.35, 0.81] | 0.367 | - |
| prosody (eGeMAPS+contour, logreg) | zurich_test_speaker | 20 | 0.300 [0.10, 0.52] | 0.245 | 0.250 | 0.444 [0.14, 0.80] | 0.286 | - |
| _majority baseline (negative)_ | pooled_all | 1535 | 0.333 | 0.267 | 0.668 | 0.985 | 0.784 | reference |
| _majority baseline (negative)_ | pooled_cremad_subset300 | 365 | 0.333 | 0.255 | 0.619 | 0.936 | 0.733 | reference |
| _majority baseline (negative)_ | cremad_test | 1470 | 0.333 | 0.271 | 0.683 | 1.000 | 0.800 | reference |
| _majority baseline (positive)_ | hume_colton | 45 | 0.333 | 0.167 | 0.333 | 0.500 | 0.333 | reference |
| _majority baseline (neutral)_ | zurich_test_speaker | 20 | 0.333 | 0.207 | 0.450 | 0.500 | 0.357 | reference |

Notes:

- The union's split is each corpus's own speaker-disjoint split, concatenated: CREMA-D 64/9/18 speakers, Hume Ava/-/Colton, Zurich 5/2/1 speakers. assert_speaker_disjoint is run on the assembled manifest.
- pooled_all is 1,535 clips of which 1,470 are CREMA-D, so it is a CREMA-D number wearing a union's name. pooled_cremad_subset300 subsamples CREMA-D test to 300 (class-stratified, seed 0) so the three corpora are less lopsided. Read the per-corpus rows first.
- This is the within-corpus regime for all three corpora at once. Its numbers are not generalisation numbers -- X1 and X2 are.

Prosody candidates (val UAR; headline is `logreg`, val-selected would be `svm_rbf`): logreg 0.606, linear_svm 0.385, svm_rbf 0.681, hist_gbdt 0.661

## Method

Three acoustic backends fitted under three data regimes. The classifier on top of the permissive and research-head representations is held identical (StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'), seed 0) so that what varies between those cells is the representation and the training data, not the estimator. The prosody backend fits all four candidates from scripts/train_prosodic.py; 'logreg' is the headline and the others are reported in eval_all_candidates. 95% intervals are a 2,000-sample percentile bootstrap over clips (scripts/eval_zurich.bootstrap_ci).

## Caveats

- TRAINING ON E5 (Hume) BREAKS THIS PROJECT'S OWN DESIGN RULE. DESIGN.md designates E5 eval-only because it is the instrument that measures prosody sensitivity under contradiction. X1 and X3 train on Ava Song's 45 clips, so for those configurations E5 is no longer a clean PSI instrument. X2, which trains on no E5 at all, is the honest PSI reading.
- Sample sizes are wildly unequal by design: 45 Hume clips per voice and 20 Zurich test clips against CREMA-D's 1,470-clip test split. Differences of a few points between backends are not resolvable at those sizes, which is why every cell carries a bootstrap interval. Read the intervals, not the ranks.
- The Zurich test speaker is ONE person and the Hume held-out voice is ONE synthetic voice. A controlled anecdote, not a population claim.
- E3 is not scored anywhere here: Zurich's speaker1 is the same human who recorded E3, and X3 trains on him.
- The research backend is CC-BY-NC-SA-4.0 (research use only) and cannot ship in a commercial product without a separate license from audEERING.
- Only three configurations were requested, so there is no 'train on Zurich alone' or 'train on Hume alone' row here; results/combos_e5_e6.json already covers those for the permissive backend.
- Bootstrap resampling is over clips, which treats them as exchangeable. The same speakers and carrier sentences recur, so the true intervals are if anything wider than these.
- PSI ON CREMA-D IS NOT A PROSODY MEASUREMENT, and the majority-baseline rows are here to show it: CREMA-D's text is neutral on every clip, so a model that only ever says 'negative' never matches the text label and scores PSI contested 1.000 on cremad_test. Read PSI on the Hume and Zurich rows, where the text label actually varies; on CREMA-D read UAR.
