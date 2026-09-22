"""Solution D: explicit prosodic features + a transparent classifier.

A, B and C all route audio through a large pretrained network. D does not.
It measures named physical quantities of the waveform -- pitch contour,
loudness dynamics, jitter, shimmer, harmonicity, pause structure -- and
hands that vector to an ordinary classifier. Nothing in the path can
represent a word, so the failure mode E5 exposed in the audeering model
(following the transcript on 79% of contradictory clips, from audio alone)
is structurally impossible here rather than merely hoped against.

It is in the repo because it was measured, not assumed: on E5's audio a
logistic regression over nine such descriptors recovered the intended
delivery at 0.811 vs 0.333 chance, while the two deep acoustic backends
managed UAR 0.511 and 0.400 (`results/e5_diagnostic.json`). Solution D is
that finding promoted from a diagnostic to a competitor, scored by the same
harness on the same sets so the comparison is honest.

The trained artifact is a single sklearn `Pipeline`: median imputation
(extraction can fail on a degenerate clip and emits NaN rather than
dropping it), standardisation, then the classifier chosen on the validation
split -- see `scripts/train_prosodic.py`, which fits several and records
every candidate's score rather than only the winner's.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np

from ssa.prosodic_features import ProsodicExtractor, describe
from ssa.types import AudioClip, Prediction, Sentiment

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "data" / "cache" / "prosodic_model.joblib"


class ProsodicSolution:
    """Frozen feature extractor + fitted classifier. No neural network."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        *,
        model_type: str | None = None,
        extractor: ProsodicExtractor | None = None,
    ) -> None:
        """`model_type` picks one of the fitted candidates; the default is
        whichever the training run selected. Being able to name one matters
        because the in-domain winner is not necessarily the one that still
        works on a new speaker -- see scripts/eval_prosodic.py."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"no prosodic model at {model_path} -- run `make train-prosodic` first"
            )
        bundle = joblib.load(model_path)
        if model_type is None:
            self._pipeline = bundle["pipeline"]
            self._model_type = bundle["model_type"]
        else:
            available = bundle.get("pipelines", {})
            if model_type not in available:
                raise KeyError(f"no fitted {model_type!r}; have {sorted(available)}")
            self._pipeline = available[model_type]
            self._model_type = model_type
        self._extractor = extractor if extractor is not None else ProsodicExtractor()
        self.name = f"D:prosodic(egemaps+contour/{self._model_type})"

    def predict(self, clip: AudioClip) -> Prediction:
        t0 = time.perf_counter()
        features = self._extractor.embed(clip)
        proba = self._pipeline.predict_proba(features.reshape(1, -1))[0]

        # Zip against the pipeline's own classes_ rather than assuming
        # sklearn's alphabetical order matches Sentiment.ordered() -- it
        # happens to, and relying on that is how labels get silently swapped.
        classes = self._pipeline.classes_
        probs = {Sentiment(c): float(p) for c, p in zip(classes, proba, strict=True)}
        sentiment = max(probs, key=probs.get)  # type: ignore[arg-type]

        return Prediction(
            sentiment=sentiment,
            probs=probs,
            confidence=probs[sentiment],
            latency_ms=(time.perf_counter() - t0) * 1000,
            solution=self.name,
        )

    def predict_batch(self, clips: Sequence[AudioClip]) -> list[Prediction]:
        return [self.predict(clip) for clip in clips]

    def explain(self, clip: AudioClip, top_k: int = 8) -> list[tuple[str, float]]:
        """The named measurements behind a prediction. A is explainable by
        its transcript and B is not explainable at all; D can at least say
        which physical quantities were extreme for this clip."""
        return describe(self._extractor.embed(clip), top_k=top_k)


def feature_importances(model_path: Path = DEFAULT_MODEL_PATH) -> list[tuple[str, float]] | None:
    """Which features the fitted model actually leans on, most first.

    Returns None for a model that exposes no importances. Reads the names
    stored with the artifact rather than recomputing them, so it cannot
    drift out of step with what the model was fitted on.
    """
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    names = bundle["feature_names"]
    clf = bundle["pipeline"][-1]

    if hasattr(clf, "feature_importances_"):
        weights = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        # Multiclass linear model: aggregate magnitude across classes.
        weights = np.abs(np.asarray(clf.coef_, dtype=float)).mean(axis=0)
    else:
        return None

    order = np.argsort(-weights)
    return [(names[i], float(weights[i])) for i in order]
