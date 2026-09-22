"""AcousticSolution tests, using a fake encoder and a toy probe fit on
synthetic embeddings -- no real WavLM model loading, so these stay fast and
network-free. The real WavLM encoder is exercised by ssa/train.py's
end-to-end run (manual/CI, not unit tests) and by test_embeddings.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ssa.solutions.acoustic import AcousticSolution
from ssa.types import SAMPLE_RATE, VAD, AudioClip, Sentiment


class FakeEncoder:
    """Returns a fixed, caller-controlled embedding regardless of audio
    content -- lets tests drive the probe's prediction deterministically."""

    def __init__(self, embedding: np.ndarray) -> None:
        self._embedding = embedding

    def embed(self, clip: AudioClip) -> np.ndarray:
        return self._embedding


def _make_toy_probe(tmp_path: Path, *, embedding_dim: int = 8) -> Path:
    """A tiny but real, fitted LogisticRegression + scaler, saved in the same
    bundle format ssa/train.py produces. 3 well-separated synthetic clusters
    so predictions are deterministic and easy to reason about."""
    rng = np.random.default_rng(0)
    n_per_class = 20
    X, y = [], []
    # Push each class's cluster far along a different axis so the probe's
    # decision is unambiguous for the fixed test embeddings below.
    centers = {
        "negative": np.array([10.0] + [0.0] * (embedding_dim - 1)),
        "neutral": np.array([0.0, 10.0] + [0.0] * (embedding_dim - 2)),
        "positive": np.array([0.0, 0.0, 10.0] + [0.0] * (embedding_dim - 3)),
    }
    for label, center in centers.items():
        pts = center + rng.normal(scale=0.5, size=(n_per_class, embedding_dim))
        X.append(pts)
        y.extend([label] * n_per_class)
    X = np.concatenate(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)

    probe_path = tmp_path / "probe.joblib"
    joblib.dump({"scaler": scaler, "clf": clf, "model_type": "logreg"}, probe_path)
    return probe_path


@pytest.fixture
def toy_probe_path(tmp_path: Path) -> Path:
    return _make_toy_probe(tmp_path)


def _fake_clip() -> AudioClip:
    return AudioClip(clip_id="fake", samples=np.zeros(SAMPLE_RATE, dtype=np.float32))


class TestAcousticSolution:
    def test_missing_probe_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"run `make train`"):
            AcousticSolution(probe_path=tmp_path / "does_not_exist.joblib")

    def test_unknown_backend_raises(self, toy_probe_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            AcousticSolution(backend="not-a-real-backend", probe_path=toy_probe_path)

    def test_research_backend_missing_thresholds_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"run `make train"):
            AcousticSolution(backend="research", thresholds_path=tmp_path / "missing.json")

    def test_predicts_sentiment_matching_the_embedding_cluster(self, toy_probe_path: Path) -> None:
        embedding_dim = 8
        # exactly at the "negative" cluster center used to fit the toy probe
        negative_embedding = np.array([10.0] + [0.0] * (embedding_dim - 1), dtype=np.float32)
        solution = AcousticSolution(
            probe_path=toy_probe_path, encoder=FakeEncoder(negative_embedding)
        )
        pred = solution.predict(_fake_clip())
        assert pred.sentiment == Sentiment.NEGATIVE
        assert pred.abstained is False

    def test_probs_are_valid_distribution(self, toy_probe_path: Path) -> None:
        embedding = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        solution = AcousticSolution(probe_path=toy_probe_path, encoder=FakeEncoder(embedding))
        pred = solution.predict(_fake_clip())
        assert set(pred.probs) == set(Sentiment)
        assert sum(pred.probs.values()) == pytest.approx(1.0)
        assert pred.confidence == pred.probs[pred.sentiment]

    def test_solution_name_reflects_backend_and_model_type(self, toy_probe_path: Path) -> None:
        embedding = np.zeros(8, dtype=np.float32)
        solution = AcousticSolution(probe_path=toy_probe_path, encoder=FakeEncoder(embedding))
        assert solution.name == "B:acoustic(permissive/logreg)"

    def test_predict_batch_matches_repeated_predict(self, toy_probe_path: Path) -> None:
        embedding = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        solution = AcousticSolution(probe_path=toy_probe_path, encoder=FakeEncoder(embedding))
        clips = [_fake_clip() for _ in range(3)]
        batch = solution.predict_batch(clips)
        singles = [solution.predict(c) for c in clips]
        assert [p.sentiment for p in batch] == [p.sentiment for p in singles]


class FakeVADEncoder:
    """Returns a fixed, caller-controlled VAD regardless of audio content."""

    def __init__(self, vad: VAD) -> None:
        self._vad = vad

    def predict_vad(self, clip: AudioClip) -> VAD:
        return self._vad


def _make_thresholds_file(tmp_path: Path, *, low: float = 0.3, high: float = 0.7) -> Path:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"low": low, "high": high}))
    return path


class TestResearchBackend:
    def test_predicts_sentiment_from_valence_threshold(self, tmp_path: Path) -> None:
        thresholds_path = _make_thresholds_file(tmp_path, low=0.3, high=0.7)
        vad = VAD(valence=0.9, arousal=0.5, dominance=0.5)
        solution = AcousticSolution(
            backend="research", thresholds_path=thresholds_path, encoder=FakeVADEncoder(vad)
        )
        pred = solution.predict(_fake_clip())
        assert pred.sentiment == Sentiment.POSITIVE
        assert pred.vad == vad

    def test_probs_are_valid_distribution(self, tmp_path: Path) -> None:
        thresholds_path = _make_thresholds_file(tmp_path)
        vad = VAD(valence=0.1, arousal=0.5, dominance=0.5)
        solution = AcousticSolution(
            backend="research", thresholds_path=thresholds_path, encoder=FakeVADEncoder(vad)
        )
        pred = solution.predict(_fake_clip())
        assert set(pred.probs) == set(Sentiment)
        assert sum(pred.probs.values()) == pytest.approx(1.0)
        assert pred.confidence >= 1 / 3  # never below chance for a 3-way problem

    def test_confidence_increases_with_distance_from_threshold(self, tmp_path: Path) -> None:
        thresholds_path = _make_thresholds_file(tmp_path, low=0.3, high=0.7)
        solution_near = AcousticSolution(
            backend="research",
            thresholds_path=thresholds_path,
            encoder=FakeVADEncoder(VAD(valence=0.71, arousal=0.5, dominance=0.5)),
        )
        solution_far = AcousticSolution(
            backend="research",
            thresholds_path=thresholds_path,
            encoder=FakeVADEncoder(VAD(valence=1.0, arousal=0.5, dominance=0.5)),
        )
        near_conf = solution_near.predict(_fake_clip()).confidence
        far_conf = solution_far.predict(_fake_clip()).confidence
        assert far_conf > near_conf

    def test_solution_name(self, tmp_path: Path) -> None:
        thresholds_path = _make_thresholds_file(tmp_path)
        solution = AcousticSolution(
            backend="research",
            thresholds_path=thresholds_path,
            encoder=FakeVADEncoder(VAD(valence=0.5, arousal=0.5, dominance=0.5)),
        )
        assert solution.name == "B:acoustic(research/audeering-vad)"
