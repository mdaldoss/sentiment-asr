"""Eval harness tests, using a fake deterministic Solution -- no real model
loading, so these stay fast and network-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ssa.eval.runner import EvalResult, evaluate, results_path
from ssa.manifest import add_congruence
from ssa.types import SAMPLE_RATE, AudioClip, Prediction, Sentiment


class FakeSolution:
    """Always predicts a fixed sentiment, deterministically, no model needed."""

    def __init__(self, fixed_sentiment: Sentiment = Sentiment.NEUTRAL) -> None:
        self.name = f"fake:{fixed_sentiment.value}"
        self._fixed = fixed_sentiment

    def predict(self, clip: AudioClip) -> Prediction:
        uniform = 1.0 / len(Sentiment)
        probs = {s: uniform for s in Sentiment}
        probs[self._fixed] = 0.6
        remainder = (1.0 - 0.6) / 2
        for s in Sentiment:
            if s != self._fixed:
                probs[s] = remainder
        return Prediction(
            sentiment=self._fixed,
            probs=probs,
            confidence=probs[self._fixed],
            latency_ms=1.5,
            solution=self.name,
        )

    def predict_batch(self, clips):
        return [self.predict(c) for c in clips]


class BrokenSolution:
    """Returns the wrong number of predictions -- must be rejected loudly."""

    name = "broken"

    def predict(self, clip):
        raise NotImplementedError

    def predict_batch(self, clips):
        return []  # always wrong count unless input is empty


@pytest.fixture
def small_manifest(tmp_path: Path) -> pd.DataFrame:
    """3 clips: 1 congruent, 2 incongruent (one for each PSI branch)."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for i, (text_s, prosody_s) in enumerate(
        [
            (Sentiment.NEUTRAL, Sentiment.NEUTRAL),  # congruent
            (Sentiment.NEUTRAL, Sentiment.NEGATIVE),  # incongruent
            (Sentiment.POSITIVE, Sentiment.NEGATIVE),  # incongruent
        ]
    ):
        wav_path = audio_dir / f"clip{i}.wav"
        sf.write(wav_path, np.zeros(SAMPLE_RATE // 2, dtype=np.float32), SAMPLE_RATE)
        rows.append(
            {
                "clip_id": f"c{i}",
                "path": str(wav_path.relative_to(tmp_path)),
                "speaker_id": "spk1",
                "source": "synthetic",
                "text": "test",
                "text_sentiment": text_s.value,
                "prosody_sentiment": prosody_s.value,
                "emotion_tag": "",
                "voice_id": "",
                "split": "test",
            }
        )
    df = pd.DataFrame(rows)
    return add_congruence(df)


class TestEvaluate:
    def test_produces_all_fields(self, small_manifest: pd.DataFrame, tmp_path: Path) -> None:
        solution = FakeSolution(Sentiment.NEGATIVE)
        result = evaluate(
            solution, small_manifest, dataset="test-ds", split_type="test", repo_root=tmp_path
        )

        assert result.solution == solution.name
        assert result.dataset == "test-ds"
        assert result.split_type == "test"
        assert result.n_clips == 3
        assert result.n_incongruent == 2
        assert 0.0 <= result.uar <= 1.0 or __import__("math").isnan(result.uar)
        assert len(result.predictions) == 3
        assert result.confusion_labels == ["negative", "neutral", "positive"]
        assert len(result.confusion_matrix) == 3
        assert result.git_sha  # non-empty, "unknown" is acceptable outside a repo

    def test_always_negative_solution_gets_psi_measurements(
        self, small_manifest: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Predicting NEGATIVE always: clip1's prosody=negative (a hit),
        clip2's prosody=negative too (a hit) -- both incongruent clips
        should register as following prosody."""
        solution = FakeSolution(Sentiment.NEGATIVE)
        result = evaluate(
            solution, small_manifest, dataset="d", split_type="test", repo_root=tmp_path
        )
        assert result.psi_contested == pytest.approx(1.0)
        assert result.psi_strict == pytest.approx(1.0)

    def test_rejects_mismatched_prediction_count(
        self, small_manifest: pd.DataFrame, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="returned 0 predictions"):
            evaluate(
                BrokenSolution(), small_manifest, dataset="d", split_type="test", repo_root=tmp_path
            )

    def test_rejects_empty_manifest(self, small_manifest: pd.DataFrame, tmp_path: Path) -> None:
        empty = small_manifest.iloc[0:0]
        with pytest.raises(ValueError, match="empty manifest"):
            evaluate(FakeSolution(), empty, dataset="d", split_type="test", repo_root=tmp_path)

    def test_predictions_preserve_clip_order_and_id(
        self, small_manifest: pd.DataFrame, tmp_path: Path
    ) -> None:
        solution = FakeSolution(Sentiment.POSITIVE)
        result = evaluate(
            solution, small_manifest, dataset="d", split_type="test", repo_root=tmp_path
        )
        assert [p["clip_id"] for p in result.predictions] == ["c0", "c1", "c2"]


class TestEvalResultRoundTrip:
    def test_json_round_trip(self, small_manifest: pd.DataFrame, tmp_path: Path) -> None:
        result = evaluate(
            FakeSolution(), small_manifest, dataset="d", split_type="test", repo_root=tmp_path
        )
        out = tmp_path / "result.json"
        result.to_json(out)

        reloaded = EvalResult.from_json(out)
        assert reloaded == result

    def test_json_is_plain_types_no_numpy(
        self, small_manifest: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Every value must be JSON-native -- a stray numpy scalar would
        serialise fine via our custom encoder path but silently break a
        plain `json.load` done by, say, the report generator."""
        result = evaluate(
            FakeSolution(), small_manifest, dataset="d", split_type="test", repo_root=tmp_path
        )
        out = tmp_path / "result.json"
        result.to_json(out)
        # a strict json.loads with no custom hooks must succeed
        data = json.loads(out.read_text())
        assert isinstance(data["uar"], float)
        assert isinstance(data["confusion_matrix"], list)
        assert isinstance(data["confusion_matrix"][0][0], int)


class TestResultsPath:
    def test_sanitises_solution_name(self, tmp_path: Path) -> None:
        p = results_path(tmp_path, "B:acoustic(permissive)", "cremad", "speaker_disjoint")
        assert "/" not in p.name
        assert p.name == "B_acoustic(permissive)__cremad__speaker_disjoint.json"
