"""Live-demo tests: the /classify contract and the preprocessing invariant.

There is no microphone in CI (or in the container this was built in), so
what is testable here is everything up to the mic: that browser-shaped PCM
becomes a clip through the *same* path an evaluated file takes, that the
endpoint validates its input, and that a Prediction serialises to something
the page can actually draw. The microphone path itself is the user's gate.

Models are stubbed -- these tests must not download WavLM or run inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssa.audio import AudioLoadError, clip_from_samples
from ssa.types import SAMPLE_RATE, VAD, Prediction, Sentiment

fastapi = pytest.importorskip("fastapi", reason="demo extra not installed")
from fastapi.testclient import TestClient  # noqa: E402

from demo.app import MODELS, app, prediction_to_dict  # noqa: E402


def _prediction(sentiment: Sentiment = Sentiment.POSITIVE, *, vad: VAD | None = None) -> Prediction:
    probs = {Sentiment.NEGATIVE: 0.1, Sentiment.NEUTRAL: 0.2, Sentiment.POSITIVE: 0.7}
    if sentiment is Sentiment.NEGATIVE:
        probs = {Sentiment.NEGATIVE: 0.7, Sentiment.NEUTRAL: 0.2, Sentiment.POSITIVE: 0.1}
    return Prediction(
        sentiment=sentiment,
        probs=probs,
        confidence=max(probs.values()),
        latency_ms=12.5,
        solution="B:acoustic(test)",
        vad=vad,
        transcript=None,
    )


class _StubSolution:
    def __init__(self, name: str, pred: Prediction) -> None:
        self.name = name
        self._pred = pred
        self.seen: list[float] = []

    def predict(self, clip):
        self.seen.append(clip.duration_s)
        return self._pred


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        MODELS, "permissive", _StubSolution("permissive", _prediction(Sentiment.POSITIVE))
    )
    monkeypatch.setattr(
        MODELS,
        "research",
        _StubSolution("research", _prediction(Sentiment.NEGATIVE, vad=VAD(0.3, 0.8, 0.5))),
    )
    monkeypatch.setattr(MODELS, "lexical", None)
    # raise_server_exceptions keeps a 500 as a response we can assert on
    return TestClient(app, raise_server_exceptions=False)


class TestClipFromSamples:
    """The demo must not take a different route into the model than
    evaluation does -- that is the whole reason this helper is shared."""

    def test_resamples_to_the_project_rate(self) -> None:
        clip = clip_from_samples(np.random.default_rng(0).normal(0, 0.1, 48_000), 48_000)
        assert clip.sr == SAMPLE_RATE
        assert clip.duration_s == pytest.approx(1.0, abs=0.01)

    def test_downmixes_stereo(self) -> None:
        stereo = np.zeros((16_000, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        clip = clip_from_samples(stereo, SAMPLE_RATE)
        assert clip.samples.ndim == 1

    def test_peak_normalises_like_load_clip(self) -> None:
        quiet = np.full(16_000, 0.01, dtype=np.float32)
        clip = clip_from_samples(quiet, SAMPLE_RATE)
        assert float(np.abs(clip.samples).max()) == pytest.approx(0.95, abs=1e-3)

    def test_silence_is_not_divided_by_zero(self) -> None:
        clip = clip_from_samples(np.zeros(16_000, dtype=np.float32), SAMPLE_RATE)
        assert np.all(np.isfinite(clip.samples))

    def test_empty_audio_raises(self) -> None:
        with pytest.raises(AudioLoadError, match="empty"):
            clip_from_samples(np.array([], dtype=np.float32), SAMPLE_RATE)

    def test_bad_sample_rate_raises(self) -> None:
        with pytest.raises(AudioLoadError, match="sample rate"):
            clip_from_samples(np.zeros(100, dtype=np.float32), 0)


class TestPredictionToDict:
    def test_probs_keep_canonical_order(self) -> None:
        out = prediction_to_dict(_prediction())
        assert list(out["probs"]) == ["negative", "neutral", "positive"]

    def test_vad_is_none_when_backend_has_none(self) -> None:
        assert prediction_to_dict(_prediction())["vad"] is None

    def test_vad_serialises_all_three_axes(self) -> None:
        out = prediction_to_dict(_prediction(vad=VAD(0.3, 0.8, 0.5)))
        assert out["vad"] == {"valence": 0.3, "arousal": 0.8, "dominance": 0.5}


class TestClassifyEndpoint:
    def _pcm(self, n: int = 16_000) -> list[float]:
        return list(np.sin(np.linspace(0, 400, n)).astype(float))

    def test_returns_one_entry_per_loaded_backend(self, client: TestClient) -> None:
        res = client.post("/classify", json={"pcm": self._pcm(), "sample_rate": SAMPLE_RATE})
        assert res.status_code == 200
        backends = [p["backend"] for p in res.json()["predictions"]]
        assert backends == ["permissive", "research"]

    def test_reports_duration(self, client: TestClient) -> None:
        res = client.post("/classify", json={"pcm": self._pcm(), "sample_rate": SAMPLE_RATE})
        assert res.json()["duration_s"] == pytest.approx(1.0, abs=0.01)

    def test_backends_may_disagree_and_both_are_reported(self, client: TestClient) -> None:
        """The project's central finding is that they disagree on real
        voices; the page must never collapse that to one answer."""
        preds = client.post(
            "/classify", json={"pcm": self._pcm(), "sample_rate": SAMPLE_RATE}
        ).json()["predictions"]
        assert {p["sentiment"] for p in preds} == {"positive", "negative"}

    def test_empty_pcm_rejected(self, client: TestClient) -> None:
        res = client.post("/classify", json={"pcm": [], "sample_rate": SAMPLE_RATE})
        assert res.status_code == 400

    def test_overlong_clip_rejected(self, client: TestClient) -> None:
        from demo.app import MAX_SAMPLES

        res = client.post(
            "/classify", json={"pcm": [0.0] * (MAX_SAMPLES + 1), "sample_rate": SAMPLE_RATE}
        )
        assert res.status_code == 413

    def test_invalid_sample_rate_rejected(self, client: TestClient) -> None:
        res = client.post("/classify", json={"pcm": self._pcm(), "sample_rate": 0})
        assert res.status_code == 422

    def test_browser_rate_is_resampled_not_rejected(self, client: TestClient) -> None:
        """Browsers commonly capture at 44.1/48 kHz; the server resamples."""
        res = client.post("/classify", json={"pcm": self._pcm(44_100), "sample_rate": 44_100})
        assert res.status_code == 200
        assert res.json()["duration_s"] == pytest.approx(1.0, abs=0.01)

    def test_503_when_no_models_loaded(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(MODELS, "permissive", None)
        monkeypatch.setattr(MODELS, "research", None)
        res = client.post("/classify", json={"pcm": self._pcm(), "sample_rate": SAMPLE_RATE})
        assert res.status_code == 503


class TestExampleEndpoint:
    """The page must open showing what it does, not an empty shell waiting
    for a microphone that may not exist."""

    def test_returns_an_example_marked_as_such(self, client: TestClient) -> None:
        body = client.get("/example").json()
        assert body["is_example"] is True

    def test_carries_the_labels_that_make_it_readable(self, client: TestClient) -> None:
        body = client.get("/example").json()
        assert body["text_sentiment"] == "positive"
        assert body["prosody_sentiment"] == "neutral"
        assert body["text"]

    def test_scores_every_loaded_backend(self, client: TestClient) -> None:
        body = client.get("/example").json()
        assert [p["backend"] for p in body["predictions"]] == ["permissive", "research"]

    def test_404_when_the_clip_is_absent(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import demo.app as demo_app

        monkeypatch.setattr(demo_app, "EXAMPLE_CLIP", Path("/nonexistent/clip.wav"))
        assert client.get("/example").status_code == 404


class TestWarmUp:
    def test_runs_one_inference_per_loaded_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without this the first real request was measured at ~17s."""
        models = type(MODELS)()
        perm = _StubSolution("permissive", _prediction())
        res = _StubSolution("research", _prediction())
        models.permissive, models.research, models.lexical = perm, res, None
        models._warm_up()
        assert len(perm.seen) == 1
        assert len(res.seen) == 1

    def test_a_failing_model_does_not_abort_startup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Broken:
            name = "broken"

            def predict(self, clip):
                raise RuntimeError("boom")

        models = type(MODELS)()
        good = _StubSolution("permissive", _prediction())
        models.permissive, models.research, models.lexical = _Broken(), good, None
        models._warm_up()  # must not raise
        assert len(good.seen) == 1


class TestPageAndHealth:
    def test_health_reports_each_model(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body == {"permissive": True, "research": True, "lexical": False}

    def test_index_serves_the_page(self, client: TestClient) -> None:
        html = client.get("/").text
        assert "<title>" in html
        assert "/classify" in html  # the page actually wires itself to the endpoint

    def test_page_uses_raw_pcm_not_mediarecorder(self, client: TestClient) -> None:
        """MediaRecorder would emit webm/opus, which soundfile cannot read."""
        html = client.get("/").text
        assert "MediaRecorder" not in html.replace("MediaRecorder would", "")
        assert "createScriptProcessor" in html
