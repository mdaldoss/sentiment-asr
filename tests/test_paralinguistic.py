"""Whisper-detection tests. Not mocked -- opensmile is fast enough to run
for real, and the point is to confirm the detector actually discriminates,
which is exactly what a mock would hide. A real synthetic-signal iteration
during development surfaced that white noise scores as perceptually LOUDER
than a clean tone under eGeMAPS's loudness measure (it isn't raw
amplitude) -- these tests lock in the fix (HNR as the primary signal)."""

from __future__ import annotations

import numpy as np
import pytest

from ssa.paralinguistic import (
    DEFAULT_THRESHOLDS,
    WhisperThresholds,
    extract_features,
    fit_whisper_thresholds,
    is_whispered,
)
from ssa.types import SAMPLE_RATE, AudioClip


def _clean_voiced_clip() -> AudioClip:
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    sig = np.zeros_like(t)
    for h in range(1, 6):
        sig += (1.0 / h) * np.sin(2 * np.pi * 120 * h * t)
    sig = sig / np.abs(sig).max() * 0.5
    return AudioClip(clip_id="voiced", samples=sig.astype(np.float32))


def _whisper_like_clip(seed: int = 0) -> AudioClip:
    """Low amplitude, purely unvoiced (no periodic component) -- the
    defining acoustic contrast with clean voiced speech."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(SAMPLE_RATE) * 0.03
    return AudioClip(clip_id="whisper", samples=sig.astype(np.float32))


class TestExtractFeatures:
    def test_returns_all_four_fields(self) -> None:
        feats = extract_features(_clean_voiced_clip())
        assert feats.loudness_mean is not None
        assert feats.hnr_mean is not None
        assert feats.jitter_mean is not None
        assert feats.shimmer_db_mean is not None

    def test_voiced_signal_has_much_higher_hnr_than_noise(self) -> None:
        """This is the validated discriminator -- confirmed directly
        before trusting it as is_whispered's primary signal."""
        voiced = extract_features(_clean_voiced_clip())
        whisper = extract_features(_whisper_like_clip())
        assert voiced.hnr_mean - whisper.hnr_mean > 5.0


class TestIsWhispered:
    def test_clean_voiced_speech_not_flagged(self) -> None:
        feats = extract_features(_clean_voiced_clip())
        assert is_whispered(feats, DEFAULT_THRESHOLDS) is False

    def test_unvoiced_low_amplitude_signal_flagged(self) -> None:
        feats = extract_features(_whisper_like_clip())
        assert is_whispered(feats, DEFAULT_THRESHOLDS) is True

    def test_high_hnr_never_flagged_regardless_of_loudness_threshold(self) -> None:
        """hnr_max is the necessary condition (module design) -- a very
        permissive loudness threshold alone must not trigger a flag."""
        feats = extract_features(_clean_voiced_clip())
        lenient = WhisperThresholds(hnr_max=100.0, loudness_max=100.0)
        strict_loudness_only = WhisperThresholds(hnr_max=-100.0, loudness_max=100.0)
        assert is_whispered(feats, lenient) is True  # both thresholds trivially satisfied
        assert is_whispered(feats, strict_loudness_only) is False  # HNR gate blocks it


class TestFitWhisperThresholds:
    def test_returns_midpoint_of_group_means(self) -> None:
        normal = [extract_features(_clean_voiced_clip())]
        whisper = [extract_features(_whisper_like_clip())]
        fitted = fit_whisper_thresholds(normal, whisper)
        assert fitted.hnr_max == pytest.approx((normal[0].hnr_mean + whisper[0].hnr_mean) / 2)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one example"):
            fit_whisper_thresholds([], [extract_features(_whisper_like_clip())])
