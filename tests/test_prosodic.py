"""Solution D tests: feature extraction on synthetic signals, and the
classifier wrapper against a stub pipeline.

Deliberately no model download and no CREMA-D: the features are computed
from signals constructed here, so the tests state what the descriptors
should do rather than what they happen to produce on real audio. The one
property that matters most is the last class -- nothing in this path may
be able to represent a word.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import ssa.prosodic_features
from ssa.prosodic_features import (
    CONTOUR_NAMES,
    EGEMAPS_NAMES,
    FEATURE_NAMES,
    N_FEATURES,
    ProsodicExtractor,
    describe,
)
from ssa.types import SAMPLE_RATE, AudioClip


def _tone(f0: float = 150.0, seconds: float = 1.5, amplitude: float = 0.4) -> AudioClip:
    """A harmonic-rich buzz at a fixed pitch -- something Praat can track."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    wave = sum(amplitude / k * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3, 4))
    return AudioClip(clip_id="tone", samples=wave.astype(np.float32), sr=SAMPLE_RATE)


def _glide(f_start: float, f_end: float, seconds: float = 1.5) -> AudioClip:
    """A pitch sweep, so the sign of the global F0 slope is known a priori."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    f0 = np.linspace(f_start, f_end, t.size)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    wave = sum(0.4 / k * np.sin(k * phase) for k in (1, 2, 3, 4))
    return AudioClip(clip_id="glide", samples=wave.astype(np.float32), sr=SAMPLE_RATE)


@pytest.fixture(scope="module")
def extractor() -> ProsodicExtractor:
    return ProsodicExtractor()


class TestFeatureVector:
    def test_egemaps_plus_contour_is_the_full_set(self) -> None:
        assert len(FEATURE_NAMES) == len(EGEMAPS_NAMES) + len(CONTOUR_NAMES) == N_FEATURES

    def test_every_feature_has_a_distinct_name(self) -> None:
        """Names are the point of this solution -- duplicates would make an
        explanation ambiguous."""
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_vector_has_the_declared_length(self, extractor: ProsodicExtractor) -> None:
        assert extractor.embed(_tone()).shape == (N_FEATURES,)

    def test_uses_the_standard_egemaps_set(self) -> None:
        """88 functionals is eGeMAPSv02; a different count means the feature
        set silently changed underneath us."""
        assert len(EGEMAPS_NAMES) == 88

    def test_extraction_is_deterministic(self, extractor: ProsodicExtractor) -> None:
        a, b = extractor.embed(_tone()), extractor.embed(_tone())
        np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=1e-5)


class TestContourFeatures:
    """The features added beyond eGeMAPS -- they exist to capture global
    trend, so a known trend must come back with the right sign."""

    def _contour(self, extractor: ProsodicExtractor, clip: AudioClip) -> dict[str, float]:
        vec = extractor.embed(clip)
        return dict(zip(CONTOUR_NAMES, vec[-len(CONTOUR_NAMES) :], strict=True))

    def test_falling_pitch_gives_negative_slope(self, extractor: ProsodicExtractor) -> None:
        c = self._contour(extractor, _glide(260.0, 120.0))
        assert c["f0_slope_st_per_s"] < 0

    def test_rising_pitch_gives_positive_slope(self, extractor: ProsodicExtractor) -> None:
        c = self._contour(extractor, _glide(120.0, 260.0))
        assert c["f0_slope_st_per_s"] > 0

    def test_monotone_has_smaller_range_than_a_glide(self, extractor: ProsodicExtractor) -> None:
        flat = self._contour(extractor, _tone(150.0))
        glide = self._contour(extractor, _glide(120.0, 260.0))
        assert flat["f0_dynamic_range_st"] < glide["f0_dynamic_range_st"]

    def test_continuous_tone_is_mostly_voiced_without_pauses(
        self, extractor: ProsodicExtractor
    ) -> None:
        c = self._contour(extractor, _tone())
        assert c["voiced_fraction"] > 0.5
        assert c["pause_ratio"] < 0.5

    def test_silence_does_not_raise(self, extractor: ProsodicExtractor) -> None:
        """Pitch tracking has nothing to find here; NaN is acceptable, a
        crash is not -- the clip must still be counted."""
        silence = AudioClip(
            clip_id="silence", samples=np.zeros(SAMPLE_RATE, dtype=np.float32), sr=SAMPLE_RATE
        )
        assert extractor.embed(silence).shape == (N_FEATURES,)


class TestDescribe:
    def test_returns_named_features(self, extractor: ProsodicExtractor) -> None:
        top = describe(extractor.embed(_tone()), top_k=5)
        assert len(top) == 5
        assert all(name in FEATURE_NAMES for name, _ in top)

    def test_sorted_by_magnitude(self, extractor: ProsodicExtractor) -> None:
        top = describe(extractor.embed(_tone()), top_k=6)
        mags = [abs(v) for _, v in top]
        assert mags == sorted(mags, reverse=True)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            describe(np.zeros(5))


class TestNoLexicalPath:
    """Solution D's core claim is structural: it cannot read words. If a
    transcript ever became reachable from here, that claim would be false
    and E5's PSI result would stop meaning what it means."""

    def test_extractor_takes_only_audio(self) -> None:
        params = inspect.signature(ProsodicExtractor.embed).parameters
        assert set(params) == {"self", "clip"}

    def test_no_asr_or_text_machinery_is_reachable(self) -> None:
        source = inspect.getsource(ssa.prosodic_features)
        code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
        for forbidden in ("whisper", "tokenizer", "text_sentiment", "LexicalSolution"):
            assert forbidden not in code, f"{forbidden!r} reachable from the prosodic path"
