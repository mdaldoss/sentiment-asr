"""Voice-quality extraction tests, verified against real parselmouth calls
on synthetic signals -- these are not mocked, since the whole point is to
confirm the Praat command wiring is correct (several were caught wrong
during development: e.g. init_weights vs post_init isn't relevant here,
but the CPPS/tilt commands required verifying exact Praat argument syntax
before trusting them)."""

from __future__ import annotations

import numpy as np

from ssa.types import SAMPLE_RATE, AudioClip
from ssa.voicehealth import voice_quality


def _harmonic_clip(amplitude: float = 0.5, f0: float = 120.0) -> AudioClip:
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    sig = np.zeros_like(t)
    for h in range(1, 6):
        sig += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    sig = sig / np.abs(sig).max() * amplitude
    return AudioClip(clip_id="harmonic", samples=sig.astype(np.float32))


def _breathy_clip(seed: int = 0) -> AudioClip:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    harmonic = np.sin(2 * np.pi * 120 * t)
    noise = rng.standard_normal(SAMPLE_RATE)
    sig = 0.15 * harmonic + 0.85 * noise
    sig = sig / np.abs(sig).max() * 0.5
    return AudioClip(clip_id="breathy", samples=sig.astype(np.float32))


class TestVoiceQuality:
    def test_clean_periodic_signal_has_high_cpps_and_hnr(self) -> None:
        vq = voice_quality(_harmonic_clip())
        assert vq.cpps is not None and vq.cpps > 10.0
        assert vq.hnr is not None and vq.hnr > 20.0

    def test_breathy_signal_has_lower_cpps_and_hnr_than_clean(self) -> None:
        clean = voice_quality(_harmonic_clip())
        breathy = voice_quality(_breathy_clip())
        assert breathy.cpps < clean.cpps
        assert breathy.hnr < clean.hnr

    def test_clean_signal_has_near_zero_jitter_and_shimmer(self) -> None:
        vq = voice_quality(_harmonic_clip())
        assert vq.jitter_local is not None and vq.jitter_local < 0.01
        assert vq.shimmer_local is not None and vq.shimmer_local < 0.01

    def test_avqi_computed_when_all_components_available(self) -> None:
        vq = voice_quality(_harmonic_clip())
        assert vq.avqi is not None
        assert 0.0 <= vq.avqi <= 10.0

    def test_avqi_none_when_a_component_is_missing(self) -> None:
        """Pure noise has no periodic frames -> jitter/shimmer are None ->
        AVQI (which needs all six inputs) must be None, not a bogus number."""
        rng = np.random.default_rng(1)
        pure_noise = AudioClip(
            clip_id="noise", samples=(rng.standard_normal(SAMPLE_RATE) * 0.3).astype(np.float32)
        )
        vq = voice_quality(pure_noise)
        assert vq.avqi is None

    def test_silence_does_not_raise(self) -> None:
        silence = AudioClip(clip_id="silence", samples=np.zeros(SAMPLE_RATE, dtype=np.float32))
        vq = voice_quality(silence)  # must not raise
        assert vq.hnr is None  # no voiced frames in silence

    def test_low_amplitude_clean_signal_still_extracts(self) -> None:
        """A quiet but clear voice -- distinct from breathy/whispered --
        should still yield a usable CPPS/HNR."""
        vq = voice_quality(_harmonic_clip(amplitude=0.05))
        assert vq.cpps is not None
        assert vq.hnr is not None
