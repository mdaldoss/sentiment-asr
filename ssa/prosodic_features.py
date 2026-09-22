"""Explicit prosodic features: the physics of the waveform, no embeddings.

This exists because of a measurement, not a hunch. `scripts/diagnose_e5.py`
showed that a plain logistic regression on nine hand-built descriptors
recovers the intended delivery on E5 at **0.811** against 0.333 chance --
higher than the same instruments score on real human speech, and far above
what either deep acoustic backend achieves on that audio (UAR 0.511 and
0.400). Simple prosodic descriptors were beating a 95M-parameter encoder at
the task the encoder exists for, so they deserve to be a solution rather
than a diagnostic.

**Two sources, deliberately:**

1. **eGeMAPS v02** (openSMILE, 88 functionals) -- the standardised
   minimalistic parameter set for affective speech: F0 statistics, loudness,
   jitter, shimmer, HNR, spectral slope and balance, MFCCs, voiced-segment
   rates. This is the field's default and is used unchanged.

2. **Contour features** (Praat via parselmouth) -- what the functionals
   under-represent. eGeMAPS summarises *local* rising/falling slopes but
   carries no single global trend, and the diagnostic's leading hypothesis
   for why the WavLM probe underperforms is precisely that pooling destroys
   the temporal contour prosody lives in. So we add the direction and spread
   of the whole utterance: global F0 and energy slope, dynamic ranges,
   pause structure, and CPPS.

Both are **mathematically blind to vocabulary**. Nothing here can read a
word, which is exactly the property the E5 result showed the audeering
model lacks -- it followed the transcript on 79% of contradictory clips
despite never receiving one. A model built only on these numbers cannot do
that, by construction rather than by hope.

Every feature is a named scalar (`FEATURE_NAMES`), so a prediction can be
attributed to measurable quantities rather than to an opaque vector.

Extraction is not cheap (~0.3s/clip), so this implements `embeddings.py`'s
`Embedder` protocol and reuses that module's resumable, clip_id-keyed cache
rather than growing a second caching mechanism.
"""

from __future__ import annotations

import logging

import numpy as np
import opensmile
import parselmouth

from ssa.types import AudioClip
from ssa.voicehealth import _extract_cpps, _extract_f0_values

logger = logging.getLogger(__name__)

_SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

EGEMAPS_NAMES: tuple[str, ...] = tuple(_SMILE.feature_names)

# What eGeMAPS's functionals summarise only locally, or not at all. Named
# for what they measure rather than how they're computed.
CONTOUR_NAMES: tuple[str, ...] = (
    "f0_slope_st_per_s",  # global melodic trend: falling = classic sadness cue
    "f0_dynamic_range_st",  # p95-p5 in semitones: monotone vs expressive
    "f0_iqr_st",
    "energy_slope_db_per_s",  # does the utterance fade out or hold up
    "energy_contrast_db",  # p95-p5 loudness: flat delivery has little
    "voiced_fraction",
    "pause_ratio",
    "cpps_db",  # the dysphonia measure valid on continuous speech
)

FEATURE_NAMES: tuple[str, ...] = EGEMAPS_NAMES + CONTOUR_NAMES
N_FEATURES = len(FEATURE_NAMES)

_SILENCE_DB = -60.0  # frames quieter than this count as pause, not speech


def _semitones(f0_hz: np.ndarray) -> np.ndarray:
    """Hz -> semitones re 100 Hz. Perceptually linear, and it makes slopes
    comparable between a low male and a high female voice, which matters
    because E5 crosses both."""
    return 12.0 * np.log2(np.maximum(f0_hz, 1e-6) / 100.0)


def _linear_slope(values: np.ndarray, times: np.ndarray) -> float:
    if len(values) < 3:
        return float("nan")
    span = times[-1] - times[0]
    if span <= 0:
        return float("nan")
    return float(np.polyfit(times, values, 1)[0])


def _contour_features(clip: AudioClip) -> dict[str, float]:
    """Global trend and spread of pitch and energy across the utterance."""
    out = dict.fromkeys(CONTOUR_NAMES, float("nan"))
    snd = parselmouth.Sound(clip.samples.astype(np.float64), sampling_frequency=clip.sr)

    try:
        pitch = snd.to_pitch()
        f0 = _extract_f0_values(snd)
        if f0.size >= 3:
            st = _semitones(f0)
            # Voiced frame times, so the slope is per second of speech
            # rather than per frame index.
            voiced_times = np.asarray(pitch.xs())[np.asarray(pitch.selected_array["frequency"]) > 0]
            if len(voiced_times) == len(st):
                out["f0_slope_st_per_s"] = _linear_slope(st, voiced_times)
            out["f0_dynamic_range_st"] = float(np.percentile(st, 95) - np.percentile(st, 5))
            out["f0_iqr_st"] = float(np.percentile(st, 75) - np.percentile(st, 25))
        total_frames = len(pitch.xs())
        if total_frames:
            out["voiced_fraction"] = float(f0.size / total_frames)
    except Exception:
        logger.debug("pitch contour failed for %s", clip.clip_id)

    try:
        intensity = snd.to_intensity()
        db = np.asarray(intensity.values).ravel()
        times = np.asarray(intensity.xs())
        speech = db > _SILENCE_DB
        if speech.sum() >= 3:
            out["energy_slope_db_per_s"] = _linear_slope(db[speech], times[speech])
            out["energy_contrast_db"] = float(
                np.percentile(db[speech], 95) - np.percentile(db[speech], 5)
            )
        if db.size:
            out["pause_ratio"] = float(1.0 - speech.mean())
    except Exception:
        logger.debug("intensity contour failed for %s", clip.clip_id)

    out["cpps_db"] = _extract_cpps(snd)
    return out


class ProsodicExtractor:
    """eGeMAPS + contour features as one vector, in `FEATURE_NAMES` order.

    Implements embeddings.py's `Embedder` protocol so the existing
    resumable cache works unchanged.
    """

    name = "egemaps+contour"

    def embed(self, clip: AudioClip) -> np.ndarray:
        try:
            frame = _SMILE.process_signal(clip.samples, clip.sr)
            egemaps = frame.to_numpy().ravel().astype(np.float32)
        except Exception:
            # Never silently drop a clip (CLAUDE.md): emit NaNs and let the
            # pipeline's imputer handle them, so the clip is still counted.
            logger.warning("eGeMAPS extraction failed for %s -- emitting NaN", clip.clip_id)
            egemaps = np.full(len(EGEMAPS_NAMES), np.nan, dtype=np.float32)

        contour = _contour_features(clip)
        tail = np.array([contour[name] for name in CONTOUR_NAMES], dtype=np.float32)
        vector = np.concatenate([egemaps, tail])
        if vector.shape[0] != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {vector.shape[0]}")
        return vector


class MemoizingExtractor:
    """Extracts each clip once, keyed by clip_id.

    Scoring several candidate classifiers over the same evaluation set
    otherwise re-runs openSMILE and Praat once per classifier, which is
    the dominant cost (~0.6s/clip) and produces identical vectors every
    time. Shared between solutions, it makes a four-model comparison cost
    what one model costs.
    """

    def __init__(self, inner: ProsodicExtractor | None = None) -> None:
        self._inner = inner if inner is not None else ProsodicExtractor()
        self._cache: dict[str, np.ndarray] = {}
        self.name = self._inner.name

    def embed(self, clip: AudioClip) -> np.ndarray:
        cached = self._cache.get(clip.clip_id)
        if cached is None:
            cached = self._inner.embed(clip)
            self._cache[clip.clip_id] = cached
        return cached


def describe(vector: np.ndarray, top_k: int = 8) -> list[tuple[str, float]]:
    """Name the largest-magnitude features of one vector -- the readable
    counterpart to an embedding, useful when explaining a prediction."""
    if vector.shape[0] != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {vector.shape[0]}")
    order = np.argsort(-np.abs(np.nan_to_num(vector)))[:top_k]
    return [(FEATURE_NAMES[i], float(vector[i])) for i in order]
