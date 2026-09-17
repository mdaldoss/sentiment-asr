"""Whisper/breathiness detection via eGeMAPS (openSMILE), 88 functionals.

Product-real for Ami: a whispering user may be unwell, unable to speak up,
or the device itself should turn its own volume down to match. The
relevant eGeMAPS features are loudness and HNR (harmonics-to-noise ratio)
-- a whisper is characteristically quiet AND noisy/breathy (low harmonic
energy relative to noise), which is a different acoustic signature from
merely "quiet but clear" speech.

Default thresholds here are reasonable priors, not fitted to real data --
`fit_whisper_thresholds` exists to calibrate them once E3's deliberately
whispered clips (scripts/record_prompts.py's WHISPER_PROMPTS) are recorded.
Until then, is_whispered's defaults should be treated as a demo, not a
validated classifier -- consistent with CLAUDE.md rule 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import opensmile

from ssa.types import AudioClip

_SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

_LOUDNESS_COL = "loudness_sma3_amean"
_HNR_COL = "HNRdBACF_sma3nz_amean"


@dataclass(frozen=True, slots=True)
class ParalinguisticFeatures:
    loudness_mean: float
    hnr_mean: float
    jitter_mean: float
    shimmer_db_mean: float


@dataclass(frozen=True, slots=True)
class WhisperThresholds:
    """hnr_max is the primary, validated discriminator: a synthetic check
    (clean periodic tone vs. pure unvoiced noise) confirmed HNR separates
    them correctly and by a wide margin (~8.8 dB vs. 0.0 dB). loudness_max
    is a secondary signal, deliberately loose by default -- a synthetic
    check surfaced a real surprise here: eGeMAPS's "loudness" is a
    perceptual loudness estimate, not raw amplitude, and broadband noise
    can score as perceptually LOUDER than a clean tone even at a much lower
    raw amplitude, because its energy spreads across more
    perceptually-weighted frequency bands. A real whisper's spectral shape
    (band-limited turbulent noise, shaped by the vocal tract) is not pure
    white noise, so this may behave better on real speech -- but that needs
    E3's actual whispered recordings to confirm, not asserted here."""

    hnr_max: float
    loudness_max: float


# Priors, not fitted -- see module docstring. loudness_max is intentionally
# permissive (see WhisperThresholds docstring) so it doesn't block a
# detection HNR would otherwise correctly make.
DEFAULT_THRESHOLDS = WhisperThresholds(hnr_max=5.0, loudness_max=2.0)


def extract_features(clip: AudioClip) -> ParalinguisticFeatures:
    df = _SMILE.process_signal(clip.samples.astype(np.float32), clip.sr)
    row = df.iloc[0]
    return ParalinguisticFeatures(
        loudness_mean=float(row[_LOUDNESS_COL]),
        hnr_mean=float(row[_HNR_COL]),
        jitter_mean=float(row["jitterLocal_sma3nz_amean"]),
        shimmer_db_mean=float(row["shimmerLocaldB_sma3nz_amean"]),
    )


def is_whispered(
    features: ParalinguisticFeatures, thresholds: WhisperThresholds = DEFAULT_THRESHOLDS
) -> bool:
    """HNR is the necessary condition (validated); loudness must also be
    below its (deliberately loose) threshold. See WhisperThresholds."""
    return (
        features.hnr_mean <= thresholds.hnr_max
        and features.loudness_mean <= thresholds.loudness_max
    )


def fit_whisper_thresholds(
    normal_features: list[ParalinguisticFeatures], whisper_features: list[ParalinguisticFeatures]
) -> WhisperThresholds:
    """Calibrate thresholds as the midpoint between the two groups' means.
    Intended to be called once real labelled examples exist (E3's whisper
    prompts) -- a simple, auditable choice over a trained classifier, since
    the point is a demo-quality threshold, not a tuned model."""
    if not normal_features or not whisper_features:
        raise ValueError("need at least one example of each class to fit thresholds")

    normal_loudness = np.mean([f.loudness_mean for f in normal_features])
    whisper_loudness = np.mean([f.loudness_mean for f in whisper_features])
    normal_hnr = np.mean([f.hnr_mean for f in normal_features])
    whisper_hnr = np.mean([f.hnr_mean for f in whisper_features])

    return WhisperThresholds(
        hnr_max=float((normal_hnr + whisper_hnr) / 2),
        loudness_max=float((normal_loudness + whisper_loudness) / 2),
    )
