"""Voice-quality extraction: CPPS, HNR, jitter, shimmer, and the AVQI
composite, via praat-parselmouth.

CPPS is highlighted deliberately: it is the one dysphonia measure with
demonstrated validity on *continuous* speech rather than sustained vowels
(verified via web research), which is all a voice-first companion device
like Ami would ever have to work with.

**Scope, per docs/TASKS.md T11**: extraction and a whisper/breathiness
demo are implemented here. The per-speaker longitudinal baseline gate
described in DESIGN.md (separating presbyphonia -- a trait, drifting over
years -- from acute illness -- a state, deviating from a personal baseline
over days) is a *design proposal*, not built: we have no longitudinal
single-speaker data to validate it against, and shipping an unvalidated
gate under the name "voice health" would be exactly the kind of
overclaiming CLAUDE.md's rule 6 forbids.

All Praat calls are wrapped defensively: a pathologically short or silent
clip can make one component's underlying computation fail (e.g. no voiced
frames for jitter/shimmer) without that being a bug in this module -- in
that case the affected field is None, and AVQI (which needs all six) is
None too, rather than raising and aborting a whole eval run over one bad
clip.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import parselmouth
from parselmouth.praat import call

from ssa.types import AudioClip

logger = logging.getLogger(__name__)

# Standard Praat periodicity-detection bounds for typical adult speech.
_PITCH_FLOOR_HZ = 75.0
_PITCH_CEILING_HZ = 500.0

_SLOPE_TILT_LOW_HZ = 100.0
_SLOPE_TILT_HIGH_HZ = 8000.0

# AVQI (Acoustic Voice Quality Index), commonly-cited version (Maryn et al.):
# AVQI = [3.295 - 0.111*CPPS - 0.073*HNR - 0.213*SL + 2.789*SLdB
#         - 0.032*Slope + 0.077*Tilt] * 2.571
# Verified by cross-referencing two independent sources before use (the
# published coefficients vary by AVQI version -- e.g. a newer v03.01 revision
# uses different numbers -- so this is explicitly ONE cited version, not
# presented as a certified clinical tool). Output nominally in [0, 10],
# higher = poorer voice quality.
_AVQI_CONST = 3.295
_AVQI_CPPS = -0.111
_AVQI_HNR = -0.073
_AVQI_SHIMMER_LOCAL = -0.213
_AVQI_SHIMMER_LOCAL_DB = 2.789
_AVQI_SLOPE = -0.032
_AVQI_TILT = 0.077
_AVQI_SCALE = 2.571


@dataclass(frozen=True, slots=True)
class VoiceQuality:
    cpps: float | None  # smoothed cepstral peak prominence, dB
    hnr: float | None  # harmonics-to-noise ratio, dB
    jitter_local: float | None
    shimmer_local: float | None
    shimmer_local_db: float | None
    slope: float | None  # LTAS slope, dB/Hz
    tilt: float | None  # LTAS regression tilt, dB/Hz
    avqi: float | None  # composite, only if all six inputs available


@dataclass(frozen=True, slots=True)
class PitchStats:
    """Descriptive F0 (fundamental frequency) statistics over voiced frames.

    Unlike VoiceQuality's dysphonia measures, this is about prosody, not
    pathology: pitch height and variability are basic acoustic correlates
    of emotional arousal (higher/more variable F0 for excited states, lower
    F0 for sad/calm ones). Added for scripts/gen_emotion_probe_d1.py, which
    needs a no-fitting, no-model-inference instrument to measure whether
    Cartesia's emotion tags are audible at all. Same defensive convention
    as voice_quality: never raises, fields are None on a pathological clip."""

    f0_mean: float | None  # Hz
    f0_std: float | None  # Hz
    f0_range: float | None  # Hz, max - min over voiced frames


def voice_quality(clip: AudioClip) -> VoiceQuality:
    """Extract voice-quality measures for one clip. Never raises on a
    pathological clip (too short, silent, unvoiced) -- affected fields are
    None instead, per the module docstring."""
    snd = parselmouth.Sound(clip.samples.astype(np.float64), sampling_frequency=clip.sr)

    cpps = _safe(_extract_cpps, snd)
    hnr = _safe(_extract_hnr, snd)
    jitter_local, shimmer_local, shimmer_local_db = _safe(_extract_jitter_shimmer, snd) or (
        None,
        None,
        None,
    )
    slope, tilt = _safe(_extract_slope_tilt, snd) or (None, None)

    avqi = None
    if None not in (cpps, hnr, shimmer_local, shimmer_local_db, slope, tilt):
        raw = (
            _AVQI_CONST
            + _AVQI_CPPS * cpps
            + _AVQI_HNR * hnr
            + _AVQI_SHIMMER_LOCAL * shimmer_local
            + _AVQI_SHIMMER_LOCAL_DB * shimmer_local_db
            + _AVQI_SLOPE * slope
            + _AVQI_TILT * tilt
        )
        avqi = float(np.clip(raw * _AVQI_SCALE, 0.0, 10.0))

    return VoiceQuality(
        cpps=cpps,
        hnr=hnr,
        jitter_local=jitter_local,
        shimmer_local=shimmer_local,
        shimmer_local_db=shimmer_local_db,
        slope=slope,
        tilt=tilt,
        avqi=avqi,
    )


def pitch_stats(clip: AudioClip) -> PitchStats:
    """Extract F0 mean/std/range for one clip. Never raises on a
    pathological clip -- fields are None instead, per the module docstring."""
    snd = parselmouth.Sound(clip.samples.astype(np.float64), sampling_frequency=clip.sr)
    voiced = _safe(_extract_f0_values, snd)
    if voiced is None or len(voiced) == 0:
        return PitchStats(f0_mean=None, f0_std=None, f0_range=None)
    return PitchStats(
        f0_mean=float(voiced.mean()),
        f0_std=float(voiced.std()),
        f0_range=float(voiced.max() - voiced.min()),
    )


def _extract_f0_values(snd: parselmouth.Sound) -> np.ndarray:
    pitch = snd.to_pitch(pitch_floor=_PITCH_FLOOR_HZ, pitch_ceiling=_PITCH_CEILING_HZ)
    values = pitch.selected_array["frequency"]
    voiced = values[values > 0]  # unvoiced frames are reported as 0 Hz
    if len(voiced) == 0:
        raise ValueError("no voiced frames for pitch")
    return voiced


def _extract_cpps(snd: parselmouth.Sound) -> float:
    power_cepstrogram = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    return call(
        power_cepstrogram,
        "Get CPPS",
        "yes",
        0.02,
        0.0005,
        60,
        330,
        0.05,
        "Parabolic",
        0.001,
        0,
        "Straight",
        "Robust",
    )


def _extract_hnr(snd: parselmouth.Sound) -> float:
    harmonicity = snd.to_harmonicity()
    values = harmonicity.values
    voiced = values[values != -200]
    if len(voiced) == 0:
        raise ValueError("no voiced frames for HNR")
    return float(voiced.mean())


def _extract_jitter_shimmer(snd: parselmouth.Sound) -> tuple[float, float, float]:
    point_process = call(snd, "To PointProcess (periodic, cc)", _PITCH_FLOOR_HZ, _PITCH_CEILING_HZ)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_local_db = call(
        [snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    for name, val in [
        ("jitter", jitter),
        ("shimmer", shimmer_local),
        ("shimmer_dB", shimmer_local_db),
    ]:
        if val is None or np.isnan(val):
            raise ValueError(f"{name} undefined (likely no periodic frames)")
    return float(jitter), float(shimmer_local), float(shimmer_local_db)


def _extract_slope_tilt(snd: parselmouth.Sound) -> tuple[float, float]:
    ltas = call(snd, "To Ltas", 1.0)
    slope = call(ltas, "Get slope", 0, 1000, 1000, 10000, "energy")
    report = call(
        ltas, "Report spectral tilt", _SLOPE_TILT_LOW_HZ, _SLOPE_TILT_HIGH_HZ, "Linear", "Robust"
    )
    match = re.search(r"Slope:\s*(-?[\d.eE+-]+)\s*dB/Hz", report)
    if match is None:
        raise ValueError(f"could not parse tilt from report: {report!r}")
    tilt = float(match.group(1))
    return float(slope), tilt


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        logger.debug("%s failed: %s", fn.__name__, exc)
        return None
