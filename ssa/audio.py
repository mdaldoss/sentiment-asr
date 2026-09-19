"""Audio loading: any input file -> a validated AudioClip.

Resampling happens exactly once, here, at load time. Everything downstream
assumes 16 kHz mono float32 and relies on AudioClip's own validation to
enforce it (see ssa/types.py).

We peak-normalise (to use the full dynamic range and avoid clipping when a
downstream model expects roughly [-1, 1]) but deliberately do NOT
loudness-normalise. Loudness is itself a prosodic cue -- a whispered "I'm
fine" and a shouted "I'm fine" should not become acoustically identical
before they ever reach a model. Removing that variation would quietly erase
part of the signal this project exists to measure.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
from librosa import resample as librosa_resample

from ssa.types import SAMPLE_RATE, AudioClip

logger = logging.getLogger(__name__)

# Peak-normalise to this target so quiet source files aren't left with a tiny
# dynamic range, without ever exceeding it (which would clip).
_PEAK_TARGET = 0.95
# Below this peak amplitude a clip is treated as silence and left un-normalised
# rather than divided by a near-zero number.
_SILENCE_FLOOR = 1e-6


class AudioLoadError(RuntimeError):
    """Raised when a file cannot be read or decoded as audio."""


def load_clip(path: Path, clip_id: str | None = None) -> AudioClip:
    """Load one audio file to a validated, 16 kHz mono, peak-normalised AudioClip.

    Raises:
        AudioLoadError: the file is missing or cannot be decoded.
    """
    if not path.exists():
        raise AudioLoadError(f"no such file: {path}")

    try:
        samples, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:  # soundfile raises several distinct error types
        raise AudioLoadError(f"could not decode {path}: {exc}") from exc

    samples = _to_mono(samples)
    samples = _resample(samples, sr, SAMPLE_RATE, path=path)
    samples = _peak_normalise(samples)

    return AudioClip(
        clip_id=clip_id if clip_id is not None else path.stem,
        samples=samples,
        sr=SAMPLE_RATE,
        path=path,
    )


def clip_from_samples(samples: np.ndarray, sr: int, clip_id: str = "live") -> AudioClip:
    """Build a clip from raw in-memory samples -- browser-captured PCM, a
    test signal -- through the *same* mono/resample/peak-normalise path as
    `load_clip`.

    The demo exists to show what the evaluated system does, so its audio
    must not take a different route into the model than evaluation audio
    does. Sharing this function is what guarantees that; duplicating the
    three steps in the demo server is how the two silently diverge.
    """
    if samples.size == 0:
        raise AudioLoadError(f"{clip_id}: empty audio")
    if sr <= 0:
        raise AudioLoadError(f"{clip_id}: invalid sample rate {sr}")

    samples = _to_mono(np.asarray(samples, dtype=np.float32))
    samples = _resample(samples, sr, SAMPLE_RATE, path=Path(clip_id))
    samples = _peak_normalise(samples)
    return AudioClip(clip_id=clip_id, samples=samples, sr=SAMPLE_RATE)


def load_many(paths: Sequence[Path]) -> list[AudioClip]:
    """Load several clips, failing loudly (not silently dropping) on any error."""
    clips: list[AudioClip] = []
    errors: list[str] = []
    for p in paths:
        try:
            clips.append(load_clip(p))
        except AudioLoadError as exc:
            errors.append(str(exc))
    if errors:
        raise AudioLoadError(
            f"{len(errors)}/{len(paths)} file(s) failed to load:\n" + "\n".join(errors)
        )
    return clips


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Downmix multi-channel audio by averaging channels."""
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    if samples.ndim == 2:
        # soundfile shape is (n_frames, n_channels)
        return samples.mean(axis=1).astype(np.float32)
    raise AudioLoadError(f"unexpected audio shape {samples.shape}")


def _resample(samples: np.ndarray, sr_in: int, sr_out: int, *, path: Path) -> np.ndarray:
    if sr_in == sr_out:
        return samples
    if sr_in <= 0:
        raise AudioLoadError(f"{path}: invalid source sample rate {sr_in}")
    resampled = librosa_resample(samples, orig_sr=sr_in, target_sr=sr_out)
    return resampled.astype(np.float32, copy=False)


def _peak_normalise(samples: np.ndarray) -> np.ndarray:
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak < _SILENCE_FLOOR:
        return samples
    return (samples * (_PEAK_TARGET / peak)).astype(np.float32)
