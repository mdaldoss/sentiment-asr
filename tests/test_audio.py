"""Audio loading invariants: sample rate, channel count, and failure modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ssa.audio import AudioLoadError, load_clip, load_many
from ssa.types import SAMPLE_RATE


def _write_sine(
    path: Path, *, sr: int, duration_s: float = 0.5, channels: int = 1, freq: float = 440.0
) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels > 1:
        tone = np.tile(tone[:, None], (1, channels))
    sf.write(path, tone, sr)


def test_round_trips_native_rate_mono(tmp_path: Path) -> None:
    p = tmp_path / "tone.wav"
    _write_sine(p, sr=SAMPLE_RATE, channels=1)
    clip = load_clip(p)
    assert clip.sr == SAMPLE_RATE
    assert clip.samples.ndim == 1
    assert clip.samples.dtype == np.float32
    assert clip.clip_id == "tone"
    assert clip.path == p


def test_resamples_44100_stereo_to_16k_mono(tmp_path: Path) -> None:
    p = tmp_path / "hi_res_stereo.wav"
    _write_sine(p, sr=44_100, channels=2)
    clip = load_clip(p)
    assert clip.sr == SAMPLE_RATE
    assert clip.samples.ndim == 1
    # duration should be approximately preserved through resampling
    expected_len = round(0.5 * SAMPLE_RATE)
    assert abs(len(clip.samples) - expected_len) < SAMPLE_RATE * 0.05


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioLoadError, match="no such file"):
        load_clip(tmp_path / "does_not_exist.wav")


def test_peak_normalised_without_clipping(tmp_path: Path) -> None:
    p = tmp_path / "quiet.wav"
    _write_sine(p, sr=SAMPLE_RATE, channels=1)
    # scale the written tone down to be very quiet, then reload
    quiet, sr = sf.read(p, dtype="float32")
    sf.write(p, quiet * 0.01, sr)
    clip = load_clip(p)
    peak = float(np.abs(clip.samples).max())
    assert 0.5 < peak <= 0.95 + 1e-6, f"expected peak-normalised amplitude, got {peak}"


def test_silence_is_not_divided_by_zero(tmp_path: Path) -> None:
    p = tmp_path / "silence.wav"
    sf.write(p, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    clip = load_clip(p)  # must not raise / produce inf or nan
    assert np.all(np.isfinite(clip.samples))
    assert np.abs(clip.samples).max() == 0.0


def test_custom_clip_id(tmp_path: Path) -> None:
    p = tmp_path / "tone.wav"
    _write_sine(p, sr=SAMPLE_RATE)
    clip = load_clip(p, clip_id="my_custom_id")
    assert clip.clip_id == "my_custom_id"


def test_load_many_success(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        p = tmp_path / f"t{i}.wav"
        _write_sine(p, sr=SAMPLE_RATE)
        paths.append(p)
    clips = load_many(paths)
    assert len(clips) == 3
    assert all(c.sr == SAMPLE_RATE for c in clips)


def test_load_many_reports_all_failures(tmp_path: Path) -> None:
    good = tmp_path / "good.wav"
    _write_sine(good, sr=SAMPLE_RATE)
    bad1 = tmp_path / "missing1.wav"
    bad2 = tmp_path / "missing2.wav"
    with pytest.raises(AudioLoadError, match=r"2/3 file\(s\) failed"):
        load_many([good, bad1, bad2])
