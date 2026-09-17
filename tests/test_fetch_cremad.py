"""Fast, network-free tests for the CREMA-D filename parsing and row-building
logic. The full download-and-extract path is exercised manually (and in CI
via `make data`), not here -- 470 MB is not something a unit test should fetch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.fetch_cremad import CremaDFetchError, _build_rows, _parse_filename
from ssa.manifest import validate_manifest


def test_parses_standard_filename() -> None:
    fields = _parse_filename(Path("1001_DFA_ANG_XX.wav"))
    assert fields == {"speaker": "1001", "sentence": "DFA", "emotion": "ANG", "intensity": "XX"}


def test_parses_the_known_truncated_intensity_quirk() -> None:
    """1040_ITH_SAD_X.wav is a real anomaly in the upstream distribution: a
    single-character intensity code instead of the usual two. We don't use
    intensity, so this should parse rather than be dropped."""
    fields = _parse_filename(Path("1040_ITH_SAD_X.wav"))
    assert fields["speaker"] == "1040"
    assert fields["emotion"] == "SAD"
    assert fields["intensity"] == "X"


def test_unparseable_filename_raises() -> None:
    with pytest.raises(CremaDFetchError, match="unexpected filename format"):
        _parse_filename(Path("not_a_valid_name.wav"))


@pytest.fixture(autouse=True)
def _repo_root_is_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_rows stores paths relative to the module's REPO_ROOT constant.
    Point it at tmp_path for every test in this file so fixture files (which
    live under tmp_path) resolve correctly."""
    monkeypatch.setattr("scripts.fetch_cremad.REPO_ROOT", tmp_path)


def _make_fake_audio_dir(tmp_path: Path, filenames: list[str]) -> Path:
    audio_dir = tmp_path / "AudioWAV"
    audio_dir.mkdir()
    for name in filenames:
        (audio_dir / name).touch()
    return audio_dir


def test_build_rows_produces_valid_manifest(tmp_path: Path) -> None:
    audio_dir = _make_fake_audio_dir(
        tmp_path,
        [
            "1001_DFA_ANG_XX.wav",
            "1001_IEO_HAP_HI.wav",
            "1002_TIE_NEU_XX.wav",
        ],
    )
    df = _build_rows(audio_dir)
    # _build_rows deliberately doesn't populate `split` -- that's main()'s job,
    # via speaker_disjoint_split / random_split. Fill it here to validate the
    # rest of the schema in isolation.
    df["split"] = ""

    validate_manifest(df, repo_root=tmp_path)
    assert len(df) == 3
    assert set(df["speaker_id"]) == {"1001", "1002"}
    assert set(df["text_sentiment"]) == {"neutral"}, "CREMA-D text is always neutral"

    row = df[df["emotion_tag"] == "ANG"].iloc[0]
    assert row["prosody_sentiment"] == "negative"
    assert row["is_congruent"] == (row["text_sentiment"] == row["prosody_sentiment"])


def test_build_rows_raises_on_unparseable_file(tmp_path: Path) -> None:
    audio_dir = _make_fake_audio_dir(tmp_path, ["1001_DFA_ANG_XX.wav", "garbage.wav"])
    with pytest.raises(CremaDFetchError, match="could not be parsed"):
        _build_rows(audio_dir)


def test_build_rows_raises_on_empty_dir(tmp_path: Path) -> None:
    audio_dir = tmp_path / "AudioWAV"
    audio_dir.mkdir()
    with pytest.raises(CremaDFetchError, match="no wav files"):
        _build_rows(audio_dir)
