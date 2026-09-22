"""CLI tests. Argument parsing and error paths are fast/network-free; the
full run (real model loading) is marked @pytest.mark.network."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.cli import _build_solution, main


class TestArgumentHandling:
    def test_missing_file_returns_error_code(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = main(["--audio", str(tmp_path / "does_not_exist.wav")])
        assert rc == 1
        assert "no such file" in capsys.readouterr().err

    def test_unknown_solution_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit):
            main(["--audio", "x.wav", "--solution", "bogus"])


class TestBuildSolution:
    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown solution"):
            _build_solution("bogus", "permissive")


@pytest.mark.network
@pytest.mark.slow
def test_end_to_end_on_real_clip(capsys: pytest.CaptureFixture) -> None:
    """Requires a trained probe and a real CREMA-D clip -- run `make data`
    and `make train` first. Verified manually during development: predicts
    'negative' (correctly) on 1001_DFA_ANG_XX.wav with a matching transcript."""
    clip_path = Path("data/cremad/AudioWAV/1001_DFA_ANG_XX.wav")
    if not clip_path.exists():
        pytest.skip("CREMA-D not fetched -- run `make data` first")
    rc = main(["--audio", str(clip_path), "--solution", "acoustic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Sentiment:" in out
