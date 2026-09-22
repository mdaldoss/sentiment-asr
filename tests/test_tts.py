"""Cartesia wrapper tests -- mocked, no real API calls or network access."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ssa.tts import generate_clip, get_client


class TestGetClient:
    def test_missing_key_raises_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="CARTESIA_API_KEY not set"):
            get_client()

    def test_present_key_constructs_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARTESIA_API_KEY", "fake-key-for-test")
        with patch("ssa.tts.Cartesia") as mock_cartesia:
            get_client()
            mock_cartesia.assert_called_once_with(api_key="fake-key-for-test")


class TestGenerateClip:
    def test_writes_response_bytes_to_out_path(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.tts.generate.return_value.read.return_value = b"fake wav bytes"
        out_path = tmp_path / "sub" / "clip.wav"

        generate_clip(mock_client, text="hello", emotion="happy", voice_id="v1", out_path=out_path)

        assert out_path.read_bytes() == b"fake wav bytes"

    def test_passes_emotion_and_voice_through(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.tts.generate.return_value.read.return_value = b""
        out_path = tmp_path / "clip.wav"

        generate_clip(mock_client, text="hi", emotion="sad", voice_id="v42", out_path=out_path)

        _args, kwargs = mock_client.tts.generate.call_args
        assert kwargs["generation_config"]["emotion"] == "sad"
        assert kwargs["voice"]["id"] == "v42"
        assert kwargs["transcript"] == "hi"

    def test_default_omits_speed_and_uses_default_model(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.tts.generate.return_value.read.return_value = b""
        out_path = tmp_path / "clip.wav"

        generate_clip(mock_client, text="hi", emotion="sad", voice_id="v42", out_path=out_path)

        _args, kwargs = mock_client.tts.generate.call_args
        assert "speed" not in kwargs["generation_config"]
        assert kwargs["model_id"] == "sonic-3"

    def test_speed_and_model_id_pass_through_when_given(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.tts.generate.return_value.read.return_value = b""
        out_path = tmp_path / "clip.wav"

        generate_clip(
            mock_client,
            text="hi",
            emotion="angry",
            voice_id="v42",
            out_path=out_path,
            speed=1.15,
            model_id="sonic-3.5",
        )

        _args, kwargs = mock_client.tts.generate.call_args
        assert kwargs["generation_config"]["speed"] == 1.15
        assert kwargs["model_id"] == "sonic-3.5"
