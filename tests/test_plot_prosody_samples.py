"""Tests for the prosody-sample selection logic -- no model inference, no
audio loading. CREMA-D's 470MB isn't committed (see .gitignore), so the
CREMA-D-dependent tests skip gracefully when data/cremad/ isn't present,
same convention as the rest of the suite for large downloaded fixtures."""

from __future__ import annotations

import pytest

from scripts.plot_prosody_samples import CREMAD_MANIFEST, select_cremad_sample, select_e3_sample

_have_cremad = CREMAD_MANIFEST.exists()


class TestSelectCremadSample:
    def test_missing_manifest_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scripts.plot_prosody_samples as mod

        monkeypatch.setattr(mod, "CREMAD_MANIFEST", mod.REPO_ROOT / "does" / "not" / "exist.csv")
        assert mod.select_cremad_sample().empty

    @pytest.mark.skipif(not _have_cremad, reason="CREMA-D not downloaded (`make data`)")
    def test_two_per_emotion_six_emotions(self) -> None:
        sample = select_cremad_sample(n_per_emotion=2)
        assert len(sample) == 12
        assert set(sample["label"]) == {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"}

    @pytest.mark.skipif(not _have_cremad, reason="CREMA-D not downloaded (`make data`)")
    def test_deterministic_given_seed(self) -> None:
        a = select_cremad_sample(seed=42)
        b = select_cremad_sample(seed=42)
        assert a["clip_id"].tolist() == b["clip_id"].tolist()

    @pytest.mark.skipif(not _have_cremad, reason="CREMA-D not downloaded (`make data`)")
    def test_paths_resolve_to_real_files(self) -> None:
        from pathlib import Path

        sample = select_cremad_sample()
        for p in sample["path"]:
            assert (Path(__file__).resolve().parent.parent / p).exists()


class TestSelectE3Sample:
    def test_missing_manifests_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scripts.plot_prosody_samples as mod

        monkeypatch.setattr(mod, "E3_MANIFESTS", {})
        assert mod.select_e3_sample().empty

    def test_selects_from_committed_e3_recordings(self) -> None:
        # E3 recordings ARE committed (unlike CREMA-D) -- see the eval_e3.py
        # commit -- so this runs unconditionally.
        sample = select_e3_sample()
        assert len(sample) > 0
        assert set(sample["prosody_sentiment"]) <= {"positive", "neutral", "negative"}
        assert all("whisper" not in cid for cid in sample["clip_id"])

    def test_no_duplicate_clip_ids(self) -> None:
        sample = select_e3_sample()
        assert sample["clip_id"].is_unique
