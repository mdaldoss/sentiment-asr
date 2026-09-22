"""Tests for E3's prompt-list construction (pure, no audio hardware needed).

The interactive recording loop itself (main, _record_one) needs a real
microphone and is not exercised here -- see the module docstring in
scripts/record_prompts.py.
"""

from __future__ import annotations

from scripts.record_prompts import build_prompts, prompts_to_manifest
from ssa.manifest import validate_manifest
from ssa.types import Sentiment


class TestBuildPrompts:
    def test_full_set_size(self) -> None:
        prompts = build_prompts("spk1", reduced=False)
        assert len(prompts) == 30  # 3 x 3 x 3 + 3 whisper

    def test_reduced_set_size(self) -> None:
        prompts = build_prompts("spk2", reduced=True)
        assert len(prompts) == 12  # 3 x 2 x 2, no whisper

    def test_full_set_includes_whisper_prompts(self) -> None:
        prompts = build_prompts("spk1", reduced=False)
        whisper = [p for p in prompts if p.is_whisper_prompt]
        assert len(whisper) == 3
        assert all(p.prosody_sentiment is None for p in whisper)

    def test_reduced_set_has_no_whisper_prompts(self) -> None:
        prompts = build_prompts("spk2", reduced=True)
        assert not any(p.is_whisper_prompt for p in prompts)

    def test_reduced_set_excludes_neutral_prosody(self) -> None:
        """Reduced targets only positive/negative -- the most informative
        contrast for a short second-speaker session."""
        prompts = build_prompts("spk2", reduced=True)
        prosodies = {p.prosody_sentiment for p in prompts if not p.is_whisper_prompt}
        assert prosodies == {Sentiment.POSITIVE, Sentiment.NEGATIVE}

    def test_clip_ids_unique_and_contain_speaker_id(self) -> None:
        prompts = build_prompts("alice", reduced=False)
        ids = [p.clip_id for p in prompts]
        assert len(ids) == len(set(ids))
        assert all("alice" in cid for cid in ids)

    def test_all_three_prosody_targets_present_in_full_set(self) -> None:
        prompts = build_prompts("spk1", reduced=False)
        prosodies = {p.prosody_sentiment for p in prompts if not p.is_whisper_prompt}
        assert prosodies == set(Sentiment)


class TestPromptsToManifest:
    def test_schema_valid(self) -> None:
        prompts = build_prompts("spk1", reduced=False)
        df = prompts_to_manifest(prompts, "spk1")
        validate_manifest(df)  # no repo_root -- files don't exist until recorded

    def test_whisper_prompts_are_congruent_by_convention(self) -> None:
        prompts = build_prompts("spk1", reduced=False)
        df = prompts_to_manifest(prompts, "spk1")
        whisper_rows = df[df["clip_id"].str.contains("whisper")]
        assert whisper_rows["is_congruent"].all()

    def test_speaker_id_propagated(self) -> None:
        prompts = build_prompts("bob", reduced=True)
        df = prompts_to_manifest(prompts, "bob")
        assert (df["speaker_id"] == "bob").all()

    def test_source_is_recorded(self) -> None:
        prompts = build_prompts("spk1", reduced=False)
        df = prompts_to_manifest(prompts, "spk1")
        assert (df["source"] == "recorded").all()

    def test_incongruent_rows_exist(self) -> None:
        """Sanity check the actual point of E3: some prompts must ask for
        text and tone to disagree."""
        prompts = build_prompts("spk1", reduced=False)
        df = prompts_to_manifest(prompts, "spk1")
        assert (~df["is_congruent"]).sum() > 0
