"""Tests for E2/E4 manifest construction -- pure functions, no API calls."""

from __future__ import annotations

from scripts.gen_synthetic import CARRIERS, E2_PROSODY_TAGS, build_e2_manifest, build_e4_manifest
from ssa.manifest import validate_manifest


class TestBuildE2Manifest:
    def test_exact_shape(self) -> None:
        df = build_e2_manifest()
        assert len(df) == 90  # 3 text-sentiments x 3 prosodies x 5 carriers x 2 voices

    def test_congruent_incongruent_split(self) -> None:
        df = build_e2_manifest()
        assert df["is_congruent"].sum() == 30
        assert (~df["is_congruent"]).sum() == 60

    def test_all_clip_ids_and_paths_unique(self) -> None:
        df = build_e2_manifest()
        assert df["clip_id"].nunique() == 90
        assert df["path"].nunique() == 90

    def test_schema_valid(self) -> None:
        df = build_e2_manifest()
        validate_manifest(df)  # no repo_root -- files don't exist until generated

    def test_two_voices_represented(self) -> None:
        df = build_e2_manifest()
        assert df["speaker_id"].nunique() == 2

    def test_every_carrier_used(self) -> None:
        df = build_e2_manifest()
        for sentiment, sentences in CARRIERS.items():
            subset = df[df["text_sentiment"] == sentiment.value]
            assert subset["text"].nunique() == len(sentences)

    def test_prosody_tags_match_e2_config(self) -> None:
        df = build_e2_manifest()
        for sentiment, tag in E2_PROSODY_TAGS.items():
            subset = df[df["prosody_sentiment"] == sentiment.value]
            assert set(subset["emotion_tag"]) == {tag}


class TestBuildE4Manifest:
    def test_roughly_requested_size(self) -> None:
        """n // 3 per sentiment class -- exact count is a multiple of 3."""
        df = build_e4_manifest(n=300, seed=0)
        assert len(df) == 300

    def test_balanced_across_sentiments(self) -> None:
        df = build_e4_manifest(n=300, seed=0)
        counts = df["text_sentiment"].value_counts()
        assert counts.nunique() == 1  # exactly equal per class by construction

    def test_deterministic_given_seed(self) -> None:
        a = build_e4_manifest(n=60, seed=42)
        b = build_e4_manifest(n=60, seed=42)
        assert a["clip_id"].tolist() == b["clip_id"].tolist()

    def test_different_seeds_differ(self) -> None:
        a = build_e4_manifest(n=60, seed=1)
        b = build_e4_manifest(n=60, seed=2)
        assert a["clip_id"].tolist() != b["clip_id"].tolist()

    def test_congruent_by_construction(self) -> None:
        """E4 is training augmentation: text and prosody are always the
        same sentiment by design (unlike E2, which deliberately crosses them)."""
        df = build_e4_manifest(n=30, seed=0)
        assert df["is_congruent"].all()

    def test_split_is_always_train(self) -> None:
        df = build_e4_manifest(n=30, seed=0)
        assert (df["split"] == "train").all()

    def test_schema_valid(self) -> None:
        df = build_e4_manifest(n=30, seed=0)
        validate_manifest(df)

    def test_draws_from_full_labelled_vocabulary_not_just_e2s_three_tags(self) -> None:
        df = build_e4_manifest(n=300, seed=0)
        assert df["emotion_tag"].nunique() > 3
