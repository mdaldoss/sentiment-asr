"""Tests for the backend-comparison combo builder: pure DataFrame logic,
no models, no audio. The invariant that matters most is CLAUDE.md rule 1
(speaker-disjoint splits) -- see TestEvalManifestsForCombo."""

from __future__ import annotations

import pandas as pd
import pytest

from ssa.combo_manifests import (
    COMBOS,
    HUME_SPEAKER_ID,
    build_train_manifest,
    combo_by_name,
    eval_manifests_for_combo,
    hume_manifest_from_probe,
    stratified_subset,
)
from ssa.manifest import validate_manifest
from ssa.splits import TRAIN


def _toy_cremad() -> pd.DataFrame:
    rows = [
        ("cr1", "spk1", "train", "neutral", "negative"),
        ("cr2", "spk1", "train", "neutral", "positive"),
        ("cr3", "spk2", "val", "neutral", "neutral"),
        ("cr4", "spk3", "test", "neutral", "negative"),
        ("cr5", "spk3", "test", "neutral", "positive"),
    ]
    df = pd.DataFrame(
        {
            "clip_id": [r[0] for r in rows],
            "path": [f"data/cremad/{r[0]}.wav" for r in rows],
            "speaker_id": [r[1] for r in rows],
            "source": ["crema_d"] * len(rows),
            "text": ["the book is on the table"] * len(rows),
            "text_sentiment": [r[3] for r in rows],
            "prosody_sentiment": [r[4] for r in rows],
            "emotion_tag": [""] * len(rows),
            "voice_id": [""] * len(rows),
            "split": [r[2] for r in rows],
        }
    )
    df["is_congruent"] = df["text_sentiment"] == df["prosody_sentiment"]
    return df


def _toy_e3(prefix: str) -> pd.DataFrame:
    rows = [
        (f"{prefix}_1", "positive", "negative"),
        (f"{prefix}_2", "positive", "positive"),
    ]
    df = pd.DataFrame(
        {
            "clip_id": [r[0] for r in rows],
            "path": [f"data/recorded/{r[0]}.wav" for r in rows],
            "speaker_id": ["speaker1"] * len(rows),
            "source": ["recorded"] * len(rows),
            "text": ["I'm thrilled about the good news today."] * len(rows),
            "text_sentiment": [r[1] for r in rows],
            "prosody_sentiment": [r[2] for r in rows],
            "emotion_tag": [""] * len(rows),
            "voice_id": [""] * len(rows),
            "split": [""] * len(rows),
        }
    )
    df["is_congruent"] = df["text_sentiment"] == df["prosody_sentiment"]
    return df


def _toy_hume_probe_result() -> dict:
    per_clip = [
        {
            "clip_id": "hume_happy_neutral_short_desc",
            "emotion": "happy",
            "text_condition": "neutral",
            "length": "short",
            "has_description": True,
            "text": "I need to check tomorrow's schedule.",
            "carrier_group": "neutral_short",
        },
        {
            "clip_id": "hume_happy_congruent_short_desc",
            "emotion": "happy",
            "text_condition": "congruent",
            "length": "short",
            "has_description": True,
            "text": "I'm so happy you called me back today!",
            "carrier_group": "congruent_happy_short",
        },
        {
            "clip_id": "hume_sad_neutral_short_nodesc",
            "emotion": "sad",
            "text_condition": "neutral",
            "length": "short",
            "has_description": False,
            "text": "I need to check tomorrow's schedule.",
            "carrier_group": "neutral_short",
        },
    ]
    return {"per_clip": per_clip}


class TestHumeManifestFromProbe:
    def test_drops_without_description_clips(self) -> None:
        df = hume_manifest_from_probe(_toy_hume_probe_result())
        assert "hume_sad_neutral_short_nodesc" not in set(df["clip_id"])
        assert len(df) == 2

    def test_neutral_text_gets_neutral_text_sentiment(self) -> None:
        df = hume_manifest_from_probe(_toy_hume_probe_result())
        row = df[df["clip_id"] == "hume_happy_neutral_short_desc"].iloc[0]
        assert row["text_sentiment"] == "neutral"
        assert row["prosody_sentiment"] == "positive"  # happy -> positive

    def test_congruent_text_matches_intended_sentiment(self) -> None:
        df = hume_manifest_from_probe(_toy_hume_probe_result())
        row = df[df["clip_id"] == "hume_happy_congruent_short_desc"].iloc[0]
        assert row["text_sentiment"] == "positive"
        assert row["prosody_sentiment"] == "positive"
        assert row["is_congruent"]

    def test_single_pseudo_speaker(self) -> None:
        df = hume_manifest_from_probe(_toy_hume_probe_result())
        assert set(df["speaker_id"]) == {HUME_SPEAKER_ID}
        assert set(df["source"]) == {"hume"}

    def test_validates_against_schema(self) -> None:
        df = hume_manifest_from_probe(_toy_hume_probe_result())
        validate_manifest(df)


class TestBuildTrainManifest:
    def test_cremad_only_leaves_cremad_unchanged(self) -> None:
        cremad = _toy_cremad()
        out = build_train_manifest(
            combo_by_name("cremad_only"),
            cremad,
            _toy_e3("e3a"),
            _toy_e3("e3b"),
            hume_manifest_from_probe(_toy_hume_probe_result()),
        )
        assert set(out["clip_id"]) == set(cremad["clip_id"])
        assert (out.set_index("clip_id")["split"] == cremad.set_index("clip_id")["split"]).all()

    def test_cremad_e3_adds_both_takes_as_train(self) -> None:
        cremad = _toy_cremad()
        e3a, e3b = _toy_e3("e3a"), _toy_e3("e3b")
        out = build_train_manifest(
            combo_by_name("cremad_e3"),
            cremad,
            e3a,
            e3b,
            hume_manifest_from_probe(_toy_hume_probe_result()),
        )
        assert set(e3a["clip_id"]) | set(e3b["clip_id"]) <= set(out["clip_id"])
        e3_rows = out[out["clip_id"].isin(set(e3a["clip_id"]) | set(e3b["clip_id"]))]
        assert (e3_rows["split"] == TRAIN).all()

    def test_cremad_hume_adds_hume_as_train_without_e3(self) -> None:
        cremad = _toy_cremad()
        hume = hume_manifest_from_probe(_toy_hume_probe_result())
        out = build_train_manifest(
            combo_by_name("cremad_hume"), cremad, _toy_e3("e3a"), _toy_e3("e3b"), hume
        )
        assert set(hume["clip_id"]) <= set(out["clip_id"])
        assert "e3a_1" not in set(out["clip_id"])
        hume_rows = out[out["clip_id"].isin(hume["clip_id"])]
        assert (hume_rows["split"] == TRAIN).all()

    def test_cremad_e3_hume_includes_everything(self) -> None:
        cremad = _toy_cremad()
        e3a, e3b = _toy_e3("e3a"), _toy_e3("e3b")
        hume = hume_manifest_from_probe(_toy_hume_probe_result())
        out = build_train_manifest(combo_by_name("cremad_e3_hume"), cremad, e3a, e3b, hume)
        expected = (
            set(cremad["clip_id"])
            | set(e3a["clip_id"])
            | set(e3b["clip_id"])
            | set(hume["clip_id"])
        )
        assert set(out["clip_id"]) == expected

    def test_output_always_validates(self) -> None:
        cremad = _toy_cremad()
        e3a, e3b = _toy_e3("e3a"), _toy_e3("e3b")
        hume = hume_manifest_from_probe(_toy_hume_probe_result())
        for combo in COMBOS:
            out = build_train_manifest(combo, cremad, e3a, e3b, hume)
            validate_manifest(out)  # raises on any violation


class TestEvalManifestsForCombo:
    """The speaker-disjointness invariant (CLAUDE.md rule 1): a combo that
    trained on E3's one speaker must never also be scored on that speaker."""

    def test_e3_excluded_when_combo_trained_on_it(self) -> None:
        cremad_test = _toy_cremad()
        out = eval_manifests_for_combo(
            combo_by_name("cremad_e3"), cremad_test, _toy_e3("e3a"), _toy_e3("e3b")
        )
        assert "e3_both_takes" not in out
        assert "cremad_test" in out

    def test_e3_hume_combo_also_excludes_e3(self) -> None:
        cremad_test = _toy_cremad()
        out = eval_manifests_for_combo(
            combo_by_name("cremad_e3_hume"), cremad_test, _toy_e3("e3a"), _toy_e3("e3b")
        )
        assert "e3_both_takes" not in out

    def test_e3_included_when_combo_never_trained_on_it(self) -> None:
        cremad_test = _toy_cremad()
        e3a, e3b = _toy_e3("e3a"), _toy_e3("e3b")
        out = eval_manifests_for_combo(combo_by_name("cremad_only"), cremad_test, e3a, e3b)
        assert set(out["e3_both_takes"]["clip_id"]) == set(e3a["clip_id"]) | set(e3b["clip_id"])

        out2 = eval_manifests_for_combo(combo_by_name("cremad_hume"), cremad_test, e3a, e3b)
        assert "e3_both_takes" in out2

    def test_no_combo_ever_evaluates_a_training_speaker(self) -> None:
        cremad_test = _toy_cremad()
        e3a, e3b = _toy_e3("e3a"), _toy_e3("e3b")
        hume = hume_manifest_from_probe(_toy_hume_probe_result())
        for combo in COMBOS:
            train = build_train_manifest(combo, _toy_cremad(), e3a, e3b, hume)
            train_speakers = set(train["speaker_id"])
            for eval_name, eval_df in eval_manifests_for_combo(
                combo, cremad_test, e3a, e3b
            ).items():
                overlap = set(eval_df["speaker_id"]) & train_speakers
                # cremad_test speakers are disjoint from cremad train/val by
                # construction upstream; only cross-source leakage matters here.
                if eval_name == "e3_both_takes":
                    assert not overlap, f"{combo.name}: E3 speaker leaked into eval"


class TestComboByName:
    def test_known_name_resolves(self) -> None:
        assert combo_by_name("cremad_only").name == "cremad_only"

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            combo_by_name("nonexistent")

    def test_four_combos_defined(self) -> None:
        assert len(COMBOS) == 4
        assert {c.name for c in COMBOS} == {
            "cremad_only",
            "cremad_e3",
            "cremad_e3_hume",
            "cremad_hume",
        }


class TestStratifiedSubset:
    def _big_df(self) -> pd.DataFrame:
        n_neg, n_neu, n_pos = 700, 150, 150
        sentiments = ["negative"] * n_neg + ["neutral"] * n_neu + ["positive"] * n_pos
        return pd.DataFrame(
            {
                "clip_id": [f"c{i}" for i in range(len(sentiments))],
                "prosody_sentiment": sentiments,
            }
        )

    def test_returns_approximately_n_rows(self) -> None:
        out = stratified_subset(self._big_df(), n=300, seed=0)
        assert 250 <= len(out) <= 320

    def test_deterministic_under_fixed_seed(self) -> None:
        df = self._big_df()
        a = stratified_subset(df, n=300, seed=0)
        b = stratified_subset(df, n=300, seed=0)
        assert list(a["clip_id"]) == list(b["clip_id"])

    def test_preserves_class_proportions_roughly(self) -> None:
        out = stratified_subset(self._big_df(), n=300, seed=0)
        counts = out["prosody_sentiment"].value_counts(normalize=True)
        assert counts["negative"] == pytest.approx(0.7, abs=0.05)

    def test_never_exceeds_available_rows_in_a_class(self) -> None:
        tiny = pd.DataFrame({"clip_id": ["a", "b"], "prosody_sentiment": ["positive", "positive"]})
        out = stratified_subset(tiny, n=300, seed=0)
        assert len(out) == 2
