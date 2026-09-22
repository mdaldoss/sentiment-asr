"""Tests for E3's pure analysis logic (recoverability CV, F0 rank
consistency) -- no audio, no model loading. Mirrors
tests/test_gen_emotion_probe_d1.py's approach for the same functions
applied to a 3-way (not 5-way) problem."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.eval_e3 import f0_rank_consistency, recoverability_cv


def _df(rows: list[dict]) -> pd.DataFrame:
    base = dict(
        f0_mean=150.0,
        f0_std=10.0,
        f0_range=30.0,
        egemaps_loudness=0.5,
        egemaps_hnr=15.0,
        egemaps_jitter=0.01,
        egemaps_shimmer_db=0.5,
        rms=0.1,
        speech_rate_cps=15.0,
    )
    return pd.DataFrame([{**base, **r} for r in rows])


class TestRecoverabilityCV:
    def test_perfectly_separable_scores_above_chance(self) -> None:
        rows = []
        for carrier in range(6):  # >=2 carriers per class needed for LOCO folds
            for sentiment, f0 in [("negative", 90.0), ("neutral", 120.0), ("positive", 160.0)]:
                rows.append({"carrier": f"c{carrier}", "intended": sentiment, "f0_mean": f0})
        result = recoverability_cv(_df(rows))
        assert result["accuracy"] is not None
        assert result["accuracy"] > result["chance"]
        assert result["chance"] == pytest.approx(1 / 3)

    def test_missing_features_dropped_not_crashed(self) -> None:
        rows = [
            {"carrier": f"c{i}", "intended": s, "f0_mean": 100.0 + i}
            for i in range(4)
            for s in ["negative", "neutral", "positive"]
        ]
        df = _df(rows)
        df.loc[0, "f0_mean"] = np.nan
        result = recoverability_cv(df)
        assert result["n_dropped_missing_features"] == 1

    def test_single_carrier_skips_cv(self) -> None:
        rows = [
            {"carrier": "only", "intended": s, "f0_mean": 100.0} for s in ["negative", "positive"]
        ]
        result = recoverability_cv(_df(rows))
        assert result["accuracy"] is None
        assert "note" in result


class TestF0RankConsistency:
    def test_chance_mean_rank_is_two_for_three_classes(self) -> None:
        rows = [
            {"carrier": f"c{i}", "intended": s, "f0_mean": 100.0}
            for i in range(3)
            for s in ["negative", "neutral", "positive"]
        ]
        result = f0_rank_consistency(_df(rows))
        assert result["chance_mean_rank"] == pytest.approx(2.0)

    def test_consistently_highest_class_has_mean_rank_near_one(self) -> None:
        rows = []
        for carrier in range(4):
            rows.append({"carrier": f"c{carrier}", "intended": "positive", "f0_mean": 200.0})
            rows.append({"carrier": f"c{carrier}", "intended": "neutral", "f0_mean": 120.0})
            rows.append({"carrier": f"c{carrier}", "intended": "negative", "f0_mean": 90.0})
        result = f0_rank_consistency(_df(rows))
        assert result["mean_rank_by_sentiment"]["positive"] == pytest.approx(1.0)
        assert result["mean_rank_by_sentiment"]["negative"] == pytest.approx(3.0)


class TestRecordedManifestPathsMatchTheirOwnDirectory:
    """Regression test: data/recorded0/manifest_speaker1.csv once pointed at
    data/recorded/*.wav (a stale copy-paste from before the second take was
    recorded), silently scoring take0 against take1's audio. Every
    manifest's path column must resolve inside its own directory."""

    @pytest.mark.parametrize("rel_dir", ["data/recorded0", "data/recorded"])
    def test_paths_point_inside_own_directory(self, rel_dir: str) -> None:
        manifest_path = Path(rel_dir) / "manifest_speaker1.csv"
        if not manifest_path.exists():
            pytest.skip(f"{manifest_path} not present in this checkout")
        df = pd.read_csv(manifest_path)
        bad = [p for p in df["path"] if not p.startswith(rel_dir + "/")]
        assert bad == [], f"{manifest_path}: paths outside {rel_dir}: {bad[:3]}"
