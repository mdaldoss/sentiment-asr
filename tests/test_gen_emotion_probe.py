"""Tests for the D0 script's pure clustering logic -- no TTS/API calls."""

from __future__ import annotations

import numpy as np

from scripts.gen_emotion_probe import _best_k


class TestBestK:
    def test_finds_well_separated_clusters(self) -> None:
        rng = np.random.default_rng(0)
        cluster_a = rng.normal(loc=[0, 0], scale=0.05, size=(20, 2))
        cluster_b = rng.normal(loc=[5, 5], scale=0.05, size=(20, 2))
        cluster_c = rng.normal(loc=[10, 0], scale=0.05, size=(20, 2))
        points = np.concatenate([cluster_a, cluster_b, cluster_c])

        k, silhouette, labels = _best_k(points)
        assert k == 3
        assert silhouette > 0.8  # tight, well-separated clusters
        assert len(labels) == len(points)

    def test_labels_length_matches_input(self) -> None:
        rng = np.random.default_rng(1)
        points = rng.normal(size=(30, 2))
        _k, _sil, labels = _best_k(points)
        assert len(labels) == 30

    def test_respects_max_k_bound_for_small_input(self) -> None:
        """With only 5 points, max_k must be clamped below len(points)."""
        rng = np.random.default_rng(2)
        points = rng.normal(size=(5, 2))
        k, _sil, _labels = _best_k(points, min_k=2, max_k=10)
        assert k < 5
