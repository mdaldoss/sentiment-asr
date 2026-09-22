"""Builds the four training-data combinations for the backend comparison
requested directly by the user: WavLM+probe (permissive) vs
audeering/wav2vec2 (research), trained/evaluated on CREMA-D alone vs with
E3 (the user's own recordings) and/or the Hume Octave synthetic probe
folded in. Cartesia is deliberately excluded (D1 already found its
emotion tags don't render on this content -- folding flat audio into
training would only add noise, not signal).

**Why E3 cannot appear in both a combo's training data and its eval set**
(CLAUDE.md rule 1, "the single most important test in the repo"): E3 has
exactly one real speaker, recorded twice (`e3a`/take0, `e3b`/take1). Once
that speaker's clips are used for training, evaluating on the *other* take
of the *same* speaker is still speaker leakage -- a different clip_id does
not make a different speaker. So combos that add E3 to training use BOTH
takes as training data and are evaluated on CREMA-D-test only; combos that
don't train on E3 keep both takes as an eval-only set the model never saw
at all, same speaker-disjoint standard as everywhere else. This is a
constant-eval-set comparison in the CREMA-D column, and an eval-set-varies
comparison in the E3 column -- reported as such (CLAUDE.md rule 4), not
silently glossed over.

Hume's single synthetic voice ("Ava Song") is a pseudo-speaker that never
appears in either eval set, so it can be added to training in any combo
without touching speaker disjointness at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ssa.carriers import D1_CONGRUENT_TEXT, D1_EMOTION_SENTIMENT, D1_NEUTRAL_TEXT
from ssa.manifest import COLUMNS, add_congruence, validate_manifest
from ssa.splits import TRAIN
from ssa.types import Sentiment

HUME_SPEAKER_ID = "hume_ava_song"


@dataclass(frozen=True, slots=True)
class ComboSpec:
    name: str
    include_e3: bool
    include_hume: bool
    description: str


COMBOS: tuple[ComboSpec, ...] = (
    ComboSpec(
        "cremad_only",
        include_e3=False,
        include_hume=False,
        description="CREMA-D train split only -- the baseline.",
    ),
    ComboSpec(
        "cremad_e3",
        include_e3=True,
        include_hume=False,
        description="CREMA-D train + both E3 takes (the user's own recordings).",
    ),
    ComboSpec(
        "cremad_e3_hume",
        include_e3=True,
        include_hume=True,
        description="CREMA-D train + both E3 takes + Hume with-description clips.",
    ),
    ComboSpec(
        "cremad_hume",
        include_e3=False,
        include_hume=True,
        description="CREMA-D train + Hume with-description clips (no E3).",
    ),
)


def hume_manifest_from_probe(
    probe_result: dict, wav_dir_rel: str = "data/probes/hume"
) -> pd.DataFrame:
    """Convert `results/hume_probe.json`'s per_clip records into a validated
    manifest. **With-description clips only**: this session's own probe
    found the without-description half reads at/below chance (near-random
    prosody signal), so its "intended emotion" is not a trustworthy label --
    including it would silently violate CLAUDE.md rule 6 (don't present a
    reasoned label as a measured one) for exactly the clips least entitled
    to it.

    `prosody_sentiment` is the emotion's intended sentiment
    (`D1_EMOTION_SENTIMENT`), the same assumption CREMA-D and D1 already
    make for acted/synthetic emotion (the actor's, or here the
    `description` field's, intended emotion stands in for ground truth --
    there is no independent listener annotation for either). `text_sentiment`
    reflects the actual words spoken: neutral carrier text is
    `Sentiment.NEUTRAL`; congruent carrier text carries the same polarity as
    the intended emotion by construction (see ssa.carriers).
    """
    rows = []
    for clip in probe_result["per_clip"]:
        if not clip["has_description"]:
            continue
        emotion = clip["emotion"]
        text_condition = clip["text_condition"]
        length = clip["length"]
        prosody_sentiment = D1_EMOTION_SENTIMENT[emotion]
        text_sentiment = (
            Sentiment.NEUTRAL if text_condition == "neutral" else D1_EMOTION_SENTIMENT[emotion]
        )
        expected_text = (
            D1_NEUTRAL_TEXT[length]
            if text_condition == "neutral"
            else D1_CONGRUENT_TEXT[emotion][length]
        )
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "path": f"{wav_dir_rel}/{clip['clip_id']}.wav",
                "speaker_id": HUME_SPEAKER_ID,
                "source": "hume",
                "text": expected_text,
                "text_sentiment": text_sentiment.value,
                "prosody_sentiment": prosody_sentiment.value,
                "emotion_tag": emotion,
                "voice_id": "Ava Song",
                "split": "",
            }
        )
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    return add_congruence(df.drop(columns=["is_congruent"], errors="ignore"))


def _as_train(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = TRAIN
    return out


def build_train_manifest(
    combo: ComboSpec,
    cremad: pd.DataFrame,
    e3a: pd.DataFrame,
    e3b: pd.DataFrame,
    hume: pd.DataFrame,
) -> pd.DataFrame:
    """The manifest to train/fit on for one combo: CREMA-D's own
    train/val/test rows unchanged (so the research backend's val-fitted
    thresholds and CREMA-D-test eval stay identical across every combo),
    plus any extra sources folded in as extra `split="train"` rows."""
    parts = [cremad]
    if combo.include_e3:
        parts.append(_as_train(e3a))
        parts.append(_as_train(e3b))
    if combo.include_hume:
        parts.append(_as_train(hume))
    out = pd.concat(parts, ignore_index=True)
    validate_manifest(out)
    return out


def eval_manifests_for_combo(
    combo: ComboSpec, cremad_test: pd.DataFrame, e3a: pd.DataFrame, e3b: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Eval manifests this combo may honestly be scored on. `cremad_test`
    is always included (CREMA-D test speakers are never touched by any
    combo). `e3_both_takes` is included only when this combo did NOT train
    on E3 -- see module docstring."""
    out = {"cremad_test": cremad_test}
    if not combo.include_e3:
        out["e3_both_takes"] = pd.concat([e3a, e3b], ignore_index=True)
    return out


def stratified_subset(
    df: pd.DataFrame, n: int = 300, *, seed: int = 0, stratify_col: str = "prosody_sentiment"
) -> pd.DataFrame:
    """Deterministic, class-proportional sample of ~n rows -- mirrors the
    300-clip CREMA-D-test-subset convention this project already uses for
    its headline eval (see results/*__speaker_disjoint_test_subset.json),
    so a 4-combo x 2-backend comparison stays cheap without abandoning
    stratification. The same seed is used for every combo, so all combos
    are scored on the exact same CREMA-D clips."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby(stratify_col):
        k = min(len(group), max(1, round(n * len(group) / len(df))))
        idx = rng.choice(group.index.to_numpy(), size=k, replace=False)
        parts.append(group.loc[idx])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def combo_by_name(name: str) -> ComboSpec:
    for combo in COMBOS:
        if combo.name == name:
            return combo
    raise KeyError(f"no combo named {name!r}, expected one of {[c.name for c in COMBOS]}")
