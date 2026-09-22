#!/usr/bin/env python3
"""E6: the Zurich multi-speaker recordings -> decoded audio + a manifest.

This is the dataset the project has been missing. Every human result so far
rests on **one** speaker (E3, recorded twice); E5 is synthetic. E6 has eight
speakers reading 35 sentences, each sentence delivered with two or three
different intended emotions, and most of its clips are incongruent -- so
it is simultaneously the first real cross-speaker evaluation set and the
first *human, multi-speaker* incongruence set, which is what PSI was
designed for and has never had.

Four things this script has to get right, each of which would silently
corrupt results if it got them wrong:

1. **Decoding.** The clips are WebM/Opus at 48 kHz, plus one session in
   `.m4a`. `soundfile` reads neither and there is no ffmpeg binary here, so
   PyAV (which bundles the libraries) decodes and resamples to the
   project's 16 kHz mono, written as PCM WAV. Everything downstream then
   uses the ordinary `load_clip` path with no special case.

2. **Counting what was built.** Every metadata row currently resolves to a
   file, across both the .webm and .m4a sessions. The existence check stays
   anyway, and reports any row it cannot resolve rather than skipping it:
   CLAUDE.md forbids silently dropping clips, and a mixed-extension dataset
   is exactly where a glob that assumes one container quietly loses a whole
   session -- which is precisely what an early pass through this data did,
   reporting 20 files missing that were simply in the other container.

3. **The speaker who is already in the project.** The `marco` folder is the
   same person as E3's `speaker1` -- the repository owner, who recorded E3
   (see the `audio recorded v1/v2 marco` commits). Giving him a fresh
   speaker id would let a model train on E6 and be evaluated on E3 with the
   same voice on both sides, which is exactly what CLAUDE.md rule 1 exists
   to prevent. He therefore keeps the id `speaker1`, so
   `assert_speaker_disjoint` can *see* the overlap instead of depending on
   anyone remembering it.

4. **The text axis.** The metadata labels intended *delivery* only -- the
   same sentence appears with two or three different emotions, which is how
   we know it is prosody and not wording. But PSI needs the lexical valence
   too, and the dataset does not carry one. It is assigned by hand below,
   from the sentence text alone. That is the analyst's judgment, not the
   dataset creators', and it is written out explicitly so a reader can
   disagree with any single row.

Usage:
    uv run python scripts/build_zurich.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.audio import SAMPLE_RATE  # noqa: E402
from ssa.manifest import write_manifest  # noqa: E402
from ssa.types import Sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_DIR = REPO_ROOT / "data" / "zurich_speech_sentiment_dataset"
OUT_DIR = REPO_ROOT / "data" / "zurich"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
SUMMARY_PATH = REPO_ROOT / "results" / "e6_build_summary.json"

# The E3 speaker, who also recorded here. Same id on purpose -- see the
# module docstring; this is what lets the leakage assertion fire.
E3_SPEAKER_ID = "speaker1"
SPEAKER_IDS: dict[str, str] = {
    "marco": E3_SPEAKER_ID,
    "Cate": "zurich_cate",
    "Paoli": "zurich_paoli",
    "Silvia": "zurich_silvia",
    "Matteo": "zurich_matteo",
    "Isinsu_Yurdunusever": "zurich_isinsu",
    "Bonzus": "zurich_bonzus",
    "Egi": "zurich_egi",
}

# Lexical valence of each sentence, assigned by hand from the words alone.
#
# The rule applied: POSITIVE or NEGATIVE only where the wording carries that
# valence on its own, NEUTRAL wherever a plain reading is genuinely
# two-sided. "I can't believe this happened" and "Whatever, it's fine" are
# NEUTRAL here not because they are bland but because the words really do
# support either reading -- forcing a side would manufacture incongruence
# that is not in the text, and PSI would then measure our labelling rather
# than the model. This is the same conservatism CLAUDE.md rule 3 applies to
# ambiguous emotion tags.
TEXT_SENTIMENT: dict[str, Sentiment] = {
    "Are you serious?": Sentiment.NEUTRAL,
    "Can we discuss this tomorrow?": Sentiment.NEUTRAL,
    "Could you repeat that, please?": Sentiment.NEUTRAL,
    "Do you mind if I ask something?": Sentiment.NEUTRAL,
    "Everything is ready to go.": Sentiment.POSITIVE,
    "Honestly, I expected more.": Sentiment.NEGATIVE,
    "I am not sure about this plan.": Sentiment.NEGATIVE,
    "I am so glad you called.": Sentiment.POSITIVE,
    "I can't believe this happened.": Sentiment.NEUTRAL,
    "I didn't expect that.": Sentiment.NEUTRAL,
    "I finished everything on the list.": Sentiment.NEUTRAL,
    "I have no idea what you mean.": Sentiment.NEGATIVE,
    "I really appreciate your time.": Sentiment.POSITIVE,
    "I told you this would happen.": Sentiment.NEGATIVE,
    "I will be there in ten minutes.": Sentiment.NEUTRAL,
    "I would rather not talk about it.": Sentiment.NEGATIVE,
    "It is just another normal day.": Sentiment.NEUTRAL,
    "It looks like the meeting was moved.": Sentiment.NEUTRAL,
    "Let me know when you are ready.": Sentiment.NEUTRAL,
    "Let us try a different approach.": Sentiment.NEUTRAL,
    "Nothing ever goes as planned.": Sentiment.NEGATIVE,
    "Okay, I'll do it.": Sentiment.NEUTRAL,
    "Please send me the file again.": Sentiment.NEUTRAL,
    "Thanks for helping me.": Sentiment.POSITIVE,
    "That explains a lot, actually.": Sentiment.NEUTRAL,
    "That is completely unacceptable.": Sentiment.NEGATIVE,
    "That worked out better than I hoped.": Sentiment.POSITIVE,
    "That's exactly what I wanted.": Sentiment.POSITIVE,
    "This has been a long week.": Sentiment.NEGATIVE,
    "This is going to take a while.": Sentiment.NEUTRAL,
    "This is my favorite part.": Sentiment.POSITIVE,
    "We need to talk about this.": Sentiment.NEUTRAL,
    "Whatever, it's fine.": Sentiment.NEUTRAL,
    "Why did nobody tell me?": Sentiment.NEGATIVE,
    "You did a great job today.": Sentiment.POSITIVE,
}


def speaker_id_for(folder: str) -> str:
    """`Cate_20260920_212919` -> the stable speaker id."""
    for prefix, sid in SPEAKER_IDS.items():
        if folder.startswith(prefix):
            return sid
    raise KeyError(f"unknown speaker folder {folder!r} -- add it to SPEAKER_IDS")


def decode_to_wav(src: Path, dst: Path) -> float:
    """WebM/Opus -> 16 kHz mono PCM WAV. Returns duration in seconds.

    Resampling happens inside the decoder rather than after it: Opus is
    48 kHz and the project works at 16 kHz, and libswresample does that
    properly. Writing the file at 16 kHz also makes `load_clip`'s own
    resample step a no-op, so a clip is never resampled twice.
    """
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    with av.open(str(src)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().ravel())
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().ravel())

    if not chunks:
        raise ValueError(f"decoded no audio from {src}")
    samples = np.concatenate(chunks).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, samples, SAMPLE_RATE, subtype="PCM_16")
    return len(samples) / SAMPLE_RATE


def main() -> None:
    if not SOURCE_DIR.exists():
        raise SystemExit(f"{SOURCE_DIR} not found")

    meta = pd.read_csv(SOURCE_DIR / "metadata.csv")
    logger.info("metadata rows: %d, participants: %d", len(meta), meta.participant_id.nunique())

    unknown = sorted(set(meta.sentence_text) - set(TEXT_SENTIMENT))
    if unknown:
        # Failing loudly beats defaulting to NEUTRAL: an unlabelled sentence
        # silently becomes "congruent" and dilutes PSI.
        raise SystemExit(
            f"{len(unknown)} sentence(s) have no hand-assigned text valence: {unknown}"
        )

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    durations: list[float] = []

    for r in meta.itertuples(index=False):
        src = SOURCE_DIR / r.audio_path
        folder = Path(r.audio_path).parts[2]
        if not src.exists():
            missing.append(r.audio_path)
            continue

        speaker = speaker_id_for(folder)
        clip_id = f"e6_{speaker}_{r.recording_id[:8]}"
        dst = OUT_DIR / f"{clip_id}.wav"
        durations.append(decode_to_wav(src, dst))

        prosody = Sentiment(r.sentiment.lower())
        text = TEXT_SENTIMENT[r.sentence_text]
        rows.append(
            {
                "clip_id": clip_id,
                "path": str(dst.relative_to(REPO_ROOT)),
                "speaker_id": speaker,
                "source": "recorded",
                "text": r.sentence_text,
                "text_sentiment": text.value,
                "prosody_sentiment": prosody.value,
                "emotion_tag": "",
                "voice_id": "",
                "is_congruent": text == prosody,
                "split": r.dataset_split if r.dataset_split != "validation" else "val",
            }
        )
        if len(rows) % 25 == 0:
            logger.info("  decoded %d clips", len(rows))

    df = pd.DataFrame(rows)
    write_manifest(df, MANIFEST_PATH)

    by_speaker = df.groupby(["split", "speaker_id"]).size()
    summary = {
        "n_metadata_rows": len(meta),
        "n_clips_built": len(df),
        "n_missing_files": len(missing),
        "missing_files": missing,
        "missing_note": (
            "Metadata rows whose audio file could not be found. Currently zero -- every row "
            "resolves, across both the .webm and .m4a sessions. Reported rather than dropped "
            "silently (CLAUDE.md: never silently drop clips) because a glob that assumes one "
            "container is exactly how a whole session goes missing unnoticed here."
        ),
        "n_speakers": int(df.speaker_id.nunique()),
        "speakers_by_split": {f"{s}/{sp}": int(n) for (s, sp), n in by_speaker.items()},
        "n_incongruent": int((~df.is_congruent).sum()),
        "incongruent_share": float((~df.is_congruent).mean()),
        "class_balance": df.prosody_sentiment.value_counts().to_dict(),
        "duration_seconds": {
            "total": float(sum(durations)),
            "mean": float(np.mean(durations)),
            "min": float(np.min(durations)),
            "max": float(np.max(durations)),
        },
        "shared_speaker_note": (
            f"The 'marco' folder is recorded by the same person as E3's speaker1, so it "
            f"carries the id {E3_SPEAKER_ID!r}. Any combination of E6-train and an E3 "
            f"evaluation therefore trips assert_speaker_disjoint rather than passing "
            f"quietly with the same voice on both sides."
        ),
        "text_sentiment_note": (
            "text_sentiment is assigned by hand from the sentence text (see TEXT_SENTIMENT "
            "in scripts/build_zurich.py); the dataset itself labels intended DELIVERY only. "
            "Ambiguous wordings are left NEUTRAL rather than forced to a side, so PSI here "
            "measures the model and not the labelling. It remains the analyst's judgment "
            "and is the one derived column in this manifest."
        ),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    logger.info("built %d clips from %d metadata rows", len(df), len(meta))
    if missing:
        logger.warning("%d metadata rows had no audio file -- see %s", len(missing), SUMMARY_PATH)
    logger.info(
        "speakers: %d, incongruent: %d/%d",
        df.speaker_id.nunique(),
        summary["n_incongruent"],
        len(df),
    )
    logger.info("%s", by_speaker.to_string())
    logger.info("wrote %s and %s", MANIFEST_PATH, SUMMARY_PATH)


if __name__ == "__main__":
    main()
