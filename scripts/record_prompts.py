#!/usr/bin/env python3
"""E3: teleprompter for recording the human incongruence set.

Same crossed design as E2 (scripts/gen_synthetic.py) -- text sentiment vs
delivered tone -- but with a real human voice as the validity anchor for
E2's synthetic clips. Ground-truth `prosody_sentiment` is what the SPEAKER
was asked to convey, same principle as E2's TTS-intent labels: we trust the
instruction given, not a re-classification by any model.

This script is meant to be run on a machine with a microphone -- it's not
exercised end-to-end in the test suite (no audio hardware in CI/sandboxes).
The prompt-list construction (`build_prompts`) is pure and fully tested;
only the interactive recording loop (`main`) needs a real device.

Usage:
    uv run python scripts/record_prompts.py                  # primary speaker, full set
    uv run python scripts/record_prompts.py --speaker-id spk2 --reduced  # second speaker
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.gen_synthetic import CARRIERS  # noqa: E402 -- reuse E2's carrier sentences
from ssa.manifest import add_congruence, write_manifest  # noqa: E402
from ssa.types import SAMPLE_RATE, Sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "data" / "recorded"

# Target tone the speaker should deliver in, spoken aloud as an instruction
# before each recording -- kept in plain English, not a jargon label.
TONE_INSTRUCTION: dict[Sentiment, str] = {
    Sentiment.POSITIVE: "cheerful / upbeat",
    Sentiment.NEUTRAL: "flat / matter-of-fact",
    Sentiment.NEGATIVE: "annoyed / upset",
}

# A few sentences read in a deliberate whisper or with exaggerated
# breathiness, for T11's paralinguistic demo -- distinct from the
# sentiment-prosody design above.
WHISPER_PROMPTS: list[str] = [
    "Can you hear me from over here?",
    "I don't want to wake anyone up.",
    "This is just between the two of us.",
]


@dataclass(frozen=True, slots=True)
class PromptSpec:
    clip_id: str
    text: str
    text_sentiment: Sentiment
    prosody_sentiment: Sentiment | None  # None for whisper prompts (not part of the PSI design)
    instruction: str
    is_whisper_prompt: bool = False


def build_prompts(speaker_id: str, *, reduced: bool = False) -> list[PromptSpec]:
    """Full: 3 text-sentiments x 3 prosody-targets x 3 carriers = 27 clips
    + 3 whisper prompts = 30. Reduced (second speaker): 3 x 2 x 2 = 12
    clips, no whisper prompts -- keeps a second recording session short.
    Together, ~42 clips across two speakers, matching docs/TASKS.md's ~40
    target for E3."""
    prosody_targets = list(Sentiment) if not reduced else [Sentiment.POSITIVE, Sentiment.NEGATIVE]
    carriers_per_sentiment = 2 if reduced else 3

    prompts: list[PromptSpec] = []
    for text_sentiment, sentences in CARRIERS.items():
        for carrier_idx, text in enumerate(sentences[:carriers_per_sentiment]):
            for prosody_sentiment in prosody_targets:
                clip_id = (
                    f"e3_{speaker_id}_{text_sentiment.value}_{carrier_idx}_"
                    f"{prosody_sentiment.value}"
                )
                prompts.append(
                    PromptSpec(
                        clip_id=clip_id,
                        text=text,
                        text_sentiment=text_sentiment,
                        prosody_sentiment=prosody_sentiment,
                        instruction=TONE_INSTRUCTION[prosody_sentiment],
                    )
                )

    if not reduced:
        for i, text in enumerate(WHISPER_PROMPTS):
            prompts.append(
                PromptSpec(
                    clip_id=f"e3_{speaker_id}_whisper_{i}",
                    text=text,
                    text_sentiment=Sentiment.NEUTRAL,
                    prosody_sentiment=None,
                    instruction="whispered, as quietly as you'd still speak to be heard",
                    is_whisper_prompt=True,
                )
            )
    return prompts


def prompts_to_manifest(prompts: list[PromptSpec], speaker_id: str) -> pd.DataFrame:
    """Whisper prompts get prosody_sentiment=text_sentiment (congruent by
    convention -- they're not part of the incongruence design; T11 reads
    them by is_whisper_prompt-equivalent naming (clip_id contains
    'whisper'), not by manifest schema, since the schema has no such column."""
    rows = []
    for p in prompts:
        prosody = p.prosody_sentiment if p.prosody_sentiment is not None else p.text_sentiment
        rows.append(
            {
                "clip_id": p.clip_id,
                "path": str((OUT_DIR / f"{p.clip_id}.wav").relative_to(REPO_ROOT)),
                "speaker_id": speaker_id,
                "source": "recorded",
                "text": p.text,
                "text_sentiment": p.text_sentiment.value,
                "prosody_sentiment": prosody.value,
                "emotion_tag": "",
                "voice_id": "",
                "split": "",
            }
        )
    df = pd.DataFrame(rows)
    return add_congruence(df)


def _record_one(duration_cap_s: float = 15.0) -> np.ndarray:
    """Record from the default input device until Enter is pressed again,
    capped at duration_cap_s. Requires `sounddevice` to find a real input
    device -- raises a clear, actionable error otherwise."""
    import sounddevice as sd

    input("  Press Enter to START recording...")
    frames: list[np.ndarray] = []

    def callback(indata, _frame_count, _time_info, _status):
        frames.append(indata.copy())

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
            input("  Recording... press Enter to STOP.")
    except sd.PortAudioError as exc:
        raise RuntimeError(
            "No audio input device found. This script must be run on a machine with a "
            "microphone (e.g. your laptop), not in a headless environment."
        ) from exc

    if not frames:
        raise RuntimeError("No audio captured -- check your microphone and try again.")
    audio = np.concatenate(frames, axis=0).reshape(-1)
    max_samples = int(duration_cap_s * SAMPLE_RATE)
    return audio[:max_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-id", default="speaker1")
    parser.add_argument("--reduced", action="store_true", help="shorter set for a second speaker")
    args = parser.parse_args()

    prompts = build_prompts(args.speaker_id, reduced=args.reduced)
    manifest = prompts_to_manifest(prompts, args.speaker_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{len(prompts)} prompts for speaker '{args.speaker_id}'. Ctrl+C to stop early.\n")
    for i, p in enumerate(prompts, start=1):
        out_path = OUT_DIR / f"{p.clip_id}.wav"
        if out_path.exists():
            logger.info("[%d/%d] already recorded: %s (skipping)", i, len(prompts), p.clip_id)
            continue

        print(f"[{i}/{len(prompts)}] Say this in a {p.instruction} tone:")
        print(f'    "{p.text}"')
        audio = _record_one()
        sf.write(out_path, audio, SAMPLE_RATE)
        print(f"  saved ({len(audio) / SAMPLE_RATE:.1f}s)\n")

    manifest_path = OUT_DIR / f"manifest_{args.speaker_id}.csv"
    write_manifest(manifest, manifest_path)
    logger.info("done: %d prompts -> %s", len(prompts), manifest_path)


if __name__ == "__main__":
    main()
