"""Cartesia TTS wrapper: the synthetic-speech generation layer for D0/E2/E4.

Generation is API-gated and optional -- per CLAUDE.md rule 5, evaluation
runs entirely from committed fixtures (data/synthetic/*.wav + manifest.csv,
data/cache/d0_probe/*.wav). This module is imported only by the gen_*
scripts, never by anything in ssa/solutions or ssa/eval, so evaluation
never depends on network access or a key.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from cartesia import Cartesia

logger = logging.getLogger(__name__)

MODEL_ID = "sonic-3"
SAMPLE_RATE = 16000

# Two fixed English voices, chosen for reproducibility. Acknowledged
# limitation (see docs/ARCHITECTURE.md's open risks): 2 voices is thin
# diversity for the synthetic sets, stated rather than over-claimed.
VOICE_IDS: dict[str, str] = {
    "skylar": "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
    "daniel": "47c38ca4-5f35-497b-b1a3-415245fb35e1",
}


def get_client() -> Cartesia:
    """Raises RuntimeError with a clear, actionable message if no key is
    configured -- this is the one place that should ever fail on a missing
    key; nothing downstream should let a network error surface confusingly."""
    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CARTESIA_API_KEY not set. Generation scripts are optional and "
            "key-gated -- evaluation runs from committed fixtures without one. "
            "Copy .env.example to .env and fill in a key to regenerate."
        )
    return Cartesia(api_key=api_key)


def generate_clip(
    client: Cartesia, *, text: str, emotion: str, voice_id: str, out_path: Path
) -> None:
    """Generate one clip and save it as 16kHz mono PCM16 WAV."""
    resp = client.tts.generate(
        model_id=MODEL_ID,
        transcript=text,
        voice={"mode": "id", "id": voice_id},
        output_format={"container": "wav", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
        generation_config={"emotion": emotion},
    )
    audio_bytes = resp.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
