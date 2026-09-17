"""Solution A: lexical-only. ASR -> text sentiment -> Prediction.

This is deliberately the "reads the transcript, not the tone" solution --
the control condition. On congruent data it should look strong; on the
incongruent sets (E2/E3) it is *expected* to score near-zero PSI, because
by construction it can only ever see the words. That is not a bug to fix,
it is the measurement working.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from faster_whisper import WhisperModel
from transformers import pipeline

from ssa.types import AudioClip, Prediction, Sentiment

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = "small"
TEXT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Confirmed by direct inspection: this model's labels are the strings
# "positive" / "neutral" / "negative", matching Sentiment.value exactly --
# no LABEL_0/1/2 remapping needed. Still validated defensively in
# _probs_from_raw so a model swap fails loudly instead of mis-mapping.


class LexicalSolution:
    """faster-whisper (small, int8, CPU) + a 3-class text sentiment model."""

    def __init__(
        self,
        *,
        whisper_model_size: str = WHISPER_MODEL_SIZE,
        text_model_id: str = TEXT_MODEL_ID,
    ) -> None:
        # CTranslate2 (faster-whisper's backend) has NO Apple MPS support --
        # CPU is the only sane default here, on this dev box and on the
        # project's stated Mac. int8 quantisation keeps it fast enough to
        # sit beside a real-time pipeline (see CLAUDE.md pinned facts).
        self._asr = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
        self._text_clf = pipeline("text-classification", model=text_model_id, top_k=None)
        self.name = f"A:lexical({whisper_model_size}+{text_model_id.split('/')[-1]})"
        logger.info("LexicalSolution ready: %s", self.name)

    def predict(self, clip: AudioClip) -> Prediction:
        t0 = time.perf_counter()
        transcript = self._transcribe(clip)

        if not transcript.strip():
            latency_ms = (time.perf_counter() - t0) * 1000
            return _abstain(self.name, latency_ms, transcript=transcript)

        raw = self._text_clf(transcript)[0]  # list[{"label": str, "score": float}]
        probs = _probs_from_raw(raw)
        sentiment = max(probs, key=probs.get)  # type: ignore[arg-type]
        latency_ms = (time.perf_counter() - t0) * 1000

        return Prediction(
            sentiment=sentiment,
            probs=probs,
            confidence=probs[sentiment],
            latency_ms=latency_ms,
            solution=self.name,
            transcript=transcript,
        )

    def predict_batch(self, clips: Sequence[AudioClip]) -> list[Prediction]:
        return [self.predict(c) for c in clips]

    def _transcribe(self, clip: AudioClip) -> str:
        segments, _info = self._asr.transcribe(clip.samples, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()


def _probs_from_raw(raw: list[dict[str, float | str]]) -> dict[Sentiment, float]:
    """Convert the pipeline's raw label/score dicts into a Sentiment-keyed,
    normalised probability distribution. Raises if a label doesn't map onto
    Sentiment -- a model swap with a different label scheme must fail loudly,
    not silently mis-map a class."""
    try:
        probs = {Sentiment(d["label"]): float(d["score"]) for d in raw}
    except ValueError as exc:
        labels = [d["label"] for d in raw]
        raise ValueError(
            f"text classifier labels {labels} don't map onto Sentiment "
            f"{[s.value for s in Sentiment]}"
        ) from exc

    if set(probs) != set(Sentiment):
        raise ValueError(f"expected all 3 sentiments, got {set(probs)}")

    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def _abstain(solution: str, latency_ms: float, *, transcript: str) -> Prediction:
    """Uniform-probability, low-confidence prediction for an empty transcript
    (silence, or ASR failure). Sentiment defaults to NEUTRAL -- the least
    harmful guess for a companion device to fall back on -- but `abstained`
    tells the caller this prediction should not be trusted."""
    uniform = 1.0 / len(Sentiment)
    return Prediction(
        sentiment=Sentiment.NEUTRAL,
        probs={s: uniform for s in Sentiment},
        confidence=uniform,
        latency_ms=latency_ms,
        solution=solution,
        abstained=True,
        transcript=transcript,
    )
