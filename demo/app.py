"""The live demo: speak into a browser, see what the model hears.

Every number this project reports is an aggregate over a corpus. That is
the right way to evaluate, and the wrong way to *show* someone what the
system does. This serves one page where a reviewer records their own voice
and immediately sees the two acoustic backends score it -- including,
usually, disagreeing with each other, which is the project's central
finding made tangible rather than tabulated.

Three design decisions worth stating:

**Raw PCM, not MediaRecorder.** The browser captures Float32 samples via
the Web Audio API and posts them as an array. MediaRecorder would emit
webm/opus, which `soundfile` cannot decode and which would drag ffmpeg in
as a dependency. Raw samples go straight through `ssa.audio.clip_from_samples`
-- the same mono/resample/peak-normalise path every evaluated clip takes,
so the demo cannot flatter the model with preprocessing evaluation didn't get.

**Models load once, at startup.** A CLI invocation spends ~20s loading
models and milliseconds classifying, which makes per-clip latency look
absurd. Loading once means the latency this page reports is inference
latency, which is the number that matters for a real-time voice pipeline.

**Both backends, always.** Showing only the shipping-licensed one would
hide the measured result that it is the less reliable of the two on real
voices (r=-0.19 vs +0.92 test-retest across two takes of the same speaker).

Usage:
    uv run --extra demo python -m demo.app      # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ssa.audio import AudioLoadError, clip_from_samples, load_clip
from ssa.solutions.acoustic import AcousticSolution
from ssa.solutions.lexical import LexicalSolution
from ssa.types import SAMPLE_RATE, Prediction, Sentiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

# A browser tab can hand us an arbitrarily long buffer; cap it so one
# request cannot pin the server. 60s at 48kHz is ~2.9M floats.
MAX_SAMPLES = 48_000 * 60

# Committed E3 recording used to populate the page before anyone records:
# positive words, flat delivery, so the solutions visibly disagree.
EXAMPLE_CLIP = REPO_ROOT / "data" / "recorded" / "e3_speaker1_positive_0_neutral.wav"
EXAMPLE_TEXT = "I'm thrilled about the good news today."


class ClassifyRequest(BaseModel):
    pcm: list[float] = Field(..., description="mono Float32 samples in [-1, 1]")
    sample_rate: int = Field(..., gt=0, le=192_000)
    with_transcript: bool = Field(
        default=True, description="run ASR too (slower); the lexical solution needs it"
    )


class _Models:
    """Loaded once at startup and reused. Kept in one object so the
    request handler can state plainly whether the research backend is
    available rather than failing mid-request if its artifact is absent."""

    def __init__(self) -> None:
        self.permissive: AcousticSolution | None = None
        self.research: AcousticSolution | None = None
        self.lexical: LexicalSolution | None = None

    def load(self) -> None:
        try:
            self.permissive = AcousticSolution(backend="permissive")
            logger.info("loaded %s", self.permissive.name)
        except FileNotFoundError:
            logger.warning("permissive probe missing -- run `make train` first")
        try:
            self.research = AcousticSolution(backend="research")
            logger.info("loaded %s", self.research.name)
        except FileNotFoundError:
            logger.warning("research thresholds missing -- run `make train BACKEND=research`")
        try:
            self.lexical = LexicalSolution()
            logger.info("loaded %s", self.lexical.name)
        except Exception as exc:  # ASR model download can fail offline
            logger.warning("lexical solution unavailable (%s) -- transcript disabled", exc)
        self._warm_up()

    def _warm_up(self) -> None:
        """Run one throwaway inference per model at startup.

        Constructing a model is not the same as having run it: the first
        real call was measured at ~17s against ~200ms for every call after,
        because framework graphs and kernels initialise lazily. Without
        this, the first person to press Record waits 17 seconds and the
        latency the page shows them is wrong by two orders of magnitude.
        """
        silence = clip_from_samples(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, "warmup")
        for solution in (self.permissive, self.research, self.lexical):
            if solution is None:
                continue
            try:
                solution.predict(silence)
            except Exception as exc:
                logger.warning("warm-up failed for %s: %s", solution.name, exc)
        logger.info("warm-up done -- first real request now sees normal latency")


MODELS = _Models()


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("loading models (once) ...")
    MODELS.load()
    logger.info("ready on http://127.0.0.1:8000")
    yield


app = FastAPI(title="Speech Sentiment Analyzer — live demo", lifespan=_lifespan)


def prediction_to_dict(pred: Prediction) -> dict[str, Any]:
    """Serialise a Prediction for the browser. Probabilities keep their
    canonical negative/neutral/positive order so the bars never reorder
    between requests."""
    return {
        "solution": pred.solution,
        "sentiment": pred.sentiment.value,
        "confidence": pred.confidence,
        "probs": {s.value: pred.probs[s] for s in Sentiment.ordered()},
        "abstained": pred.abstained,
        "latency_ms": pred.latency_ms,
        "vad": (
            None
            if pred.vad is None
            else {
                "valence": pred.vad.valence,
                "arousal": pred.vad.arousal,
                "dominance": pred.vad.dominance,
            }
        ),
        "transcript": pred.transcript,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "permissive": MODELS.permissive is not None,
        "research": MODELS.research is not None,
        "lexical": MODELS.lexical is not None,
    }


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict[str, Any]:
    if not req.pcm:
        raise HTTPException(status_code=400, detail="no audio submitted")
    if len(req.pcm) > MAX_SAMPLES:
        raise HTTPException(
            status_code=413,
            detail=f"clip too long: {len(req.pcm)} samples (max {MAX_SAMPLES})",
        )

    try:
        clip = clip_from_samples(np.asarray(req.pcm, dtype=np.float32), req.sample_rate)
    except AudioLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results: dict[str, Any] = {"duration_s": clip.duration_s, "predictions": []}
    for label, solution in (
        ("permissive", MODELS.permissive),
        ("research", MODELS.research),
    ):
        if solution is None:
            continue
        pred = solution.predict(clip)
        results["predictions"].append({"backend": label, **prediction_to_dict(pred)})

    if req.with_transcript and MODELS.lexical is not None:
        pred = MODELS.lexical.predict(clip)
        results["predictions"].append({"backend": "lexical", **prediction_to_dict(pred)})

    if not results["predictions"]:
        raise HTTPException(
            status_code=503,
            detail="no models loaded -- run `make train` (and `make train BACKEND=research`)",
        )
    return results


@app.get("/example")
def example() -> dict[str, Any]:
    """Classify one committed recording so the page opens showing what it
    does, instead of an empty shell waiting for a microphone.

    The clip is deliberately chosen: the words are plainly positive and the
    delivery is flat, so the acoustic and lexical solutions disagree. That
    disagreement is the entire thesis of the project, visible before the
    reviewer has recorded anything.
    """
    if not EXAMPLE_CLIP.exists():
        raise HTTPException(status_code=404, detail="example clip not in this checkout")
    clip = load_clip(EXAMPLE_CLIP, clip_id="example")
    payload: dict[str, Any] = {
        "is_example": True,
        "text": EXAMPLE_TEXT,
        "text_sentiment": "positive",
        "prosody_sentiment": "neutral",
        "duration_s": clip.duration_s,
        "predictions": [],
    }
    for label, solution in (
        ("permissive", MODELS.permissive),
        ("research", MODELS.research),
        ("lexical", MODELS.lexical),
    ):
        if solution is None:
            continue
        payload["predictions"].append(
            {"backend": label, **prediction_to_dict(solution.predict(clip))}
        )
    return payload


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
