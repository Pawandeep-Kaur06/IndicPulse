import json
from pathlib import Path

# ── Load .env FIRST so GEMINI_API_KEY is available before pipeline init ────
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Create app FIRST, then mount static files ──────────────────────────────
app = FastAPI(title="IndicPulse API", version="1.0.0")

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parent / "static"),
    name="static",
)

# ── Load the pipeline once at startup ──────────────────────────────────────
from src.pipeline import IndicPulsePipeline
pipeline = IndicPulsePipeline()


# ── Request model ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Request body for a prediction.

    The client should provide a single field ``message`` containing the raw user text.
    """
    message: str


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the static UI playground."""
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"), media_type="text/html")


@app.get("/status")
def get_status():
    """Simple health-check endpoint.

    Returns the service status and a short description of each loaded module.
    """
    return {
        "status": "online",
        "modules": {
            "language": "IndicCMix TF-IDF (≈99.7% accuracy)",
            "emotion": "EmoHinD-58K Baseline (≈64.7% accuracy)",
            "intent": "Fine-Tuned XLM-RoBERTa (calibrated thresholds)",
        },
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """Run the full IndicPulse pipeline on the supplied message.

    Parameters
    ----------
    request: PredictRequest
        JSON payload with a single ``message`` key.

    Returns
    -------
    dict
        JSON with language, emotion, intents, and source information.
    """
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    return pipeline.predict(text, detailed=True)
