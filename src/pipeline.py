"""
IndicPulse: Unified Multi-Module Text-Understanding Pipeline
Combines Language Identification, Emotion Recognition, and Intent Classification
for English, Hinglish, and Punglish user messages.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from google import genai


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
LANGUAGE_MODEL_PATH = MODELS_DIR / "language_tfidf_logreg.joblib"
EMOTION_MODEL_PATH = MODELS_DIR / "emotion_tfidf_logreg.joblib"
INTENT_MODEL_PATH = MODELS_DIR / "intent-xlmr-final"


# Affect lexicon for code-mixed Romanized Hinglish, Punglish, and English
EMOTION_LEXICON = {
    "Anger": [
        "gussa", "ghussa", "gusse", "krodh", "angry", "anger", "furious",
        "rage", "bakwaas", "bakwas", "pissed", "hate", "dimaag kharab",
        "dimag kharab", "khun khol", "laanat", "lanat", "pagal hai kya"
    ],
    "Annoyance": [
        "annoyed", "irritated", "tang", "pareshan", "pareshaan", "kharab",
        "bekaar", "bekar", "frustrat", "slow", "bore", "issue", "problem",
        "stuck", "error", "failed", "hang", "bekaar service", "bakwas service"
    ],
    "Gratitude": [
        "thanks", "thank you", "thank", "shukriya", "dhanyawad", "dhanyavad",
        "dhanwaad", "dhanvaad", "appreciate", "meharbani", "mehrbani", "grateful"
    ],
    "Admiration": [
        "awesome", "amazing", "great", "superb", "badhiya", "vadhiya", "vadiya",
        "ghaint", "sira", "kamaal", "kamal", "best", "excellent", "fantastic",
        "fabulous", "mast", "zabardast"
    ],
    "Amusement": [
        "haha", "hehe", "lol", "rofl", "lmao", "funny", "mazak", "mazaak",
        "hasna", "joke", "chutkula"
    ],
    "Love": [
        "love", "pyar", "pyaar", "dil", "favourite", "favorite", "mohabbat",
        "jaan", "pyara", "pyaari"
    ],
    "Approval": [
        "agree", "approved", "theek hai", "thik hai", "sahi", "correct",
        "good", "nice", "perfect", "done", "okay", "alright"
    ],
}

DEFAULT_INTENT_LABELS = ["Complaint", "Opinion", "Other", "Question", "Request"]
DEFAULT_THRESHOLDS = {
    "Complaint": 0.24,
    "Question": 0.24,
    "Request": 0.30,
    "Opinion": 0.25,
    "Other": 0.27,
}


class LanguageModule:
    """Module for identifying English, Hinglish, or Punglish text."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or LANGUAGE_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Language model not found at {self.model_path}. "
                "Run train_language_model() first."
            )
        self.model = joblib.load(self.model_path)

    def predict(self, text: str) -> str:
        """Predict language using the trained classifier."""
        text_str = str(text).strip()
        if not text_str:
            return "English"
        # Direct prediction returns the class label
        return self.model.predict([text_str])[0]


    def predict_proba(self, text: str) -> Dict[str, float]:
        text_str = str(text).strip()
        if not text_str:
            return {"English": 1.0, "Hinglish": 0.0, "Punglish": 0.0}
        probs = self.model.predict_proba([text_str])[0]
        classes = self.model.named_steps["classifier"].classes_
        return {str(cls): round(float(p), 4) for cls, p in zip(classes, probs)}


class EmotionModule:
    """
    Module for recognizing emotion across English, Hinglish, Punglish, and Devanagari Hindi.
    Classes: Anger, Annoyance, Gratitude, Admiration, Amusement, Love, Approval, Neutral.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or EMOTION_MODEL_PATH
        self.devanagari_model = None
        if self.model_path.exists():
            self.devanagari_model = joblib.load(self.model_path)

    def predict(self, text: str) -> tuple:
        """Return a tuple (emotion, source).
        source values:
          - "hindi-model" if Devanagari‑script model was used
          - "lexicon" if a keyword from EMOTION_LEXICON matched
          - "none" when no cue was found (Neutral)
        """
        text_str = str(text).strip()
        if not text_str:
            return ("Neutral", "none")

        # If text is in Devanagari script, use the trained ML baseline
        if re.search(r"[\u0900-\u097F]", text_str) and self.devanagari_model is not None:
            pred = self.devanagari_model.predict([text_str])[0]
            return (str(pred).capitalize(), "hindi-model")

        # For Romanized text (English, Hinglish, Punglish), use affect detection
        text_lower = text_str.lower()
        for emotion, keywords in EMOTION_LEXICON.items():
            for kw in keywords:
                # Full word‑boundary match to avoid false‑positive prefixes
                if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                    # Return emotion and indicate it came from lexicon
                    return (emotion, "lexicon")

        # No lexicon match found
        return ("Neutral", "none")


class IntentModule:
    """
    Module for multi-label intent classification:
    Labels: Question, Request, Complaint, Opinion, Other.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.model_path = model_path or INTENT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(f"Intent model not found at {self.model_path}.")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.eval()
        self.labels = DEFAULT_INTENT_LABELS
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        # Initialize Gemini client configuration
        self._gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self._gemini_api_key:
            self._gemini_client = genai.Client(api_key=self._gemini_api_key)
        else:
            raise EnvironmentError("GEMINI_API_KEY not set in environment")

    def predict_proba(self, text: str) -> Dict[str, Any]:
        text_str = str(text).strip()
        if not text_str:
            return {"probabilities": {l: 0.0 for l in self.labels}, "intents": ["Other"]}

        inputs = self.tokenizer(
            text_str,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.sigmoid(logits)[0].cpu().numpy()
        scores = {label: round(float(p), 4) for label, p in zip(self.labels, probabilities)}

        selected = [
            label
            for label, prob in scores.items()
            if prob >= self.thresholds.get(label, 0.25)
        ]
        # Retain all intents that exceed their respective thresholds.
        # If none exceed, fall back to the single highest‑probability label.
        if not selected:
            top_label = max(scores.items(), key=lambda x: x[1])[0]
            selected = [top_label]

        return {"probabilities": scores, "intents": sorted(selected)}

    def predict(self, text: str) -> List[str]:
        return self.predict_proba(text)["intents"]


class IndicPulsePipeline:
    """
    Unified multi-module pipeline orchestrating:
    1. Language Identification (English / Hinglish / Punglish)
    2. Emotion Recognition (Anger, Annoyance, Gratitude, Admiration, etc.)
    3. Multi-label Intent Classification (Question, Request, Complaint, Opinion, Other)
    """

    def __init__(
        self,
        language_model_path: Optional[Path] = None,
        emotion_model_path: Optional[Path] = None,
        intent_model_path: Optional[Path] = None,
        intent_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.language_module = LanguageModule(language_model_path)
        self.emotion_module = EmotionModule(emotion_model_path)
        self.intent_module = IntentModule(intent_model_path, intent_thresholds)

    def predict(self, message: str, detailed: bool = False) -> Dict[str, Any]:
        """Process a single input message and return combined understanding output."""
        import time
        start_time = time.perf_counter()

        # Handle completely empty input early
        if not message or not str(message).strip():
            return {
                "message": message,
                "language": None,
                "emotion": "Neutral",
                "intent": [],
                "source": {"emotion": "none"},
                "details": {"no_input": True},
            }

        language = self.language_module.predict(message)
        # EmotionModule now returns (emotion, source)
        emotion, emotion_source = self.emotion_module.predict(message)
        intent_data = self.intent_module.predict_proba(message)
        intents = intent_data["intents"]
        # Determine confidence
        max_intent_conf = max(intent_data["probabilities"].values()) if intent_data["probabilities"] else 0.0
        fallback_needed = max_intent_conf < 0.3 or emotion_source == "none"
        if fallback_needed:
            try:
                gemini_resp = self._call_gemini_fallback(message)
                if gemini_resp:
                    emotion = gemini_resp.get("emotion", emotion)
                    emotion_source = "gemini"
                    intents = gemini_resp.get("intent", intents)
            except Exception:
                pass
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

        result = {
            "message": message,
            "language": language,
            "emotion": emotion,
            "intent": intents,
            "source": {"emotion": emotion_source},
        }

        if detailed:
            lang_scores = self.language_module.predict_proba(message)
            result["details"] = {
                "language_scores": lang_scores,
                "intent_scores": intent_data["probabilities"],
                "emotion_source": emotion_source,
                "latency_ms": latency_ms,
                "no_cue": emotion_source == "none",
            }

        return result

    def predict_batch(self, messages: List[str], detailed: bool = False) -> List[Dict[str, Any]]:
        """Process multiple input messages sequentially."""
        return [self.predict(msg, detailed=detailed) for msg in messages]

    def _call_gemini_fallback(self, text: str) -> Dict[str, Any]:
        """Call Gemini model for low‑confidence cases.
        Returns a dict with keys 'emotion' and 'intent'.
        """
        prompt = (
            "You are a helpful assistant that classifies the following user message. "
            "Return a JSON object with two fields: `emotion` (one of the supported emotions) "
            "and `intent` (a list of intent labels from [Question, Request, Complaint, Opinion, Other]). "
            f"Message: {text}"
        )
        response = self._gemini_client.models.generate_content(
            model=self._gemini_model_name,
            contents=prompt,
        )
        try:
            parsed = json.loads(response.text)
            return {"emotion": parsed.get("emotion", "Neutral"), "intent": parsed.get("intent", [])}
        except Exception:
            return {}

