# IndicPulse

IndicPulse is an NLP research project for identifying English, Hinglish, and Punglish text, with planned modules for emotion recognition and intent classification.

## Current Progress

### Emotion Classification

- Source dataset: EmoHinD-58K
- Source task: Multi-label, fine-grained Hindi emotion recognition
- Baseline subset: 8 single-label emotion classes
- Classes: admiration, amusement, anger, annoyance, approval, gratitude, love, neutral
- Model: Character-level TF-IDF + Logistic Regression with balanced class weights
- Validation accuracy: 64.69%
- Validation macro F1: 0.6093
- Note: Multi-label examples and rare-label examples were excluded from this first baseline.

### Language Identification

- Classes: English, Hinglish, Punglish
- Dataset: AI4Bharat IndicCMix
- Model: Character-level TF-IDF + Logistic Regression
- Validation accuracy: 99.13%
- Validation macro F1: 0.9913

### Intent Classification (Multi-Label)

- Classes: `Question`, `Request`, `Complaint`, `Opinion`, `Other` (multi-label delimiter: `|`)
- Candidate dataset: `data/processed/intent_training_candidates.csv` contains 978 reviewed rows from the prior set and the v3 addition. Seven reviewed ambiguous examples have empty intent lists and are included as all-zero multi-label examples.
- Model: Fine-tuned `xlm-roberta-base` with multi-label binary cross-entropy loss
- Current data split: 684 train / 147 validation / 147 test. The v3 model was fine-tuned for 3 epochs; the best validation loss was 0.3186.
- With per-label thresholds selected on validation, the held-out test score is 0.7652 micro F1 and 0.7658 macro F1. Per-label test F1: Complaint 0.6824, Opinion 0.6667, Other 0.7692, Question 0.9231, Request 0.7879. These results estimate performance on this split and do not guarantee accuracy on future user messages.
- Validation-selected thresholds: Complaint 0.34, Opinion 0.27, Other 0.27, Question 0.29, Request 0.32.

## Unified Multi-Module Pipeline

IndicPulse combines Language Identification, Emotion Recognition, and Intent Classification into a single unified interface:

```python
from src.pipeline import IndicPulsePipeline

pipeline = IndicPulsePipeline()
result = pipeline.predict("Mera refund abhi tak nahi aaya, bahut gussa hai.")
print(result)
```

Output:
```json
{
  "message": "Mera refund abhi tak nahi aaya, bahut gussa hai.",
  "language": "Hinglish",
  "emotion": "Anger",
  "intent": ["Complaint", "Question"]
}
```

### CLI Usage

Analyze a single message:
```bash
python scripts/run_pipeline.py --message "Mera refund abhi tak nahi aaya, bahut gussa hai."
```

Interactive mode:
```bash
python scripts/run_pipeline.py --interactive
```

Run demo across English, Hinglish, and Punglish:
```bash
python scripts/run_pipeline.py --demo
```

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Web app

The recommended server is the FastAPI app:

```powershell
.\run_api.bat
```

Set `GEMINI_API_KEY` in a local `.env` file to enable Gemini fallback. Without a key, the local classifiers still run; if a fallback is needed, the interface reports that Gemini is unavailable. Set `GEMINI_MODEL` only if you want to override the default `gemini-3.8-flash`.

Each result reports its source. Low-confidence or closely tied language scores, strong Romanized Punjabi cues, intent scores near a decision threshold, and missing local emotion cues trigger one Gemini fallback request. An exact match keeps the local result and its source attribution; a disagreement replaces only a field marked uncertain. Missing local emotion cues are supplied by Gemini. Intent bars show scores from the same source as the final intent labels: local XLM-R scores for local labels or Gemini confidence estimates when Gemini supplies the intent. Neither is a calibrated probability. Detailed output includes the active local intent thresholds to help identify a server that has not been restarted after a model update. Gemini output is schema-constrained and checked against supported labels; invalid output or an API failure leaves local predictions in place. This validation prevents malformed labels but cannot guarantee semantic correctness.

## Dataset Attribution

- IndicCMix: AI4Bharat, via Hugging Face
- MASSIVE: Amazon Alexa / Amazon Science, CC BY 4.0
- EmoInHindi: Singh et al., LREC 2022; restricted to research use and not redistributed in this repository
