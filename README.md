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
- Dataset: 400 manually annotated real messages (280 train / 60 validation / 60 test) across English, Hinglish, and Punglish + 86 synthetic benchmark samples
- Model: Fine-tuned `xlm-roberta-base` with multi-label binary cross-entropy loss
- Test micro F1 (calibrated threshold ~0.24-0.30): ~0.40 - 0.43 (baseline zero-shot on synthetic smoke test: 0.59)

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

## Dataset Attribution

- IndicCMix: AI4Bharat, via Hugging Face
- MASSIVE: Amazon Alexa / Amazon Science, CC BY 4.0
- EmoInHindi: Singh et al., LREC 2022; restricted to research use and not redistributed in this repository