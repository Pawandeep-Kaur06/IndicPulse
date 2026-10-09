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

## Dataset

The language-identification dataset was created from AI4Bharat IndicCMix:

- English: `english`
- Hinglish: `romanized_casual` from the Hindi subset
- Punglish: `romanized_casual` from the Punjabi subset

The local processed dataset contains 14,991 balanced samples:

- Hinglish: 4,999
- English: 4,999
- Punglish: 4,993

Dataset files are excluded from the repository. Download them separately from the original source and run the preparation scripts.

## Planned Modules

1. Language identification — completed baseline
2. Intent classification — MASSIVE Hindi and English
3. Emotion recognition — EmoInHindi and a verified Hinglish resource
4. External real-world evaluation — manually written, consented English/Hinglish/Punglish samples

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