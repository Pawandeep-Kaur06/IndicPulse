"""
Threshold Tuning Script for Fine-Tuned Intent Classification Model.
Sweeps sigmoid cutoff thresholds on the validation set to maximize Micro/Macro F1.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/intent-xlmr-final"
VALIDATION_FILE = "data/processed/intent_validation.csv"
LABELS = ["Complaint", "Opinion", "Other", "Question", "Request"]


def to_vector(labels):
    vector = [0] * len(LABELS)
    for label in str(labels).split("|"):
        if label.strip() in LABELS:
            vector[LABELS.index(label.strip())] = 1
    return vector


def main():
    parser = argparse.ArgumentParser(description="Find optimal classification thresholds on validation data.")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--val-file", default=VALIDATION_FILE, help="Path to validation CSV")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    print(f"Loading validation data from {args.val_file}...")
    df = pd.read_csv(args.val_file, dtype=str).fillna("")

    y_true = []
    probabilities = []

    for _, row in df.iterrows():
        inputs = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        probabilities.append(probs)
        y_true.append(to_vector(row["intent"]))

    probabilities = np.array(probabilities)
    y_true = np.array(y_true)

    print("\n" + "=" * 60)
    print("GLOBAL THRESHOLD SWEEP (0.10 to 0.45)")
    print("=" * 60)
    best_global_th = 0.25
    best_global_f1 = -1

    for t in np.linspace(0.10, 0.45, 36):
        preds = (probabilities >= t).astype(int)
        # Fallback to argmax if no label active
        for i in range(len(preds)):
            if preds[i].sum() == 0:
                preds[i, probabilities[i].argmax()] = 1
        micro = f1_score(y_true, preds, average="micro", zero_division=0)
        macro = f1_score(y_true, preds, average="macro", zero_division=0)
        if micro > best_global_f1:
            best_global_f1 = micro
            best_global_th = t
        print(f"Threshold: {t:.2f} | Micro F1: {micro:.4f} | Macro F1: {macro:.4f}")

    print(f"\nOptimal Global Threshold: {best_global_th:.2f} (Micro F1: {best_global_f1:.4f})")

    print("\n" + "=" * 60)
    print("PER-LABEL OPTIMAL THRESHOLDS")
    print("=" * 60)
    tuned_thresholds = {}
    for j, label in enumerate(LABELS):
        best_t = 0.25
        best_f = -1
        for t in np.linspace(0.10, 0.40, 31):
            f = f1_score(y_true[:, j], (probabilities[:, j] >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f = f
                best_t = t
        tuned_thresholds[label] = round(float(best_t), 3)
        print(f"{label:12s}: best threshold = {best_t:.3f} (Val F1 = {best_f:.4f})")

    print("\nRecommended threshold configuration dict:")
    print(tuned_thresholds)


if __name__ == "__main__":
    main()
