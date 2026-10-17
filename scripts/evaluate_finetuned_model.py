import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score

MODEL_DIR = "models/intent-xlmr-final"
LABELS = ["Complaint", "Opinion", "Other", "Question", "Request"]

DEFAULT_THRESHOLDS = {
    "Complaint": 0.34,
    "Question": 0.29,
    "Request": 0.32,
    "Opinion": 0.27,
    "Other": 0.27,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned XLM-RoBERTa model with calibrated thresholds."
    )
    parser.add_argument(
        "--test-file",
        default="data/processed/intent_test.csv",
        help="Path to evaluation CSV",
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="Path to a fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Global prediction threshold (overrides per-class tuned thresholds)",
    )
    parser.add_argument(
        "--tune-on-validation",
        action="store_true",
        help="Automatically tune per-class thresholds on intent_validation.csv",
    )
    return parser.parse_args()


def to_vector(labels):
    vector = [0] * len(LABELS)
    for label in labels.split("|"):
        if label.strip() in LABELS:
            vector[LABELS.index(label.strip())] = 1
    return vector


def main():
    args = parse_args()
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    thresholds = dict(DEFAULT_THRESHOLDS)

    if args.tune_on_validation:
        print("Tuning per-label thresholds on validation split...")
        val_df = pd.read_csv("data/processed/intent_validation.csv", dtype=str).fillna("")
        val_probs = []
        val_true = []
        for _, row in val_df.iterrows():
            inp = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            with torch.no_grad():
                lg = model(**inp).logits
            val_probs.append(torch.sigmoid(lg)[0].cpu().numpy())
            val_true.append(to_vector(row["intent"]))

        val_probs = np.array(val_probs)
        val_true = np.array(val_true)

        for j, label in enumerate(LABELS):
            best_f1 = -1
            best_t = 0.25
            for t in np.linspace(0.10, 0.40, 31):
                f = f1_score(val_true[:, j], (val_probs[:, j] >= t).astype(int), zero_division=0)
                if f > best_f1:
                    best_f1 = f
                    best_t = round(float(t), 3)
            thresholds[label] = best_t
            print(f"  {label:10s}: threshold = {best_t:.3f} (Val F1 = {best_f1:.4f})")

    elif args.threshold is not None:
        thresholds = {label: args.threshold for label in LABELS}
        print(f"Using fixed threshold: {args.threshold}")
    else:
        print("Using validation-calibrated thresholds:")
        for l, t in thresholds.items():
            print(f"  {l:10s}: {t:.2f}")

    print(f"\nEvaluating on {args.test_file}...")
    df = pd.read_csv(args.test_file, dtype=str).fillna("")

    y_true = []
    y_pred = []

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

        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

        # Apply thresholds per label
        pred_vector = [
            1 if probabilities[j] >= thresholds[LABELS[j]] else 0
            for j in range(len(LABELS))
        ]

        # Argmax fallback if no threshold is met
        if sum(pred_vector) == 0:
            pred_vector[int(probabilities.argmax())] = 1

        y_true.append(to_vector(row["intent"]))
        y_pred.append(pred_vector)

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=LABELS,
            zero_division=0,
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
