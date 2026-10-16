import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, multilabel_confusion_matrix

VALID_INTENTS = ["Complaint", "Opinion", "Other", "Question", "Request"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-label intent prediction CSV against gold annotations."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions CSV (columns: id/message_id, intent)",
    )
    parser.add_argument(
        "--gold",
        default="data/processed/intent_synthetic_candidates.csv",
        help="Path to gold dataset CSV (default: intent_synthetic_candidates.csv)",
    )
    return parser.parse_args()


def get_id_column(df):
    if "message_id" in df.columns:
        return "message_id"
    if "id" in df.columns:
        return "id"
    raise KeyError(f"Neither 'id' nor 'message_id' found in columns: {list(df.columns)}")


def to_vector(labels):
    labels_set = {label.strip() for label in str(labels).split("|") if label.strip()}
    return [1 if intent in labels_set else 0 for intent in VALID_INTENTS]


def main():
    args = parse_args()
    gold = pd.read_csv(args.gold, dtype=str, encoding="utf-8-sig").fillna("")
    preds = pd.read_csv(args.predictions, dtype=str, encoding="utf-8-sig").fillna("")

    gold_id_col = get_id_column(gold)
    pred_id_col = get_id_column(preds)

    assert list(gold[gold_id_col]) == list(preds[pred_id_col]), (
        f"Prediction IDs ({pred_id_col}) do not match gold dataset IDs ({gold_id_col})"
    )

    y_true = [to_vector(labels) for labels in gold["intent"]]
    y_pred = [to_vector(labels) for labels in preds["intent"]]

    print("=" * 60)
    print(f"EVALUATING PREDICTIONS: {args.predictions}")
    print(f"GOLD REFERENCE:         {args.gold}")
    print("=" * 60)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=VALID_INTENTS,
            zero_division=0,
            digits=4,
        )
    )

    print("\nPer-intent confusion matrices:")
    print("Columns: [TN, FP], [FN, TP]")
    for intent, matrix in zip(VALID_INTENTS, multilabel_confusion_matrix(y_true, y_pred)):
        print(f"\n{intent}\n{matrix}")


if __name__ == "__main__":
    main()