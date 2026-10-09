import pandas as pd
import re
from collections import Counter

LABEL_NAMES = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}

# Load dataset
train = pd.read_csv("data/raw/emohind/emoHi-train.csv")

# Robust parser for labels
def parse_labels(val):
    if pd.isna(val):
        return []
    val = str(val).strip("[]").strip()
    parts = re.split(r"[,\s]+", val)  # split on commas or whitespace
    return [int(p) for p in parts if p.isdigit()]

train["label_list"] = train["labels"].apply(parse_labels)

# Count label frequencies
label_counts = Counter(
    label
    for labels in train["label_list"]
    for label in labels
)

print("Emotion-label counts in training data:\n")

# Print counts sorted by frequency
for label_id, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{label_id:2d} | {LABEL_NAMES[label_id]:15s} | {count:5d}")

# Show examples with multiple labels
print("\nExamples with multiple emotion labels:")
multi_label_rows = train[train["label_list"].apply(len) > 1]

if len(multi_label_rows) == 0:
    print("No multi-label rows found in this split.")
else:
    for _, row in multi_label_rows.head(5).iterrows():
        names = [LABEL_NAMES[label] for label in row["label_list"]]
        print("\nText:", row["text"])
        print("IDs:", row["label_list"])
        print("Emotions:", names)
