import pandas as pd
import re

LABEL_NAMES = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    15: "gratitude",
    18: "love",
    27: "neutral",
}

FILES = {
    "train": "data/raw/emohind/emoHi-train.csv",
    "validation": "data/raw/emohind/emoHi-valid.csv",
    "test": "data/raw/emohind/emoHi-test.csv",
}

# Robust parser for labels (handles commas or spaces)
def parse_labels(val):
    if pd.isna(val):
        return []
    val = str(val).strip("[]").strip()
    parts = re.split(r"[,\s]+", val)  # split on commas or whitespace
    return [int(p) for p in parts if p.isdigit()]

for split_name, file_path in FILES.items():
    df = pd.read_csv(file_path)

    # Parse labels safely
    df["label_list"] = df["labels"].apply(parse_labels)

    # Keep only single-label rows
    df = df[df["label_list"].apply(len) == 1].copy()
    df["label_id"] = df["label_list"].apply(lambda labels: labels[0])

    # Keep only labels we care about
    df = df[df["label_id"].isin(LABEL_NAMES)].copy()
    df["emotion"] = df["label_id"].map(LABEL_NAMES)

    # Drop duplicates and NaNs
    output = df[["text", "emotion"]].dropna().drop_duplicates(subset=["text"])

    # Save processed file
    output_file = f"data/processed/emotion_{split_name}.csv"
    output.to_csv(output_file, index=False, encoding="utf-8")

    # Print summary
    print(f"\n{split_name.upper()}")
    print("Rows:", len(output))
    print(output["emotion"].value_counts())
    print("Saved:", output_file)
