from pathlib import Path

import pandas as pd
from datasets import load_dataset

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

datasets = {
    "Hindi": load_dataset("AmazonScience/massive", "hi-IN"),
    "English": load_dataset("AmazonScience/massive", "en-US"),
}

for split in ["train", "validation", "test"]:
    parts = []

    for language, dataset in datasets.items():
        data = dataset[split]
        intent_names = data.features["intent"].names

        part = pd.DataFrame({
            "id": data["id"],
            "text": data["utt"],
            "intent": [
                intent_names[label_id]
                for label_id in data["intent"]
            ],
            "language": language,
            "source": "MASSIVE",
        })

        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)

    # Save as massive_intent_{split}.csv to avoid overwriting real project intent splits
    output_file = output_dir / f"massive_intent_{split}.csv"
    combined.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n{split.upper()}")
    print("Rows:", len(combined))
    print("Missing text:", combined["text"].isna().sum())
    print("Number of intents:", combined["intent"].nunique())
    print(combined["language"].value_counts())
    print("Saved:", output_file)

    if split == "train":
        print("\nTraining examples per intent:")
        print(combined["intent"].value_counts().to_string())