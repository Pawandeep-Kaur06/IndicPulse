import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/processed/intent_training_candidates.csv"
OUTPUT_DIR = "data/processed"

df = pd.read_csv(INPUT, dtype=str).fillna("")

assert len(df) > 100, f"Expected 100 rows, found {len(df)}"
assert df["message_id"].is_unique, "Duplicate message_id found"
assert df["text"].str.strip().ne("").all(), "Empty text found"
assert df["intent"].str.strip().ne("").all(), "Unlabeled row found"
assert df["label_status"].eq("labeled").all(), "Row is not marked labeled"

train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["language"],
)

validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["language"],
)

train.to_csv(f"{OUTPUT_DIR}/intent_train.csv", index=False)
validation.to_csv(f"{OUTPUT_DIR}/intent_validation.csv", index=False)
test.to_csv(f"{OUTPUT_DIR}/intent_test.csv", index=False)

print(f"Train: {len(train)} rows")
print(f"Validation: {len(validation)} rows")
print(f"Test: {len(test)} rows")

print("\nLanguage distribution:")
print(df["language"].value_counts().to_string())