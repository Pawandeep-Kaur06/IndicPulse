import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/processed/intent_training_candidates.csv"
OUTPUT_DIR = "data/processed"

df = pd.read_csv(INPUT, dtype=str).fillna("")

assert df["message_id"].is_unique, "Duplicate message_id found"
assert df["text"].str.strip().ne("").all(), "Empty text found"
assert df["label_status"].eq("labeled").all(), "Every reviewed row must be marked labeled"

# Reviewed but ambiguous examples have an empty intent list. Keep them in the
# splits as all-zero multi-label targets, and require the uncertainty flag.
empty_intent = df["intent"].str.strip().eq("")
assert df.loc[empty_intent, "review_needed"].str.lower().eq("true").all(), (
    "An empty intent is allowed only for a reviewed uncertain example"
)
assert len(df) > 100, f"Expected more than 100 reviewed rows, found {len(df)}"

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
print(f"Reviewed uncertain examples included with no positive intent: {int(empty_intent.sum())}")

print("\nLanguage distribution:")
print(df["language"].value_counts().to_string())
