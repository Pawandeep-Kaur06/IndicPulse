import pandas as pd
from pathlib import Path

PATH = Path("data/processed/intent_synthetic_candidates.csv")
EXPECTED_IDS = [f"syn{i:03d}" for i in range(1, 87)]
VALID_LANGUAGES = {"English", "Hinglish", "Punglish"}
VALID_INTENTS = {"Question", "Request", "Complaint", "Opinion", "Other"}

df = pd.read_csv(PATH, dtype=str).fillna("")

assert list(df.columns) == [
    "id", "text", "language", "intent", "source", "review_status"
], "Unexpected columns or column order"

assert len(df) == 86, f"Expected 86 rows, found {len(df)}"
assert df["id"].tolist() == EXPECTED_IDS, "IDs are missing, duplicated, or out of order"
assert df["id"].is_unique, "Duplicate IDs found"
assert df["text"].str.strip().ne("").all(), "Empty text found"
assert df["language"].isin(VALID_LANGUAGES).all(), "Invalid language label found"
assert df["source"].eq("synthetic").all(), "Non-synthetic source found"
assert df["review_status"].eq("ai_reviewed").all(), "Row is not marked ai_reviewed"

invalid_intents = []
for row_id, labels in zip(df["id"], df["intent"]):
    split_labels = [label.strip() for label in labels.split("|")]
    if not split_labels or any(label not in VALID_INTENTS for label in split_labels):
        invalid_intents.append((row_id, labels))

assert not invalid_intents, f"Invalid intent labels: {invalid_intents}"

print("Validation passed.")
print("\nRows by language:")
print(df["language"].value_counts().sort_index().to_string())

all_labels = df["intent"].str.split("|").explode().str.strip()
print("\nLabel frequency:")
print(all_labels.value_counts().sort_index().to_string())

print("\nMulti-label rows:", int(df["intent"].str.contains("\\|").sum()))