import pandas as pd

df = pd.read_csv("data/processed/intent_pilot.csv")

allowed_labels = {"Question", "Request", "Complaint", "Opinion", "Other"}

assert len(df) == 25, f"Expected 25 rows, found {len(df)}"
assert not df.isna().any().any(), "Some fields are missing."
assert df["id"].is_unique, "Duplicate IDs found."
assert df["source"].eq("synthetic").all(), "Check source values."

df["intent_list"] = df["intents"].str.split("|", regex=False)

for _, row in df.iterrows():
    labels = set(row["intent_list"])

    assert labels <= allowed_labels, (
        f"Unknown label in row {row['id']}: {labels}"
    )
    assert not ("Other" in labels and len(labels) > 1), (
        f"Other must be used alone in row {row['id']}"
    )

print("Pilot CSV checks passed.")
print("Rows:", len(df))

print("\nLanguage counts:")
print(df["language"].value_counts())

print("\nIntent counts (multi-label rows count toward each label):")
print(df["intent_list"].explode().value_counts())

print("\nRows with multiple intents:")
print(df["intent_list"].apply(len).gt(1).sum())