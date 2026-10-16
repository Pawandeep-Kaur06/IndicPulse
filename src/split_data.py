import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path

DATA_DIR = Path("data/processed")
input_file = DATA_DIR / "indicpulse_language_data.csv" if (DATA_DIR / "indicpulse_language_data.csv").exists() else Path("indicpulse_language_data.csv")

df = pd.read_csv(input_file)

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["language"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["language"],
    random_state=42
)

out_dir = DATA_DIR if DATA_DIR.exists() else Path(".")
train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
val_df.to_csv(out_dir / "validation.csv", index=False, encoding="utf-8")
test_df.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain class counts:")
print(train_df["language"].value_counts())

print("\nValidation class counts:")
print(val_df["language"].value_counts())

print("\nTest class counts:")
print(test_df["language"].value_counts())