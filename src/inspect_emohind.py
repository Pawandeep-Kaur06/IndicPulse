import pandas as pd

train = pd.read_csv("data/raw/emohind/emoHi-train.csv")
valid = pd.read_csv("data/raw/emohind/emoHi-valid.csv")
test = pd.read_csv("data/raw/emohind/emoHi-test.csv")

print("Train shape:", train.shape)
print("Validation shape:", valid.shape)
print("Test shape:", test.shape)

print("\nColumns:")
print(train.columns.tolist())

print("\nFirst 5 training rows:")
print(train.head().to_string())

print("\nMissing values in train:")
print(train.isna().sum())

print("\nFirst row:")
print(train.iloc[0].to_dict())