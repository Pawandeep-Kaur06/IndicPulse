import pandas as pd

df = pd.read_csv("indicpulse_language_data.csv")

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nClass counts:")
print(df["language"].value_counts())

print("\nAverage text length by class:")
df["text_length"] = df["text"].str.len()
print(df.groupby("language")["text_length"].mean().round(2))

print("\nFive random examples:")
print(
    df[["text", "language"]]
    .sample(5, random_state=42)
    .to_string(index=False)
)