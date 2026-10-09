from datasets import load_dataset
import pandas as pd

N = 5000

hindi = load_dataset(
    "ai4bharat/IndicCMix",
    data_files="hi.parquet",
    split="train"
)

punjabi = load_dataset(
    "ai4bharat/IndicCMix",
    data_files="pa.parquet",
    split="train"
)

english_df = pd.DataFrame({
    "text": hindi["english"][:N],
    "language": "English",
    "source": "IndicCMix"
})

hinglish_df = pd.DataFrame({
    "text": hindi["romanized_casual"][:N],
    "language": "Hinglish",
    "source": "IndicCMix"
})

punglish_df = pd.DataFrame({
    "text": punjabi["romanized_casual"][:N],
    "language": "Punglish",
    "source": "IndicCMix"
})

df = pd.concat(
    [english_df, hinglish_df, punglish_df],
    ignore_index=True
)

df = df.dropna(subset=["text"])
df = df.drop_duplicates(subset=["text"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("indicpulse_language_data.csv", index=False, encoding="utf-8")

print(df.head(10))
print("\nClass counts:")
print(df["language"].value_counts())
print("\nSaved:", len(df), "rows")