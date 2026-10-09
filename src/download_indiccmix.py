from datasets import load_dataset

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

print("Hindi rows:", len(hindi))
print("Punjabi rows:", len(punjabi))

print("\nHindi example:")
print(hindi[0])

print("\nPunjabi example:")
print(punjabi[0])