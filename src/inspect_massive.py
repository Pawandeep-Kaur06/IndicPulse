from datasets import load_dataset

hindi = load_dataset(
    "AmazonScience/massive",
    "hi-IN"
)

english = load_dataset(
    "AmazonScience/massive",
    "en-US"
)

print("Hindi splits:")
for split_name, split_data in hindi.items():
    print(split_name, len(split_data))

print("\nEnglish splits:")
for split_name, split_data in english.items():
    print(split_name, len(split_data))

print("\nHindi columns:")
print(hindi["train"].column_names)

print("\nHindi first example:")
print(hindi["train"][0])

print("\nEnglish first example:")
print(english["train"][0])

print("\nIntent labels:")
print(hindi["train"].features["intent"].names)

print("\nNumber of intents:")
print(len(hindi["train"].features["intent"].names))

