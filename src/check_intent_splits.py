import pandas as pd

train = pd.read_csv("data/processed/intent_train.csv", dtype={"message_id": str})
valid = pd.read_csv("data/processed/intent_validation.csv", dtype={"message_id": str})
test = pd.read_csv("data/processed/intent_test.csv", dtype={"message_id": str})

train_labels = set(train["intent"])

print("Training intents absent from validation:")
print(sorted(train_labels - set(valid["intent"])))

print("\nTraining intents absent from test:")
print(sorted(train_labels - set(test["intent"])))

print("\nSource-ID overlap between splits:")
print("Train / validation:", len(set(train["message_id"]) & set(valid["message_id"])))
print("Train / test:", len(set(train["message_id"]) & set(test["message_id"])))
print("Validation / test:", len(set(valid["message_id"]) & set(test["message_id"])))
