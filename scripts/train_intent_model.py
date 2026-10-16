import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

MODEL_NAME = "xlm-roberta-base"
LABELS = ["Complaint", "Opinion", "Other", "Question", "Request"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

def load_data(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    texts = df["text"].tolist()
    labels = []
    for intent in df["intent"]:
        vector = [0.0] * len(LABELS)
        for label in intent.split("|"):
            vector[LABEL_TO_ID[label.strip()]] = 1.0
        labels.append(vector)
    return Dataset.from_dict({"text": texts, "labels": labels})

train_ds = load_data("data/processed/intent_train.csv")
val_ds = load_data("data/processed/intent_validation.csv")
test_ds = load_data("data/processed/intent_test.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probabilities >= 0.5).astype(int)
    true = np.array(labels)

    tp = ((predictions == 1) & (true == 1)).sum()
    fp = ((predictions == 1) & (true == 0)).sum()
    fn = ((predictions == 0) & (true == 1)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {"precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="models/intent-xlmr",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("models/intent-xlmr-final")
tokenizer.save_pretrained("models/intent-xlmr-final")

metrics = trainer.evaluate(test_ds)
print(metrics)