import pandas as pd
from transformers import pipeline

INPUT = "data/processed/intent_synthetic_candidates.csv"
OUTPUT = "data/processed/intent_model_predictions.csv"

classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
)

candidate_labels = [
    "a direct question that asks for information and ends with a question word or question mark",
    "a request asking someone to perform an action",
    "a complaint reporting a problem or dissatisfaction",
    "a clear personal preference, evaluation, or judgment about something",
    "a factual statement, announcement, status update, or notification that does not ask a question, request an action, express dissatisfaction, or give a personal preference",
]

label_map = {
    "a direct question that asks for information and ends with a question word or question mark": "Question",
    "a request asking someone to perform an action": "Request",
    "a complaint reporting a problem or dissatisfaction": "Complaint",
    "a clear personal preference, evaluation, or judgment about something": "Opinion",
    "a factual statement, announcement, status update, or notification that does not ask a question, request an action, express dissatisfaction, or give a personal preference": "Other",
}

BASE_THRESHOLD = 0.50
QUESTION_THRESHOLD = 0.70
REQUEST_THRESHOLD = 0.60
COMPLAINT_THRESHOLD = 0.55

df = pd.read_csv(INPUT, dtype=str, encoding="utf-8-sig").fillna("")
predictions = []

for _, row in df.iterrows():
    result = classifier(row["text"], candidate_labels, multi_label=True)

    thresholds = {
    "Question": QUESTION_THRESHOLD,
    "Request": REQUEST_THRESHOLD,
    "Complaint": COMPLAINT_THRESHOLD,
}

    selected = []
    for label, score in zip(result["labels"], result["scores"]):
        intent = label_map[label]
        threshold = thresholds.get(intent, BASE_THRESHOLD)
        if score >= threshold:
            selected.append(intent)

    predictions.append({
        "id": row["id"],
        "intent": "|".join(selected) if selected else "Other",
    })

pd.DataFrame(predictions).to_csv(OUTPUT, index=False)
print(f"Saved {len(predictions)} predictions to {OUTPUT}")