import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

train_df = pd.read_csv("data/processed/emotion_train.csv")
val_df = pd.read_csv("data/processed/emotion_validation.csv")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150000
    )),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(train_df["text"], train_df["emotion"])

predictions = model.predict(val_df["text"])

accuracy = accuracy_score(val_df["emotion"], predictions)
macro_f1 = f1_score(
    val_df["emotion"],
    predictions,
    average="macro"
)

print("Validation accuracy:", round(accuracy, 4))
print("Validation macro F1:", round(macro_f1, 4))

print("\nClassification report:")
print(classification_report(
    val_df["emotion"],
    predictions,
    digits=4
))