import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(train_df["text"], train_df["language"])

val_predictions = model.predict(val_df["text"])

print("Validation accuracy:", round(
    accuracy_score(val_df["language"], val_predictions), 4
))

print("\nClassification report:")
print(classification_report(
    val_df["language"],
    val_predictions,
    digits=4
))