import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from pathlib import Path
import joblib

DATA_DIR = Path("data/processed")
train_file = DATA_DIR / "train.csv" if (DATA_DIR / "train.csv").exists() else Path("train.csv")
val_file = DATA_DIR / "validation.csv" if (DATA_DIR / "validation.csv").exists() else Path("validation.csv")

train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)

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

Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(model, "models/language_tfidf_logreg.joblib")
print("Saved language model to models/language_tfidf_logreg.joblib")

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