import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

os.makedirs("models", exist_ok=True)

train_df = pd.read_csv("data/processed/emotion_train.csv")

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

joblib.dump(model, "models/emotion_tfidf_logreg.joblib")

print("Saved model: models/emotion_tfidf_logreg.joblib")
print("Emotion classes:")
print(model.named_steps["classifier"].classes_)
