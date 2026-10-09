import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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

labels = sorted(train_df["emotion"].unique())

cm = confusion_matrix(
    val_df["emotion"],
    predictions,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(11, 9))
display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)

plt.title("Emotion Baseline: Validation Confusion Matrix")
plt.tight_layout()
plt.savefig("results/emotion_confusion_matrix.png", dpi=200)
plt.show()

errors = val_df.copy()
errors["prediction"] = predictions

errors = errors[errors["emotion"] != errors["prediction"]]

print("Total validation errors:", len(errors))

print("\nMost common confusion pairs:")
print(
    errors.groupby(["emotion", "prediction"])
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("\nTen misclassified examples:")
print(
    errors[["text", "emotion", "prediction"]]
    .sample(10, random_state=42)
    .to_string(index=False)
)