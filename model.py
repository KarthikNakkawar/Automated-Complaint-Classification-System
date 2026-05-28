import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------
# Load Dataset
# -------------------------------
import os
file_path = os.path.join(os.path.dirname(__file__), "complaints.csv")
data = pd.read_csv(file_path)

print("Dataset loaded:", len(data))

# -------------------------------
# Text Cleaning
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["category"]

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(X)

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -------------------------------
# Train Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")
print(cm)

# -------------------------------
# Plot Confusion Matrix
# -------------------------------
plt.figure(figsize=(7,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("static/confusion_matrix.png")

print("\nConfusion matrix saved to static/confusion_matrix.png")

plt.close()

# -------------------------------
# Save Model
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully.")