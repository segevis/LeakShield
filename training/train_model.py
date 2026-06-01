# ==========================================================
# Model Training Module
# ----------------------------------------------------------
# This module is responsible for training the leakage
# classification model on a dataset provided by the user.
#
# Expected dataset format:
# - A CSV file located in: data/training_dataset.csv
# - Required columns:
#     * text  -> the input sentence or text segment
#     * label -> numeric class:
#                0 = NON_LEAK
#                1 = LEAK
#
# Main responsibilities:
# 1. Load the dataset
# 2. Validate and clean it
# 3. Split it into train/test sets
# 4. Convert text into TF-IDF features
# 5. Train a Logistic Regression classifier
# 6. Evaluate model performance
# 7. Save the trained model and vectorizer into /models
#
# This training module is independent from the main system,
# allowing future retraining on new datasets without changing
# the inference pipeline.
# ==========================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = "data/hebrew_data_leakage_dataset_18000_combined_v1.csv"
MODEL_OUTPUT_PATH = "models/leak_classifier.pkl"
VECTORIZER_OUTPUT_PATH = "models/tfidf_vectorizer.pkl"


def validate_dataset(df):
    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain the following columns: 'text' and 'label'."
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError("Label column must contain only 0 and 1.")

    return df


def train_model():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH}"
        )

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    df = validate_dataset(df)

    X = df["text"]
    y = df["label"]

    print("\nDataset size:", len(df))
    print("Class distribution:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nBuilding TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training classifier...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test_vec)

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT_PATH)

    print("\nTraining completed successfully.")
    print(f"Saved model to: {MODEL_OUTPUT_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Training failed: {e}")