# ==========================================================
# Group K-Fold Training and Evaluation
# ----------------------------------------------------------
# This module evaluates the leakage classifier using
# Group K-Fold cross validation.
#
# Main idea:
# - Samples are split by group_id
# - Similar templates/groups are kept together
# - This reduces data leakage between train and test
#
# This is a stricter and more realistic evaluation method
# for synthetic datasets with repeated patterns.
# ==========================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

DATASET_PATH = "data/hebrew_data_leakage_dataset_18000_combined_v1.csv"
MODEL_OUTPUT_PATH = "models/leak_classifier.pkl"
VECTORIZER_OUTPUT_PATH = "models/tfidf_vectorizer.pkl"


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    required = {"text", "label", "group_id"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain text, label, group_id columns")

    df = df.dropna(subset=["text", "label", "group_id"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["group_id"] = df["group_id"].astype(str)

    X = df["text"]
    y = df["label"]
    groups = df["group_id"]

    print("\nDataset size:", len(df))
    print("Class distribution:")
    print(df["label"].value_counts())
    print("Number of groups:", df["group_id"].nunique())

    gkf = GroupKFold(n_splits=5)

    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2)
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(
            max_iter=1500,
            random_state=42
        )

        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        acc_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)

        print("Accuracy:", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall:", round(rec, 4))
        print("F1:", round(f1, 4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    print(f"\n{'=' * 60}")
    print("Average results across folds:")
    print("Accuracy:", round(np.mean(acc_scores), 4))
    print("Precision:", round(np.mean(precision_scores), 4))
    print("Recall:", round(np.mean(recall_scores), 4))
    print("F1:", round(np.mean(f1_scores), 4))

    # train final model on full dataset
    print("\nTraining final model on full dataset...")

    final_vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2)
    )
    X_all_vec = final_vectorizer.fit_transform(X)

    final_model = LogisticRegression(
        max_iter=1500,
        random_state=42
    )
    final_model.fit(X_all_vec, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, MODEL_OUTPUT_PATH)
    joblib.dump(final_vectorizer, VECTORIZER_OUTPUT_PATH)

    print("\nFinal model saved successfully.")
    print(f"Saved model to: {MODEL_OUTPUT_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}")