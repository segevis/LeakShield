# ==========================================================
# Grouped Training Script
# ----------------------------------------------------------
# Uses GroupShuffleSplit instead of random split to reduce
# overly optimistic results on synthetic datasets.
#
# Expected dataset columns:
# - text
# - label
# - group_id
#
# This version performs clean evaluation on the new dataset
# without aggressive biasing:
# - no class_weight override
# - standard threshold = 0.50
# - keeps full error analysis
# - saves results to separate experiment files
# ==========================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_PATH = "data/hebrew_data_leakage_dataset_18000_combined_v1.csv"

MODEL_OUTPUT_PATH = "models/leak_classifier_experiment.pkl"
VECTORIZER_OUTPUT_PATH = "models/tfidf_vectorizer_experiment.pkl"

FALSE_NEGATIVES_OUTPUT = "output/false_negatives_experiment.csv"
FALSE_POSITIVES_OUTPUT = "output/false_positives_experiment.csv"
FULL_RESULTS_OUTPUT = "output/evaluation_results_experiment.csv"


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

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    print("\nBuilding TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training classifier...")

    model = LogisticRegression(
        max_iter=1500,
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    print("\nEvaluating model...")

    y_probs = model.predict_proba(X_test_vec)[:, 1]

    print("\nProbability distribution for class 1:")
    print("Min:", np.min(y_probs))
    print("10%:", np.percentile(y_probs, 10))
    print("25%:", np.percentile(y_probs, 25))
    print("50%:", np.percentile(y_probs, 50))
    print("75%:", np.percentile(y_probs, 75))
    print("90%:", np.percentile(y_probs, 90))
    print("Max:", np.max(y_probs))

    threshold = 0.50
    y_pred = (y_probs >= threshold).astype(int)

    print(f"\nEvaluation threshold: {threshold}")

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Create results DataFrame for analysis
    results_df = pd.DataFrame({
        "text": X_test.values,
        "true_label": y_test.values,
        "pred_label": y_pred,
        "prob_leak": y_probs
    })

    # False Negatives: true 1, predicted 0
    false_negatives = results_df[
        (results_df["true_label"] == 1) &
        (results_df["pred_label"] == 0)
    ].copy()

    # False Positives: true 0, predicted 1
    false_positives = results_df[
        (results_df["true_label"] == 0) &
        (results_df["pred_label"] == 1)
    ].copy()

    print("\n==========================================================")
    print("ERROR ANALYSIS")
    print("==========================================================")

    print("\nTotal False Negatives:", len(false_negatives))
    if len(false_negatives) > 0:
        print("Average probability of False Negatives:")
        print(false_negatives["prob_leak"].mean())

        print("\n--- Sample False Negatives (first 20) ---")
        for i, (_, row) in enumerate(false_negatives.head(20).iterrows(), start=1):
            print("\n----------------------------")
            print(f"[FN {i}]")
            print("Text:", row["text"])
            print("Prob:", row["prob_leak"])

    print("\nTotal False Positives:", len(false_positives))
    if len(false_positives) > 0:
        print("Average probability of False Positives:")
        print(false_positives["prob_leak"].mean())

        print("\n--- Sample False Positives (first 20) ---")
        for i, (_, row) in enumerate(false_positives.head(20).iterrows(), start=1):
            print("\n----------------------------")
            print(f"[FP {i}]")
            print("Text:", row["text"])
            print("Prob:", row["prob_leak"])

    # Save outputs
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT_PATH)

    results_df.to_csv(FULL_RESULTS_OUTPUT, index=False, encoding="utf-8-sig")
    false_negatives.to_csv(FALSE_NEGATIVES_OUTPUT, index=False, encoding="utf-8-sig")
    false_positives.to_csv(FALSE_POSITIVES_OUTPUT, index=False, encoding="utf-8-sig")

    print("\nTraining completed successfully.")
    print(f"Saved model to: {MODEL_OUTPUT_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_OUTPUT_PATH}")
    print(f"Saved full evaluation results to: {FULL_RESULTS_OUTPUT}")
    print(f"Saved false negatives to: {FALSE_NEGATIVES_OUTPUT}")
    print(f"Saved false positives to: {FALSE_POSITIVES_OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}")