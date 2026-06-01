import os
import sys
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Allow running the script from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DATASET_PATH = "data/hebrew_data_leakage_dataset_19000_combined_v2_payroll_fix.csv"

MODEL_OUTPUT_PATH = "models/leak_classifier_svm_19000_groupkfold.pkl"
VECTORIZER_OUTPUT_PATH = "models/tfidf_vectorizer_svm_19000_groupkfold.pkl"
RESULTS_OUTPUT_PATH = "output/svm_groupkfold_results_19000.csv"


def main():
    print("Loading dataset...")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"text", "label", "group_id"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print()
    print(f"Dataset size: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts())
    print(f"Number of groups: {df['group_id'].nunique()}")

    X = df["text"].astype(str)
    y = df["label"]
    groups = df["group_id"]

    group_kfold = GroupKFold(n_splits=5)

    fold_results = []

    print()
    print("=" * 60)
    print("Running TF-IDF + Linear SVM with GroupKFold on 19k dataset")
    print("=" * 60)

    for fold_index, (train_index, test_index) in enumerate(
        group_kfold.split(X, y, groups),
        start=1,
    ):
        print()
        print("=" * 60)
        print(f"Fold {fold_index}")

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        classifier = LinearSVC(
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
        )

        classifier.fit(X_train_vec, y_train)
        y_pred = classifier.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

        print()
        print("Confusion Matrix:")
        print(matrix)

        print()
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        fold_results.append(
            {
                "fold": fold_index,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": matrix[0][0],
                "fp": matrix[0][1],
                "fn": matrix[1][0],
                "tp": matrix[1][1],
            }
        )

    results_df = pd.DataFrame(fold_results)

    print()
    print("=" * 60)
    print("Average results across folds:")
    print(f"Accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"Precision: {results_df['precision'].mean():.4f}")
    print(f"Recall: {results_df['recall'].mean():.4f}")
    print(f"F1: {results_df['f1'].mean():.4f}")

    os.makedirs("output", exist_ok=True)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print(f"Saved SVM GroupKFold results to: {RESULTS_OUTPUT_PATH}")

    print()
    print("Training final SVM model on full 19k dataset...")

    final_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
    )

    X_full_vec = final_vectorizer.fit_transform(X)

    final_classifier = LinearSVC(
        class_weight="balanced",
        max_iter=10000,
        random_state=42,
    )

    final_classifier.fit(X_full_vec, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_classifier, MODEL_OUTPUT_PATH)
    joblib.dump(final_vectorizer, VECTORIZER_OUTPUT_PATH)

    print()
    print("Final SVM model saved successfully.")
    print(f"Saved model to: {MODEL_OUTPUT_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()