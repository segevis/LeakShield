import os
import sys
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Allow running from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


EXPERIMENTS = [
    {
        "experiment_name": "logistic_18000",
        "dataset_name": "18k_combined",
        "dataset_path": "data/hebrew_data_leakage_dataset_18000_combined_v1.csv",
        "model_type": "logistic",
        "model_output_path": "models/comparison_logistic_18000.pkl",
        "vectorizer_output_path": "models/comparison_tfidf_logistic_18000.pkl",
    },
    {
        "experiment_name": "svm_18000",
        "dataset_name": "18k_combined",
        "dataset_path": "data/hebrew_data_leakage_dataset_18000_combined_v1.csv",
        "model_type": "svm",
        "model_output_path": "models/comparison_svm_18000.pkl",
        "vectorizer_output_path": "models/comparison_tfidf_svm_18000.pkl",
    },
    {
        "experiment_name": "logistic_19000_augmented",
        "dataset_name": "19k_augmented_payroll_fix",
        "dataset_path": "data/hebrew_data_leakage_dataset_19000_combined_v2_payroll_fix.csv",
        "model_type": "logistic",
        "model_output_path": "models/comparison_logistic_19000.pkl",
        "vectorizer_output_path": "models/comparison_tfidf_logistic_19000.pkl",
    },
    {
        "experiment_name": "svm_19000_augmented",
        "dataset_name": "19k_augmented_payroll_fix",
        "dataset_path": "data/hebrew_data_leakage_dataset_19000_combined_v2_payroll_fix.csv",
        "model_type": "svm",
        "model_output_path": "models/comparison_svm_19000.pkl",
        "vectorizer_output_path": "models/comparison_tfidf_svm_19000.pkl",
    },
]


SUMMARY_OUTPUT_PATH = "output/model_comparison_summary.csv"
FOLDS_OUTPUT_PATH = "output/model_comparison_folds.csv"


def build_classifier(model_type):
    if model_type == "logistic":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    if model_type == "svm":
        return LinearSVC(
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def evaluate_experiment(experiment):
    experiment_name = experiment["experiment_name"]
    dataset_name = experiment["dataset_name"]
    dataset_path = experiment["dataset_path"]
    model_type = experiment["model_type"]

    print()
    print("=" * 80)
    print(f"Starting experiment: {experiment_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Model type: {model_type}")
    print("=" * 80)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    required_columns = {"text", "label", "group_id"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in {dataset_path}: {missing_columns}")

    print()
    print(f"Dataset size: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts())
    print(f"Number of groups: {df['group_id'].nunique()}")

    X = df["text"].astype(str)
    y = df["label"]
    groups = df["group_id"]

    group_kfold = GroupKFold(n_splits=5)

    fold_rows = []

    for fold_index, (train_index, test_index) in enumerate(
        group_kfold.split(X, y, groups),
        start=1,
    ):
        print()
        print("-" * 60)
        print(f"{experiment_name} | Fold {fold_index}")

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

        classifier = build_classifier(model_type)
        classifier.fit(X_train_vec, y_train)

        y_pred = classifier.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = matrix.ravel()

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print()
        print("Confusion Matrix:")
        print(matrix)
        print()
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        fold_rows.append(
            {
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                "model_type": model_type,
                "fold": fold_index,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    folds_df = pd.DataFrame(fold_rows)

    summary = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "model_type": model_type,
        "dataset_size": len(df),
        "num_groups": df["group_id"].nunique(),
        "avg_accuracy": folds_df["accuracy"].mean(),
        "avg_precision": folds_df["precision"].mean(),
        "avg_recall": folds_df["recall"].mean(),
        "avg_f1": folds_df["f1"].mean(),
        "total_fp": folds_df["fp"].sum(),
        "total_fn": folds_df["fn"].sum(),
    }

    print()
    print("=" * 60)
    print(f"Average results for {experiment_name}:")
    print(f"Accuracy:  {summary['avg_accuracy']:.4f}")
    print(f"Precision: {summary['avg_precision']:.4f}")
    print(f"Recall:    {summary['avg_recall']:.4f}")
    print(f"F1:        {summary['avg_f1']:.4f}")
    print(f"Total FP:  {summary['total_fp']}")
    print(f"Total FN:  {summary['total_fn']}")

    print()
    print(f"Training final model for {experiment_name} on the full dataset...")

    final_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
    )

    X_full_vec = final_vectorizer.fit_transform(X)

    final_classifier = build_classifier(model_type)
    final_classifier.fit(X_full_vec, y)

    os.makedirs("models", exist_ok=True)

    joblib.dump(final_classifier, experiment["model_output_path"])
    joblib.dump(final_vectorizer, experiment["vectorizer_output_path"])

    print(f"Saved final model to: {experiment['model_output_path']}")
    print(f"Saved final vectorizer to: {experiment['vectorizer_output_path']}")

    return summary, fold_rows


def main():
    os.makedirs("output", exist_ok=True)

    all_summaries = []
    all_fold_rows = []

    for experiment in EXPERIMENTS:
        summary, fold_rows = evaluate_experiment(experiment)
        all_summaries.append(summary)
        all_fold_rows.extend(fold_rows)

    summary_df = pd.DataFrame(all_summaries)
    folds_df = pd.DataFrame(all_fold_rows)

    summary_df = summary_df.sort_values(
        by=["avg_f1", "avg_recall", "avg_accuracy"],
        ascending=False,
    )

    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    folds_df.to_csv(FOLDS_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("=" * 80)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 80)

    print(
        summary_df[
            [
                "experiment_name",
                "dataset_name",
                "model_type",
                "avg_accuracy",
                "avg_precision",
                "avg_recall",
                "avg_f1",
                "total_fp",
                "total_fn",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved summary results to: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved fold-level results to: {FOLDS_OUTPUT_PATH}")

    best_model = summary_df.iloc[0]

    print()
    print("=" * 80)
    print("BEST MODEL ACCORDING TO AVG F1")
    print("=" * 80)
    print(f"Experiment: {best_model['experiment_name']}")
    print(f"Dataset:    {best_model['dataset_name']}")
    print(f"Model type: {best_model['model_type']}")
    print(f"Accuracy:   {best_model['avg_accuracy']:.4f}")
    print(f"Precision:  {best_model['avg_precision']:.4f}")
    print(f"Recall:     {best_model['avg_recall']:.4f}")
    print(f"F1:         {best_model['avg_f1']:.4f}")
    print(f"Total FP:   {int(best_model['total_fp'])}")
    print(f"Total FN:   {int(best_model['total_fn'])}")


if __name__ == "__main__":
    main()