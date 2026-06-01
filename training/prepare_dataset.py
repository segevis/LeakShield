# ==========================================================
# Dataset Preparation Module
# ----------------------------------------------------------
# This module helps convert external datasets into the format
# required by the training pipeline.
#
# Expected final format:
# - CSV file with two columns:
#     * text
#     * label
#
# This script supports simple column renaming and optional
# mapping of textual labels to numeric labels.
#
# Example:
# Input dataset columns:
#   sentence, class_name
#
# Output dataset columns:
#   text, label
#
# Example label mapping:
#   NON_LEAK -> 0
#   LEAK     -> 1
#
# This script is useful when a new dataset arrives in a
# different structure.
# ==========================================================

import os
import pandas as pd

INPUT_DATASET_PATH = "data/raw_dataset.csv"
OUTPUT_DATASET_PATH = "data/training_dataset.csv"

# Change these values depending on the external dataset format
TEXT_COLUMN_NAME = "sentence"
LABEL_COLUMN_NAME = "class_name"

LABEL_MAPPING = {
    "NON_LEAK": 0,
    "LEAK": 1
}


def prepare_dataset():
    if not os.path.exists(INPUT_DATASET_PATH):
        raise FileNotFoundError(
            f"Input dataset file not found: {INPUT_DATASET_PATH}"
        )

    print("Loading raw dataset...")
    df = pd.read_csv(INPUT_DATASET_PATH)

    print("\nOriginal columns:")
    print(df.columns.tolist())

    if TEXT_COLUMN_NAME not in df.columns or LABEL_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Input dataset must contain '{TEXT_COLUMN_NAME}' and '{LABEL_COLUMN_NAME}'."
        )

    prepared_df = df[[TEXT_COLUMN_NAME, LABEL_COLUMN_NAME]].copy()
    prepared_df = prepared_df.rename(columns={
        TEXT_COLUMN_NAME: "text",
        LABEL_COLUMN_NAME: "label"
    })

    if prepared_df["label"].dtype == object:
        prepared_df["label"] = prepared_df["label"].map(LABEL_MAPPING)

    prepared_df = prepared_df.dropna(subset=["text", "label"])
    prepared_df["text"] = prepared_df["text"].astype(str)
    prepared_df["label"] = prepared_df["label"].astype(int)

    prepared_df.to_csv(OUTPUT_DATASET_PATH, index=False, encoding="utf-8-sig")

    print("\nPrepared dataset saved successfully.")
    print(f"Saved to: {OUTPUT_DATASET_PATH}")

    print("\nFirst 5 prepared rows:")
    print(prepared_df.head())


if __name__ == "__main__":
    try:
        prepare_dataset()
    except Exception as e:
        print(f"Dataset preparation failed: {e}")