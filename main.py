# ==========================================================
# Main Application Module
# ----------------------------------------------------------
# This is the core orchestrator of the system.
#
# Responsibilities:
# - Coordinates the full pipeline:
#     PDF → Text → Sentences → Detection → Classification
#
# Pipeline flow:
# 1. Extract text from PDF
# 2. Normalize and clean text
# 3. Split text into sentences
# 4. For each sentence:
#    - Apply Regex detection
#    - Apply HeBERT NER detection
#    - Apply ML classification
# 5. Save results to JSON and CSV files
#
# This module connects all components into one system.
# ==========================================================

import os
import json
import csv

from config import *
from pdf_processor import extract_text
from utils import normalize_text, split_document
from detectors import detect_regex, detect_hebert
from ml_classifier import LeakClassifier


def save_results(results):
    os.makedirs("output", exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    rows = []
    for page in results:
        for item in page["sentences"]:
            rows.append({
                "page": page["page"],
                "sentence": item["sentence"],
                "ml_label": item["ml"]["label"],
                "confidence": item["ml"]["confidence"]
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "sentence", "ml_label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not os.path.exists(PDF_PATH):
        print("PDF file not found")
        return

    classifier = LeakClassifier(MODEL_PATH, VECTORIZER_PATH, threshold=0.45)

    #  רק regex חזק באמת
    STRONG_TYPES = {"ID", "PASSWORD", "EMAIL", "PHONE"}

    pages = extract_text(PDF_PATH)

    all_results = []

    for page_number, text in pages:
        clean_text = normalize_text(text)
        sentences = split_document(clean_text)

        page_data = {
            "page": page_number,
            "sentences": []
        }

        for sentence in sentences:
            regex_res = detect_regex(sentence)
            hebert_res = detect_hebert(sentence)
            ml_res = classifier.classify(sentence)

            #  איזון: רק מקרים חזקים מכריחים LEAK
            if any(r["type"] in STRONG_TYPES for r in regex_res):
                ml_res["label"] = "LEAK"
                ml_res["confidence"] = max(ml_res["confidence"], 0.99)

            page_data["sentences"].append({
                "sentence": sentence,
                "regex": regex_res,
                "hebert": hebert_res,
                "ml": ml_res
            })

        all_results.append(page_data)

    save_results(all_results)

    print("✔ Analysis completed successfully")
    print("✔ Results saved in output folder")


if __name__ == "__main__":
    main()