# ==========================================================
# Analysis Pipeline Module
# ----------------------------------------------------------
# Reusable analysis engine for both:
# - main.py command-line execution
# - gui.py future desktop interface
#
# Pipeline:
# PDF -> text extraction -> normalization -> splitting
# -> Regex detection -> HeBERT NER -> ML classification
# -> JSON/CSV output
# ==========================================================

import os
import json
import csv

from pdf_processor import extract_text
from utils import normalize_text, split_document
from detectors import detect_regex, detect_hebert
from ml_classifier import LeakClassifier
from config import MODEL_PATH, VECTORIZER_PATH


STRONG_TYPES = {"ID", "PASSWORD", "EMAIL", "PHONE"}


def save_results(results, output_json, output_csv):
    """
    Save analysis results to JSON and CSV files.
    """
    output_json_dir = os.path.dirname(output_json)
    output_csv_dir = os.path.dirname(output_csv)

    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)

    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    rows = []

    for page in results:
        for item in page["sentences"]:
            rows.append(
                {
                    "page": page["page"],
                    "sentence": item["sentence"],
                    "ml_label": item["ml"]["label"],
                    "confidence": item["ml"]["confidence"],
                    "regex_types": ", ".join([r["type"] for r in item["regex"]]),
                    "regex_values": ", ".join([str(r["value"]) for r in item["regex"]]),
                }
            )

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "page",
            "sentence",
            "ml_label",
            "confidence",
            "regex_types",
            "regex_values",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_pdf(
    pdf_path,
    output_json="output/results.json",
    output_csv="output/results.csv",
    threshold=0.45,
    debug=False,
):
    """
    Analyze a PDF file and save JSON/CSV outputs.

    Args:
        pdf_path: PDF file path.
        output_json: JSON output path.
        output_csv: CSV output path.
        threshold: Classification threshold.

    Returns:
        Dictionary summary of the analysis.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Input file must be a PDF file.")

    classifier = LeakClassifier(
        MODEL_PATH,
        VECTORIZER_PATH,
        threshold=threshold,
    )

    pages = extract_text(pdf_path, debug=debug)

    all_results = []
    total_sentences = 0
    total_leaks = 0

    for page_number, text in pages:
        clean_text = normalize_text(text)
        sentences = split_document(clean_text)

        page_data = {
            "page": page_number,
            "sentences": [],
        }

        for sentence in sentences:
            regex_res = detect_regex(sentence)
            hebert_res = detect_hebert(sentence)
            ml_res = classifier.classify(sentence)

            if any(r["type"] in STRONG_TYPES for r in regex_res):
                ml_res["label"] = "LEAK"
                ml_res["confidence"] = max(ml_res["confidence"], 0.99)

            if ml_res["label"] == "LEAK":
                total_leaks += 1

            total_sentences += 1

            page_data["sentences"].append(
                {
                    "sentence": sentence,
                    "regex": regex_res,
                    "hebert": hebert_res,
                    "ml": ml_res,
                }
            )

        all_results.append(page_data)

    save_results(all_results, output_json, output_csv)

    return {
        "status": "success",
        "pdf_path": pdf_path,
        "pages": len(all_results),
        "total_sentences": total_sentences,
        "total_leaks": total_leaks,
        "output_json": output_json,
        "output_csv": output_csv,
    }