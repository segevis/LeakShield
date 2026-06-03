# ==========================================================
# Analysis Pipeline Module
# ----------------------------------------------------------
# Reusable analysis engine for both:
# - main.py command-line execution
# - gui.py desktop interface
#
# Pipeline:
# PDF -> text extraction -> normalization -> splitting
# -> Regex detection
# -> Fine-tuned HeBERT typed leakage detection
# -> Final SVM classification
# -> Decision Engine
# -> JSON/CSV output
# ==========================================================

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

from pdf_processor import extract_text
from utils import normalize_text, split_document
from detectors import detect_regex
from hebert_typed_leak_detector import detect_hebert_typed_leaks
from ml_classifier_svm_final import predict_svm_final
from decision_engine import make_final_decision


def _safe_join(values: List[Any]) -> str:
    return ", ".join([str(value) for value in values if value is not None])


def _finding_types(findings: List[Dict[str, Any]]) -> List[str]:
    return [str(item.get("type")) for item in findings if item.get("type")]


def _finding_values(findings: List[Dict[str, Any]]) -> List[str]:
    return [str(item.get("value")) for item in findings if item.get("value")]


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
            regex_findings = item.get("regex", [])
            hebert_findings = item.get("hebert_typed", [])
            svm_result = item.get("svm", {})
            decision = item.get("decision", {})

            rows.append(
                {
                    "page": page["page"],
                    "sentence": item["sentence"],

                    "final_label": decision.get("final_label"),
                    "risk_level": decision.get("risk_level"),
                    "final_confidence": decision.get("final_confidence"),
                    "reasons": " | ".join(decision.get("reasons", [])),

                    "svm_label": svm_result.get("label"),
                    "svm_confidence": svm_result.get("confidence"),
                    "svm_margin": svm_result.get("margin"),
                    "svm_source": svm_result.get("source"),

                    "regex_types": _safe_join(_finding_types(regex_findings)),
                    "regex_values": _safe_join(_finding_values(regex_findings)),

                    "hebert_typed_types": _safe_join(_finding_types(hebert_findings)),
                    "hebert_typed_values": _safe_join(_finding_values(hebert_findings)),

                    # Backward-compatible columns from the old CSV format.
                    "ml_label": svm_result.get("label"),
                    "confidence": svm_result.get("confidence"),
                }
            )

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "page",
            "sentence",

            "final_label",
            "risk_level",
            "final_confidence",
            "reasons",

            "svm_label",
            "svm_confidence",
            "svm_margin",
            "svm_source",

            "regex_types",
            "regex_values",

            "hebert_typed_types",
            "hebert_typed_values",

            # Backward-compatible old names.
            "ml_label",
            "confidence",
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
        threshold: kept for backward compatibility. The new SVM final wrapper
                   does not use this threshold directly.
        debug: passed to PDF extraction.

    Returns:
        Dictionary summary of the analysis.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Input file must be a PDF file.")

    pages = extract_text(pdf_path, debug=debug)

    all_results = []

    total_sentences = 0
    total_leaks = 0
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0

    for page_number, text in pages:
        clean_text = normalize_text(text)
        sentences = split_document(clean_text)

        page_data = {
            "page": page_number,
            "sentences": [],
        }

        for sentence in sentences:
            regex_res = detect_regex(sentence)
            hebert_typed_res = detect_hebert_typed_leaks(sentence)
            svm_res = predict_svm_final(sentence)
            svm_res["text"] = sentence

            decision = make_final_decision(
                regex_res=regex_res,
                hebert_typed_res=hebert_typed_res,
                ml_res=svm_res,
            )

            if decision.get("final_label") == "LEAK":
                total_leaks += 1

            risk_level = decision.get("risk_level")

            if risk_level == "HIGH":
                high_risk_count += 1
            elif risk_level == "MEDIUM":
                medium_risk_count += 1
            else:
                low_risk_count += 1

            total_sentences += 1

            page_data["sentences"].append(
                {
                    "sentence": sentence,

                    # New layer outputs.
                    "regex": regex_res,
                    "hebert_typed": hebert_typed_res,
                    "svm": svm_res,
                    "decision": decision,

                    # Backward-compatible keys for older GUI / result consumers.
                    "hebert": hebert_typed_res,
                    "ml": svm_res,
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
        "high_risk": high_risk_count,
        "medium_risk": medium_risk_count,
        "low_risk": low_risk_count,
        "output_json": output_json,
        "output_csv": output_csv,
    }