# ==========================================================
# Analysis Pipeline Module
# ----------------------------------------------------------
# Reusable analysis engine for both:
# - main.py command-line execution
# - gui.py desktop interface
#
# Production pipeline:
# PDF -> text extraction -> normalization -> splitting
# -> Regex detection
# -> HeBERT Multi-Task V2 (Token + Sequence heads)
# -> Decision Engine V2
# -> JSON/CSV V2 output
# ==========================================================

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from pdf_processor import extract_text
from utils import normalize_text, split_document
from detectors import detect_regex
from sensitive_text_analyzer import analyze_sensitive_text
from decision_engine import make_final_decision_v2


SCHEMA_VERSION = "2.0"
PIPELINE_NAME = "LeakShield"
DEFAULT_MULTITASK_MODEL_PATH = "models/hebert_multitask_v2/best"
DEFAULT_MULTITASK_CONFIG_PATH = "config/hebert_multitask_v2_production.json"
VALID_SEQUENCE_METADATA_STATES = {"VALID_LEAK", "VALID_NON_LEAK"}

CSV_FIELDNAMES = [
    "schema_version",
    "page",
    "sentence",
    "analysis_status",
    "final_label",
    "risk_level",
    "confidence_value",
    "confidence_source",
    "reasons",
    "evidence_sources",
    "regex_types",
    "regex_values",
    "token_status",
    "token_types",
    "token_values",
    "token_scores",
    "sequence_status",
    "sequence_label",
    "sequence_prediction",
    "sequence_confidence",
    "sequence_probability_leak",
    "sequence_temperature",
    "sequence_threshold",
    "sequence_source",
    "sequence_validity",
    "model_errors",
]


def _json_cell(value: Any) -> str:
    """Serialize a CSV cell that can contain structured values."""
    return json.dumps(value, ensure_ascii=False)


def _finding_types(findings: List[Dict[str, Any]]) -> List[str]:
    """Return finding types in their original order."""
    return [str(item.get("type")) for item in findings if item.get("type")]


def _finding_values(findings: List[Dict[str, Any]]) -> List[str]:
    """Return finding values in their original order."""
    return [str(item.get("value")) for item in findings if item.get("value")]


def _token_scores(findings: List[Dict[str, Any]]) -> List[float]:
    """Return valid token scores in their original order."""
    scores: List[float] = []

    for item in findings:
        try:
            scores.append(float(item.get("score", 0.0)))
        except (TypeError, ValueError):
            continue

    return scores


def _sequence_result(sequence_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Return the sequence result dict or an empty dict."""
    result = sequence_analysis.get("result")
    return result if isinstance(result, dict) else {}


def _finite_number(value: Any) -> float | None:
    """Return a finite numeric value, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def _configured_model_paths() -> tuple[str, str]:
    """Return shared Multi-Task model and production config paths."""
    return (
        os.environ.get(
            "HEBERT_MULTITASK_MODEL_PATH", DEFAULT_MULTITASK_MODEL_PATH
        ),
        os.environ.get(
            "HEBERT_MULTITASK_CONFIG_PATH", DEFAULT_MULTITASK_CONFIG_PATH
        ),
    )



def _merge_model_errors(*error_lists: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge model errors while removing exact duplicates."""
    merged: List[Dict[str, Any]] = []
    seen = set()

    for errors in error_lists:
        for error in errors or []:
            if not isinstance(error, dict):
                continue

            normalized = {
                "component": str(error.get("component", "")),
                "source": str(error.get("source", "")),
                "error_type": str(error.get("error_type", "")),
                "message": str(error.get("message", "")),
            }
            key = (
                normalized["component"],
                normalized["source"],
                normalized["error_type"],
                normalized["message"],
            )

            if key in seen:
                continue

            seen.add(key)
            merged.append(normalized)

    return merged


def _document_analysis_status(statuses: List[str]) -> str:
    """Compute document-level analysis status from sentence statuses."""
    if not statuses:
        return "COMPLETE"

    failed_count = sum(status == "FAILED" for status in statuses)

    if failed_count == len(statuses):
        return "FAILED"

    if failed_count > 0:
        return "PARTIAL"

    if any(status == "PARTIAL" for status in statuses):
        return "PARTIAL"

    return "COMPLETE"


def _build_empty_summary(total_pages: int) -> Dict[str, int]:
    """Return an empty V2 summary object."""
    return {
        "total_pages": total_pages,
        "total_sentences": 0,
        "total_leaks": 0,
        "total_non_leaks": 0,
        "total_undetermined": 0,
        "high_risk": 0,
        "medium_risk": 0,
        "low_risk": 0,
        "unknown_risk": 0,
        "complete_analyses": 0,
        "partial_analyses": 0,
        "failed_analyses": 0,
        "model_errors_count": 0,
    }


def _update_summary(summary: Dict[str, int], sentence_item: Dict[str, Any]) -> None:
    """Update summary counts from one sentence item."""
    decision = sentence_item["decision"]
    final_label = decision.get("final_label")
    risk_level = decision.get("risk_level")
    analysis_status = decision.get("analysis_status")

    summary["total_sentences"] += 1

    if final_label == "LEAK":
        summary["total_leaks"] += 1
    elif final_label == "NON_LEAK":
        summary["total_non_leaks"] += 1
    else:
        summary["total_undetermined"] += 1

    if risk_level == "HIGH":
        summary["high_risk"] += 1
    elif risk_level == "MEDIUM":
        summary["medium_risk"] += 1
    elif risk_level == "LOW":
        summary["low_risk"] += 1
    else:
        summary["unknown_risk"] += 1

    if analysis_status == "COMPLETE":
        summary["complete_analyses"] += 1
    elif analysis_status == "PARTIAL":
        summary["partial_analyses"] += 1
    else:
        summary["failed_analyses"] += 1

    summary["model_errors_count"] += len(sentence_item.get("model_errors", []))


def _extract_sequence_metadata(
    sequence_analysis: Dict[str, Any],
    decision: Dict[str, Any],
) -> tuple[float | None, float | None]:
    """Return temperature and threshold from a valid sequence result."""
    if sequence_analysis.get("status") != "OK":
        return None, None

    sequence_support = decision.get("sequence_support", {})
    if sequence_support.get("validity") not in VALID_SEQUENCE_METADATA_STATES:
        return None, None

    result = _sequence_result(sequence_analysis)
    if not result:
        return None, None

    temperature = _finite_number(result.get("temperature"))
    threshold = _finite_number(result.get("threshold"))

    if temperature is None or threshold is None:
        return None, None

    if temperature <= 0 or not 0.0 <= threshold <= 1.0:
        return None, None

    return temperature, threshold


def _build_csv_row(page: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    """Build one CSV row from one sentence item."""
    decision = item["decision"]
    token_analysis = item["token_analysis"]
    token_results = token_analysis.get("results") or []
    sequence_analysis = item["sequence_analysis"]
    sequence_result = _sequence_result(sequence_analysis)
    sequence_support = decision.get("sequence_support", {})

    return {
        "schema_version": SCHEMA_VERSION,
        "page": page["page"],
        "sentence": item["sentence"],
        "analysis_status": decision.get("analysis_status"),
        "final_label": decision.get("final_label"),
        "risk_level": decision.get("risk_level"),
        "confidence_value": decision.get("confidence_value"),
        "confidence_source": decision.get("confidence_source"),
        "reasons": _json_cell(decision.get("reasons", [])),
        "evidence_sources": _json_cell(decision.get("evidence_sources", [])),
        "regex_types": _json_cell(_finding_types(item.get("regex_results", []))),
        "regex_values": _json_cell(_finding_values(item.get("regex_results", []))),
        "token_status": token_analysis.get("status"),
        "token_types": _json_cell(_finding_types(token_results)),
        "token_values": _json_cell(_finding_values(token_results)),
        "token_scores": _json_cell(_token_scores(token_results)),
        "sequence_status": sequence_analysis.get("status"),
        "sequence_label": sequence_result.get("label"),
        "sequence_prediction": sequence_result.get("prediction"),
        "sequence_confidence": sequence_result.get("confidence"),
        "sequence_probability_leak": sequence_result.get("probability_leak"),
        "sequence_temperature": sequence_result.get("temperature"),
        "sequence_threshold": sequence_result.get("threshold"),
        "sequence_source": sequence_analysis.get("source"),
        "sequence_validity": sequence_support.get("validity"),
        "model_errors": _json_cell(item.get("model_errors", [])),
    }


def save_results(results: Dict[str, Any], output_json: str, output_csv: str) -> None:
    """Save V2 analysis results to JSON and CSV files."""
    output_json_dir = os.path.dirname(output_json)
    output_csv_dir = os.path.dirname(output_csv)

    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)

    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    rows = [
        _build_csv_row(page, item)
        for page in results.get("pages", [])
        for item in page.get("sentences", [])
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
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
    Analyze a PDF file and save JSON/CSV V2 outputs.

    Args:
        pdf_path: PDF file path.
        output_json: JSON output path.
        output_csv: CSV output path.
        threshold: kept for backward-compatible callers; unused in V2.
        debug: passed to PDF extraction.

    Returns:
        Dictionary summary of the analysis.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Input file must be a PDF file.")

    extracted_pages = extract_text(pdf_path, debug=debug)
    pages: List[Dict[str, Any]] = []
    sentence_statuses: List[str] = []
    summary = _build_empty_summary(total_pages=len(extracted_pages))
    sequence_temperature = None
    sequence_threshold = None
    multitask_model_path, multitask_config_path = _configured_model_paths()

    for page_number, text in extracted_pages:
        clean_text = normalize_text(text)
        sentences = split_document(clean_text)

        page_data = {
            "page": page_number,
            "sentences": [],
        }

        for sentence in sentences:
            regex_results = detect_regex(sentence)
            ai_analysis = analyze_sensitive_text(sentence)
            token_analysis = ai_analysis["token_analysis"]
            sequence_analysis = ai_analysis["sequence_analysis"]

            decision = make_final_decision_v2(
                regex_results=regex_results,
                token_analysis=token_analysis,
                sequence_analysis=sequence_analysis,
                analysis_status=ai_analysis["analysis_status"],
            )

            model_errors = _merge_model_errors(
                ai_analysis.get("model_errors", []),
                decision.get("model_errors", []),
            )

            sentence_item = {
                "page": page_number,
                "sentence": sentence,
                "regex_results": regex_results,
                "token_analysis": token_analysis,
                "sequence_analysis": sequence_analysis,
                "decision": decision,
                "model_errors": model_errors,
            }

            if sequence_temperature is None or sequence_threshold is None:
                temperature, threshold_value = _extract_sequence_metadata(
                    sequence_analysis,
                    decision,
                )
                if temperature is not None and threshold_value is not None:
                    sequence_temperature = temperature
                    sequence_threshold = threshold_value

            analysis_status = decision.get("analysis_status")
            if analysis_status not in {"COMPLETE", "PARTIAL", "FAILED"}:
                analysis_status = "FAILED"

            sentence_statuses.append(analysis_status)
            _update_summary(summary, sentence_item)
            page_data["sentences"].append(sentence_item)

        pages.append(page_data)

    document_analysis_status = _document_analysis_status(sentence_statuses)
    summary["total_pages"] = len(pages)

    results = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "pipeline": PIPELINE_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_file": str(pdf_path),
            "multitask_model": multitask_model_path,
            "multitask_config": multitask_config_path,
            "model_architecture": "shared_hebert_encoder_two_heads",
            "active_detection_layers": ["regex", "hebert_multitask_v2"],
            "sequence_temperature": sequence_temperature,
            "sequence_threshold": sequence_threshold,
            "analysis_status": document_analysis_status,
            "model_errors_count": summary["model_errors_count"],
        },
        "pages": pages,
        "summary": summary,
    }

    save_results(results, output_json, output_csv)

    return {
        "status": "success",
        "schema_version": SCHEMA_VERSION,
        "pdf_path": pdf_path,
        "pages": len(pages),
        "total_sentences": summary["total_sentences"],
        "total_leaks": summary["total_leaks"],
        "total_non_leaks": summary["total_non_leaks"],
        "total_undetermined": summary["total_undetermined"],
        "high_risk": summary["high_risk"],
        "medium_risk": summary["medium_risk"],
        "low_risk": summary["low_risk"],
        "unknown_risk": summary["unknown_risk"],
        "analysis_status": document_analysis_status,
        "model_errors_count": summary["model_errors_count"],
        "output_json": output_json,
        "output_csv": output_csv,
    }
