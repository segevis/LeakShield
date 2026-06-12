from __future__ import annotations

import math
from typing import Any, Dict, List


# ==========================================================
# Production Decision Engine
# ----------------------------------------------------------
# Authoritative LEAK sources:
# 1. Regex found an explicit structured sensitive value.
# 2. The HeBERT Multi-Task Sequence Head classified LEAK.
#
# The Token Head localizes and types sensitive spans. It does
# not establish LEAK by itself when Sequence says NON_LEAK and
# Regex found nothing.
# ==========================================================


def _get_types(findings: List[Dict[str, Any]]) -> List[str]:
    return sorted({str(item.get("type")) for item in findings if item.get("type")})


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _binary_prediction(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        prediction = int(value)
    except (TypeError, ValueError):
        return None
    return prediction if prediction in (0, 1) else None


def _validate_sequence_result(sequence_analysis: Dict[str, Any]) -> Dict[str, Any]:
    status = str(sequence_analysis.get("status", "NOT_RUN"))
    if status == "ERROR":
        return {"validity": "ERROR", "prediction": None, "probability_leak": None, "threshold": None}
    if status != "OK":
        return {"validity": "NOT_RUN", "prediction": None, "probability_leak": None, "threshold": None}

    result = sequence_analysis.get("result")
    if not isinstance(result, dict):
        return {"validity": "INVALID_RESULT", "prediction": None, "probability_leak": None, "threshold": None}

    label = str(result.get("label", "")).upper()
    prediction = _binary_prediction(result.get("prediction"))
    probability = _finite_float(result.get("probability_leak"))
    threshold = _finite_float(result.get("threshold"))

    valid_range = (
        prediction is not None
        and probability is not None
        and threshold is not None
        and 0.0 <= probability <= 1.0
        and 0.0 <= threshold <= 1.0
    )
    if not valid_range:
        validity = "INVALID_RESULT"
    elif label == "LEAK" and prediction == 1 and probability >= threshold:
        validity = "VALID_LEAK"
    elif label == "NON_LEAK" and prediction == 0 and probability < threshold:
        validity = "VALID_NON_LEAK"
    else:
        validity = "INVALID_RESULT"

    return {
        "validity": validity,
        "prediction": prediction,
        "probability_leak": probability,
        "threshold": threshold,
    }


def _max_token_score(results: List[Dict[str, Any]]) -> float | None:
    scores: List[float] = []
    for item in results:
        score = _finite_float(item.get("score"))
        if score is not None and 0.0 <= score <= 1.0:
            scores.append(score)
    return max(scores) if scores else None


def _collect_model_errors(
    token_analysis: Dict[str, Any],
    sequence_analysis: Dict[str, Any],
) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    for component, analysis in (("token", token_analysis), ("sequence", sequence_analysis)):
        if analysis.get("status") != "ERROR" or not analysis.get("error"):
            continue
        errors.append({
            "component": component,
            "source": str(analysis.get("source", "")),
            "error_type": str(analysis.get("error_type", "RuntimeError")),
            "message": str(analysis.get("error")),
        })
    return errors


def _sequence_support(
    sequence_analysis: Dict[str, Any],
    *,
    used_for_final_label: bool,
    sequence_state: Dict[str, Any],
) -> Dict[str, Any]:
    result = sequence_analysis.get("result")
    if not isinstance(result, dict):
        result = {}
    return {
        "status": str(sequence_analysis.get("status", "NOT_RUN")),
        "validity": sequence_state["validity"],
        "label": result.get("label"),
        "prediction": sequence_state["prediction"],
        "confidence": result.get("confidence"),
        "probability_leak": sequence_state["probability_leak"],
        "temperature": result.get("temperature"),
        "threshold": sequence_state["threshold"],
        "source": sequence_analysis.get("source"),
        "used_for_final_label": bool(used_for_final_label),
        "used_for_risk_level": bool(used_for_final_label),
    }


def make_final_decision_v2(
    regex_results: List[Dict[str, Any]],
    token_analysis: Dict[str, Any],
    sequence_analysis: Dict[str, Any],
    analysis_status: str,
) -> Dict[str, Any]:
    regex_results = regex_results or []
    token_analysis = token_analysis or {}
    sequence_analysis = sequence_analysis or {}

    token_results = token_analysis.get("results")
    if not isinstance(token_results, list):
        token_results = []

    has_regex = bool(regex_results)
    has_token_spans = token_analysis.get("status") == "OK" and bool(token_results)
    sequence_state = _validate_sequence_result(sequence_analysis)
    has_sequence_leak = sequence_state["validity"] == "VALID_LEAK"
    has_sequence_non_leak = sequence_state["validity"] == "VALID_NON_LEAK"

    evidence_sources: List[str] = []
    reasons: List[str] = []
    if has_regex:
        evidence_sources.append("regex")
        reasons.append("Regex detected one or more structured sensitive values.")
    if has_sequence_leak:
        evidence_sources.append("hebert_sequence")
        reasons.append("The HeBERT Multi-Task Sequence Head classified the segment as LEAK.")
    if has_token_spans:
        evidence_sources.append("hebert_token_localization")
        reasons.append("The HeBERT Multi-Task Token Head localized candidate sensitive spans.")

    model_errors = _collect_model_errors(token_analysis, sequence_analysis)

    if has_regex or has_sequence_leak:
        final_label = "LEAK"
        risk_level = "HIGH" if has_regex and has_sequence_leak else "MEDIUM"
        if has_sequence_leak:
            result = sequence_analysis.get("result") or {}
            confidence_value = result.get("confidence")
            confidence_source = "hebert_sequence"
        else:
            confidence_value = 0.99
            confidence_source = "regex_deterministic"
    elif has_sequence_non_leak:
        final_label = "NON_LEAK"
        risk_level = "LOW"
        result = sequence_analysis.get("result") or {}
        confidence_value = result.get("confidence")
        confidence_source = "hebert_sequence"
        if has_token_spans:
            reasons.append(
                "Token spans are retained for localization, but the Token Head alone does not establish LEAK."
            )
        else:
            reasons.append("Neither Regex nor the Sequence Head found leakage.")
    else:
        final_label = "UNDETERMINED"
        risk_level = "UNKNOWN"
        confidence_value = None
        confidence_source = "none"
        reasons.append(
            "No reliable decision could be made because the Sequence Head was unavailable or invalid and Regex found no explicit value."
        )

    effective_status = str(analysis_status)
    if model_errors and effective_status == "COMPLETE":
        effective_status = "PARTIAL"
    if final_label == "UNDETERMINED":
        effective_status = "FAILED"

    return {
        "final_label": final_label,
        "risk_level": risk_level,
        "confidence_value": confidence_value,
        "confidence_source": confidence_source,
        "reasons": reasons,
        "evidence_sources": evidence_sources,
        "analysis_status": effective_status,
        "regex_types": _get_types(regex_results),
        "token_types": _get_types(token_results),
        "sequence_support": _sequence_support(
            sequence_analysis,
            used_for_final_label=has_sequence_leak,
            sequence_state=sequence_state,
        ),
        "token_support": {
            "status": str(token_analysis.get("status", "NOT_RUN")),
            "span_count": len(token_results),
            "max_score": _max_token_score(token_results),
            "used_for_final_label": False,
            "used_for_localization": bool(has_token_spans),
        },
        "model_errors": model_errors,
    }
