from __future__ import annotations

from typing import Any

from hebert_multitask_detector import SOURCE, get_multitask_predictor


def analyze_sensitive_text(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {
            "analysis_status": "COMPLETE",
            "token_analysis": {
                "status": "OK", "results": [], "source": "empty_text", "error": None
            },
            "sequence_analysis": {
                "status": "OK",
                "result": {
                    "label": "NON_LEAK",
                    "prediction": 0,
                    "confidence": 1.0,
                    "probability_leak": 0.0,
                    "temperature": 0.0,
                    "threshold": 0.0,
                    "source": "empty_text",
                },
                "source": "empty_text",
                "error": None,
            },
            "runtime": {"source": "empty_text", "single_forward_pass": True},
            "model_errors": [],
        }

    try:
        result = get_multitask_predictor().predict(text)
        return {
            "analysis_status": "COMPLETE",
            "token_analysis": {
                "status": "OK",
                "results": result["token_results"],
                "source": SOURCE,
                "error": None,
            },
            "sequence_analysis": {
                "status": "OK",
                "result": result["sequence_result"],
                "source": SOURCE,
                "error": None,
            },
            "runtime": result["runtime"],
            "model_errors": [],
        }
    except Exception as error:
        public_error = {
            "component": "multitask",
            "source": SOURCE,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        return {
            "analysis_status": "FAILED",
            "token_analysis": {
                "status": "ERROR",
                "results": [],
                "source": SOURCE,
                "error": str(error),
            },
            "sequence_analysis": {
                "status": "ERROR",
                "result": None,
                "source": SOURCE,
                "error": str(error),
            },
            "runtime": {"source": SOURCE, "single_forward_pass": True},
            "model_errors": [public_error],
        }
