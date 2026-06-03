# ==========================================================
# Detection Module - Regex only
# ----------------------------------------------------------
# This module now contains only rule-based detectors.
#
# Important:
# The old HeBERT NER model is no longer loaded here.
# The fine-tuned typed HeBERT detector is loaded from:
# hebert_typed_leak_detector.py
# ==========================================================

from __future__ import annotations

import re
from typing import Dict, List


# ----------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------

EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
PHONE_PATTERN = r"\b05\d{8}\b"
ID_PATTERN = r"(?<!\d)\d{9}(?!\d)"

# Generic long numeric account-like values
ACCOUNT_PATTERN = r"(?<!\d)\d{8,10}(?!\d)"

# Money values:
# 12,500 ש"ח
# 12500 ש"ח
# 12,500 שח
MONEY_PATTERN = r'(?<!\d)\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:ש["״]?ח|שח)(?!\w)'

# Ticket / request IDs
TICKET_PATTERN = r"\b(?:REQ|INC|SR|HR|FIN)-\d{6}\b"

# Simple password-like patterns that often appear in synthetic / temporary credentials
PASSWORD_PATTERN = r"(?<!\S)[A-Za-z]+\d{4,}!(?!\S)|\b[A-Za-z]+\d{4,}!"

# Common technical secret keywords.
# These do not replace HeBERT/SVM, but help catch explicit technical leakage.
API_KEY_PATTERN = r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|refresh[_-]?token|private[_-]?key)\b"
CONNECTION_STRING_PATTERN = r"(?i)\b(?:connection\s*string|database\s*url|db[_-]?url|jdbc:|mongodb://|postgres://|mysql://)\b"


def _append_matches(
    results: List[Dict[str, str]],
    pattern: str,
    text: str,
    detection_type: str,
) -> None:
    for match in re.finditer(pattern, text):
        value = match.group(0)
        results.append(
            {
                "type": detection_type,
                "value": value,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
            }
        )


def detect_regex(text: str) -> List[Dict[str, str]]:
    """
    Detect structured sensitive patterns with regex.

    Returns:
        List of findings:
        {
            "type": "...",
            "value": "...",
            "start": int,
            "end": int,
            "source": "regex"
        }
    """
    if not text or not text.strip():
        return []

    results: List[Dict[str, str]] = []

    _append_matches(results, EMAIL_PATTERN, text, "EMAIL")
    _append_matches(results, PHONE_PATTERN, text, "PHONE")
    _append_matches(results, ID_PATTERN, text, "ID")
    _append_matches(results, MONEY_PATTERN, text, "MONEY")
    _append_matches(results, TICKET_PATTERN, text, "TICKET")
    _append_matches(results, PASSWORD_PATTERN, text, "PASSWORD")
    _append_matches(results, API_KEY_PATTERN, text, "TECHNICAL_SECRET_KEYWORD")
    _append_matches(results, CONNECTION_STRING_PATTERN, text, "CONNECTION_STRING_KEYWORD")

    # Account-like numbers: avoid duplicates with phone/ID.
    protected_values = {
        item["value"]
        for item in results
        if item["type"] in {"PHONE", "ID"}
    }

    for match in re.finditer(ACCOUNT_PATTERN, text):
        value = match.group(0)

        if value in protected_values:
            continue

        results.append(
            {
                "type": "ACCOUNT",
                "value": value,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
            }
        )

    return results


# Backward compatibility:
# Some older code may still import detect_hebert from detectors.py.
# We keep this function, but it no longer loads the old avichr/heBERT_NER model.
def detect_hebert(text: str):
    return []