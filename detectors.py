# ==========================================================
# Detection Module (Regex + HeBERT)
# ----------------------------------------------------------
# This module is responsible for detecting potentially
# sensitive information in Hebrew text using two approaches:
#
# 1. Rule-based detection (Regex)
#    Detects structured patterns such as:
#    - Email addresses
#    - Phone numbers
#    - Israeli ID numbers
#    - Bank/account-like numbers
#    - Money amounts
#    - Ticket IDs
#    - Simple password-like strings
#
# 2. AI-based detection (HeBERT NER)
#    Detects contextual entities such as:
#    - Person names
#    - Organizations
#    - Locations
#    - Dates
#
# The module also filters noisy HeBERT outputs to reduce
# false detections from emails, symbols, short fragments,
# and numeric garbage.
# ==========================================================

import re
from transformers import pipeline

# ----------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------

EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
PHONE_PATTERN = r'\b05\d{8}\b'
ID_PATTERN = r'(?<!\d)\d{9}(?!\d)'

# Generic long numeric account-like values (bank account, internal account)
ACCOUNT_PATTERN = r'(?<!\d)\d{8,10}(?!\d)'

# Money values such as:
# 12,500 ש"ח
# 12500 ש"ח
# 12,500 שח
MONEY_PATTERN = r'(?<!\d)\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:ש["״]?ח|שח)(?!\w)'

# Ticket / request IDs
TICKET_PATTERN = r'\b(?:REQ|INC|SR|HR|FIN)-\d{6}\b'

# Simple password-like patterns that often appear in synthetic / temporary credentials
PASSWORD_PATTERN = r'(?<!\S)[A-Za-z]+\d{4,}!(?!\S)|\b[A-Za-z]+\d{4,}!'

# ----------------------------------------------------------
# Load HeBERT model
# ----------------------------------------------------------

ner = pipeline(
    "token-classification",
    model="avichr/heBERT_NER",
    tokenizer="avichr/heBERT_NER",
    aggregation_strategy="simple"
)

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def find_protected_spans(text):
    """
    Find spans of regex-detected items so HeBERT output that overlaps
    with them can be filtered out.
    """
    spans = []

    patterns = [
        EMAIL_PATTERN,
        PHONE_PATTERN,
        ID_PATTERN,
        ACCOUNT_PATTERN,
        MONEY_PATTERN,
        TICKET_PATTERN,
        PASSWORD_PATTERN,
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end()))

    return spans


def overlaps_protected_span(start, end, protected_spans):
    for span_start, span_end in protected_spans:
        if start < span_end and end > span_start:
            return True
    return False


def clean_entity_value(value):
    value = value.strip()
    value = value.strip('"').strip("'").strip()
    value = re.sub(r'\s+', ' ', value)
    return value


def is_noisy_entity(entity_value):
    """
    Filter low-quality HeBERT outputs.
    """
    value = clean_entity_value(entity_value)

    if not value:
        return True

    # subword fragments
    if "##" in value:
        return True

    # pure digits
    if value.isdigit():
        return True

    # single-character garbage
    if len(value) == 1:
        return True

    # very short latin fragments like "le", "com", "ya"
    if re.fullmatch(r'[A-Za-z]{1,3}', value):
        return True

    # symbols / punctuation only
    if re.fullmatch(r'[\W_]+', value):
        return True

    # email leftovers
    if "@" in value:
        return True

    # obvious domain leftovers
    lowered = value.lower()
    bad_tokens = {
        "gmail", "hotmail", "outlook", "mail", "com", "co", "org", "net"
    }
    if lowered in bad_tokens:
        return True

    # token leftovers from masking
    if "TOKEN" in value:
        return True

    # garbage money remainder
    if value in {"ש", "ח", 'ש"ח', "שח"}:
        return True

    return False


# ----------------------------------------------------------
# Regex detection
# ----------------------------------------------------------

def detect_regex(text):
    """
    Detect structured sensitive patterns with regex.
    """
    results = []

    # First detect specific patterns
    emails = re.findall(EMAIL_PATTERN, text)
    phones = re.findall(PHONE_PATTERN, text)
    ids = re.findall(ID_PATTERN, text)
    money_values = re.findall(MONEY_PATTERN, text)
    tickets = re.findall(TICKET_PATTERN, text)
    passwords = re.findall(PASSWORD_PATTERN, text)
    accounts = re.findall(ACCOUNT_PATTERN, text)

    for item in emails:
        results.append({
            "type": "EMAIL",
            "value": item
        })

    for item in phones:
        results.append({
            "type": "PHONE",
            "value": item
        })

    for item in ids:
        results.append({
            "type": "ID",
            "value": item
        })

    # Avoid duplicates: if account looks like phone or ID, skip it
    protected_numbers = set(phones + ids)

    for item in accounts:
        if item in protected_numbers:
            continue
        results.append({
            "type": "ACCOUNT",
            "value": item
        })

    for item in money_values:
        results.append({
            "type": "MONEY",
            "value": item
        })

    for item in tickets:
        results.append({
            "type": "TICKET",
            "value": item
        })

    for item in passwords:
        results.append({
            "type": "PASSWORD",
            "value": item
        })

    return results

# ----------------------------------------------------------
# HeBERT detection
# ----------------------------------------------------------

def detect_hebert(text):
    """
    Detect contextual entities with HeBERT while filtering noisy outputs.
    """
    results = []
    protected_spans = find_protected_spans(text)

    allowed_types = {
        "B_PERS",
        "B_ORG",
        "B_LOC",
        "B_DATE"
    }

    entities = ner(text)

    for ent in entities:
        entity_type = ent.get("entity_group", "")
        entity_value = clean_entity_value(ent.get("word", ""))
        entity_score = float(ent.get("score", 0.0))
        entity_start = ent.get("start", -1)
        entity_end = ent.get("end", -1)

        if entity_type not in allowed_types:
            continue

        if entity_score < 0.55:
            continue

        if is_noisy_entity(entity_value):
            continue

        if entity_start != -1 and entity_end != -1:
            if overlaps_protected_span(entity_start, entity_end, protected_spans):
                continue

        results.append({
            "type": entity_type,
            "value": entity_value,
            "score": round(entity_score, 4)
        })

    return results