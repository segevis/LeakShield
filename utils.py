# ==========================================================
# Text Processing Utilities Module
# ----------------------------------------------------------
# This module is responsible for:
# 1. Normalizing Hebrew text extracted from PDF files
# 2. Splitting text into logical blocks
# 3. Splitting each block into smaller sentence-like units
#
# The goal is to make the system work better on generic
# Hebrew documents, even when PDF extraction is imperfect.
# ==========================================================

import re


def normalize_text(text):
    text = text.replace("\r", "\n")
    text = text.replace("״", '"')
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")
    text = text.replace("–", "-")

    # Fix spacing issues
    text = re.sub(r'[ \t]+', ' ', text)

    # Fix broken Hebrew-number joins
    text = re.sub(r'([א-ת])(\d{4,})', r'\1 \2', text)

    # Normalize separators
    text = re.sub(r'\n\s*---\s*\n', '\n---\n', text)

    # Remove extra empty lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def is_heading(line):
    line = line.strip()

    if not line:
        return False

    # Common Hebrew heading patterns
    heading_prefixes = [
        "נושא:",
        "סיכום פגישה:",
        "הודעה בצוות:",
        "לתיעוד פנימי:",
        "אבטחת מידע:",
        "מידע רפואי:",
        "מידע פיננסי:",
        "שיחה פנימית:",
        "מסמך משפטי:",
        "הודעת מערכת:",
        "דוח משאבי אנוש:",
        "עדכון טכני:",
        "תזכורת:",
        "מידע רגיש:",
        "הודעה כללית:",
        "הודעת הנהלה:",
        "סיכום שיחה עם ספק:"
    ]

    if line in ["---"]:
        return True

    if any(line.startswith(prefix) for prefix in heading_prefixes):
        return True

    # Generic heuristic: short line ending with colon
    if len(line) < 50 and line.endswith(":"):
        return True

    return False


def split_to_blocks(text):
    """
    Split document into logical blocks using:
    - separators like ---
    - headings
    - empty lines
    """
    lines = text.split("\n")
    blocks = []
    current_block = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if current_block:
                blocks.append(" ".join(current_block).strip())
                current_block = []
            continue

        if is_heading(stripped):
            if current_block:
                blocks.append(" ".join(current_block).strip())
                current_block = []

            if stripped != "---":
                current_block.append(stripped)
            else:
                if current_block:
                    blocks.append(" ".join(current_block).strip())
                    current_block = []
            continue

        current_block.append(stripped)

    if current_block:
        blocks.append(" ".join(current_block).strip())

    # remove very short garbage blocks
    blocks = [b for b in blocks if b and len(b) > 2]
    return blocks


def split_block_to_sentences(block):
    """
    Split a block into smaller sentence-like units.
    Keeps Hebrew documents reasonably stable without over-splitting.
    """
    parts = re.split(r'(?<=[.!?])\s+|\s+\-\-\-\s+|\n+', block)

    sentences = []
    for part in parts:
        cleaned = part.strip().strip('"').strip("'").strip()
        if cleaned:
            sentences.append(cleaned)

    return sentences


def split_document(text):
    """
    Full document splitting pipeline:
    1. Split into logical blocks
    2. Split each block into sentence-like units
    """
    blocks = split_to_blocks(text)
    results = []

    for block in blocks:
        sentences = split_block_to_sentences(block)

        # If the block did not split at all and is too long, keep it as one unit
        if not sentences:
            continue

        for sentence in sentences:
            results.append(sentence)

    return results