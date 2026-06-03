# ==========================================================
# Text Processing Utilities Module
# ----------------------------------------------------------
# This module is responsible for:
# 1. Normalizing Hebrew text extracted from PDF files
# 2. Splitting text into logical blocks
# 3. Splitting each block into smaller sentence-like units
# 4. Conservatively merging short headings with their content
#
# The goal is to make the system work on generic Hebrew PDF
# documents even when PDF extraction is imperfect:
# - line breaks may disappear
# - numbered lists may be merged into one line
# - bullets may be removed or merged
# - paragraphs may be extracted as long blocks
#
# Important principle:
# The splitter is generic. It is not tailored to one test PDF.
# ==========================================================

import re


# Internal markers used only during text processing.
BOUNDARY = "\n<SPLIT_BOUNDARY>\n"


def normalize_text(text):
    if text is None:
        return ""

    text = str(text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize common quotes.
    text = text.replace("״", '"')
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")
    text = text.replace("׳", "'")

    # Normalize dashes.
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Remove invisible direction/BOM characters.
    text = text.replace("\u200f", "")
    text = text.replace("\u200e", "")
    text = text.replace("\ufeff", "")

    # Fix spacing issues.
    text = re.sub(r"[ \t]+", " ", text)

    # Fix broken Hebrew-number joins.
    text = re.sub(r"([א-ת])(\d{4,})", r"\1 \2", text)
    text = re.sub(r"(\d{4,})([א-ת])", r"\1 \2", text)

    # Fix broken Hebrew-English joins.
    text = re.sub(r"([א-ת])([A-Za-z_]{2,})", r"\1 \2", text)
    text = re.sub(r"([A-Za-z_]{2,})([א-ת])", r"\1 \2", text)

    # Normalize separators.
    text = re.sub(r"\n\s*---\s*\n", "\n---\n", text)

    # Remove too many empty lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _protect_special_patterns(text):
    """
    Protect patterns that contain dots/slashes/colons so sentence splitting
    will not break them incorrectly.

    Protected examples:
    - emails
    - URLs
    - dates
    - times
    - decimals
    - filenames
    """
    protected = {}
    counter = 0

    patterns = [
        # Emails
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

        # URLs
        r"\bhttps?://[^\s]+",
        r"\bwww\.[^\s]+",

        # Dates: 15/03/2026 or 15.03.2026
        r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b",

        # Times: 08:14
        r"\b\d{1,2}:\d{2}\b",

        # Decimal numbers: 3.14
        r"\b\d+\.\d+\b",

        # Common filenames
        r"\b[\w\-]+\.(?:pdf|docx|doc|xlsx|csv|json|txt|py|js|ts|env|zip)\b",
    ]

    def replace_match(match):
        nonlocal counter
        key = f"<PROTECTED_{counter}>"
        protected[key] = match.group(0)
        counter += 1
        return key

    for pattern in patterns:
        text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

    return text, protected


def _restore_special_patterns(text, protected):
    for key, value in protected.items():
        text = text.replace(key, value)

    return text


def is_heading(line):
    """
    Detect generic heading-like lines.

    This is only a heuristic. It is used mainly when the PDF extraction
    preserved line breaks.
    """
    line = str(line).strip()

    if not line:
        return False

    if line == "---":
        return True

    # Short generic heading ending with colon.
    if len(line) <= 70 and line.endswith(":"):
        return True

    heading_prefixes = [
        "נושא:",
        "סיכום:",
        "סיכום פגישה:",
        "סיכום שיחה:",
        "הודעה:",
        "הודעה בצוות:",
        "לתיעוד:",
        "לתיעוד פנימי:",
        "אבטחת מידע:",
        "מידע רפואי:",
        "מידע פיננסי:",
        "מידע רגיש:",
        "מידע מסחרי:",
        "שיחה פנימית:",
        "מסמך משפטי:",
        "הודעת מערכת:",
        "דוח:",
        "דוח משאבי אנוש:",
        "עדכון:",
        "עדכון טכני:",
        "תזכורת:",
        "הודעה כללית:",
        "הודעת הנהלה:",
        "סיכום שיחה עם ספק:",
    ]

    if any(line.startswith(prefix) for prefix in heading_prefixes):
        return True

    # Very short line without sentence punctuation can be a heading.
    if len(line) <= 45 and not re.search(r"[.!?؟]$", line):
        words = line.split()

        if 2 <= len(words) <= 7:
            return True

    return False


def _insert_generic_boundaries(text):
    """
    Insert generic boundaries into extracted text.

    This is not tailored to one test format.
    It handles common document structures:
    - numbered list items
    - bullets
    - separators
    - punctuation-based sentence endings
    """

    # Normalize bullets to boundaries.
    bullet_chars = r"[\u2022•●▪▫◦]"
    text = re.sub(rf"(?:^|\s)({bullet_chars})\s+(?=\S)", BOUNDARY, text)

    # Split before numbered list items:
    # 1. text
    # 1.text
    # 1) text
    #
    # Avoid dates/decimals by checking previous char is not digit / slash / dot / colon.
    text = re.sub(
        r"(?<![\d/.:])(?:^|\s)(\d{1,3})\s*[\.\)]\s*(?=[^\d\s])",
        BOUNDARY + r"\1.",
        text,
    )

    # Split before Hebrew clause markers:
    # א. text
    # ב) text
    text = re.sub(
        r"(?:^|\s)([א-ת])\s*[\.\)]\s+(?=\S)",
        BOUNDARY + r"\1. ",
        text,
    )

    # Split on explicit separators.
    text = re.sub(r"\s*---+\s*", BOUNDARY, text)

    # Split after sentence-ending punctuation followed by likely new sentence.
    # Protected emails/dates were replaced earlier.
    text = re.sub(
        r"(?<=[.!?؟])\s+(?=[א-תA-Z0-9])",
        BOUNDARY,
        text,
    )

    # Split after colon only when it looks like a short heading at line start.
    text = re.sub(
        r"(?:(?<=\n)|^)(.{2,50}:)\s+",
        r"\1" + BOUNDARY,
        text,
    )

    return text


def split_to_blocks(text):
    """
    Split document into logical blocks using:
    - empty lines
    - separators
    - headings
    - generic inserted boundaries
    """
    if not text or not str(text).strip():
        return []

    text = normalize_text(text)

    protected_text, protected = _protect_special_patterns(text)
    protected_text = _insert_generic_boundaries(protected_text)

    lines = protected_text.split("\n")
    blocks = []
    current_block = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if current_block:
                block = " ".join(current_block).strip()
                block = _restore_special_patterns(block, protected)
                blocks.append(block)
                current_block = []
            continue

        if stripped == "<SPLIT_BOUNDARY>":
            if current_block:
                block = " ".join(current_block).strip()
                block = _restore_special_patterns(block, protected)
                blocks.append(block)
                current_block = []
            continue

        restored_stripped = _restore_special_patterns(stripped, protected)

        if is_heading(restored_stripped):
            if current_block:
                block = " ".join(current_block).strip()
                block = _restore_special_patterns(block, protected)
                blocks.append(block)
                current_block = []

            if restored_stripped != "---":
                current_block.append(stripped)

            continue

        current_block.append(stripped)

    if current_block:
        block = " ".join(current_block).strip()
        block = _restore_special_patterns(block, protected)
        blocks.append(block)

    blocks = [re.sub(r"\s+", " ", b).strip() for b in blocks]
    blocks = [b for b in blocks if b and len(b) > 2]

    return blocks


def _soft_split_long_text(text, max_chars=350):
    """
    Fallback for long blocks that still did not split.

    This is generic:
    - prefer splitting around semicolon/colon/comma when the block is too long
    - keep chunks large enough to preserve context
    - avoid tiny fragments
    """
    text = str(text).strip()

    if len(text) <= max_chars:
        return [text]

    candidates = re.split(r"(?<=[;:])\s+|(?<=,)\s+", text)

    results = []
    current = ""

    for part in candidates:
        part = part.strip()

        if not part:
            continue

        if not current:
            current = part
            continue

        if len(current) + 1 + len(part) <= max_chars:
            current += " " + part
        else:
            if len(current) >= 25:
                results.append(current.strip())
                current = part
            else:
                current += " " + part

    if current.strip():
        results.append(current.strip())

    final_results = []

    for item in results:
        if len(item) <= max_chars:
            final_results.append(item)
            continue

        words = item.split()
        chunk = []

        for word in words:
            candidate = " ".join(chunk + [word])

            if len(candidate) <= max_chars:
                chunk.append(word)
            else:
                if chunk:
                    final_results.append(" ".join(chunk).strip())
                chunk = [word]

        if chunk:
            final_results.append(" ".join(chunk).strip())

    return [x for x in final_results if len(x.strip()) > 2]


def split_block_to_sentences(block):
    """
    Split a block into smaller sentence-like units.

    Designed for general Hebrew documents:
    - sentence punctuation
    - line boundaries already inserted by split_to_blocks
    - fallback for very long blocks
    """
    if not block or not str(block).strip():
        return []

    block = str(block).strip()

    protected_block, protected = _protect_special_patterns(block)

    parts = re.split(
        r"(?<=[.!?؟])\s+(?=[א-תA-Za-z0-9])|\n+",
        protected_block,
    )

    sentences = []

    for part in parts:
        part = part.strip().strip('"').strip("'").strip()

        if not part:
            continue

        part = _restore_special_patterns(part, protected)
        part = re.sub(r"\s+", " ", part).strip()

        if len(part) < 2:
            continue

        # Remove fragments that are only numbering/punctuation.
        if re.fullmatch(r"[\d\s\.\)\-_:]+", part):
            continue

        for sub_part in _soft_split_long_text(part):
            sub_part = sub_part.strip()

            if len(sub_part) > 2:
                sentences.append(sub_part)

    return sentences


def _starts_with_numbered_item(text):
    """
    Detect generic numbered list items:
    1. text
    1) text
    23. text
    """
    if not text:
        return False

    text = str(text).strip()

    return bool(re.match(r"^\d{1,3}\s*[\.\)]", text))


def _looks_like_short_heading(text):
    """
    Generic conservative heuristic:
    Detect only short heading/label units that should be attached
    to the following content.

    This intentionally does NOT merge long sentence fragments.
    """
    if not text or not str(text).strip():
        return False

    text = str(text).strip()
    words = text.split()

    # Do not merge numbered list items with the next item.
    if _starts_with_numbered_item(text):
        return False

    # Short heading ending with colon:
    # "פרטי עובד:"
    # "מידע רפואי:"
    # "סיכום פגישה:"
    if text.endswith(":") and len(text) <= 80 and len(words) <= 8:
        return True

    # Very short label-like fragments ending with Hebrew maqaf/dash.
    # Example: "שרתי ה־"
    if len(words) <= 5 and (text.endswith("־") or text.endswith("-")):
        return True

    return False


def _merge_incomplete_units(units, max_merged_chars=300):
    """
    Conservative merge:
    Merge only short headings/labels with the following unit.

    This prevents over-merging independent sentences.
    It is generic and suitable for many PDF documents.
    """
    if not units:
        return []

    merged = []
    i = 0

    while i < len(units):
        current = str(units[i]).strip()

        if not current:
            i += 1
            continue

        if _looks_like_short_heading(current) and i + 1 < len(units):
            next_unit = str(units[i + 1]).strip()

            # Do not merge heading with another numbered item.
            if next_unit and not _starts_with_numbered_item(next_unit):
                candidate = f"{current} {next_unit}".strip()

                if len(candidate) <= max_merged_chars:
                    merged.append(candidate)
                    i += 2
                    continue

        # Remove tiny meaningless fragments.
        if len(current) <= 3:
            i += 1
            continue

        if re.fullmatch(r"[\d\s\.\)\-_:]+", current):
            i += 1
            continue

        merged.append(current)
        i += 1

    return merged


def split_document(text):
    """
    Full document splitting pipeline:
    1. Normalize text
    2. Split into logical blocks
    3. Split each block into sentence-like units
    4. Conservatively merge short headings with their content
    5. Return stable text units for detection

    This is generic and does not depend on a specific document format.
    """
    text = normalize_text(text)

    if not text:
        return []

    blocks = split_to_blocks(text)
    results = []

    for block in blocks:
        sentences = split_block_to_sentences(block)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) < 2:
                continue

            results.append(sentence)

    # Conservative generic post-processing:
    # prevent short headings from being analyzed alone,
    # but avoid over-merging independent sentences.
    results = _merge_incomplete_units(results)

    return results