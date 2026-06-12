
# ==========================================================
# Text Processing Utilities Module - segmentation V3
# ==========================================================

from __future__ import annotations

import re
from typing import Dict, List, Tuple


BOUNDARY = "\n<SPLIT_BOUNDARY>\n"
MAX_UNIT_CHARS = 260
TARGET_UNIT_CHARS = 180
MIN_UNIT_CHARS = 4
MIN_NATURAL_CHUNK_CHARS = 35

_METADATA_RE = re.compile(
    r"^(?:"
    r"נושא|תאריך|סיווג|מיועד\s+עבור|עבור|אל|מאת|עותק|גרסה|מחבר|"
    r"subject|date|classification|to|from|cc|version|author"
    r")\s*[:：]",
    re.IGNORECASE,
)

_COMMON_HEADING_WORDS = {
    "תקציר", "תקציר מנהלים", "סיכום", "מבוא", "רקע", "מטרה", "מטרות",
    "עלויות פיתוח", "הכנסות", "הכנסות ממוצרים מרכזיים",
    "תלות עסקית ואסטרטגית", "הערכת סיכונים", "סיכונים",
    "כיווני פעילות עתידיים", "הנחיות אבטחת מידע",
    "מסקנות", "המלצות", "נספח", "נספחים",
    "executive summary", "summary", "introduction", "background",
    "conclusions", "recommendations", "security guidelines",
}

_SENTENCE_START_RE = re.compile(r"[א-תA-Z0-9]")
_END_PUNCT_RE = re.compile(r"[.!?؟]$")
_ONLY_MARKERS_RE = re.compile(r"[\d\s\.\)\-_:]+")
_NUMBERED_ITEM_RE = re.compile(r"^\d{1,3}\s*[\.\)]")
_HEBREW_ITEM_RE = re.compile(r"^[א-ת]\s*[\.\)]\s+")


def normalize_text(text):
    """Normalize extraction artifacts while preserving paragraph structure."""
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    text = (
        text.replace("״", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("׳", "'")
        .replace("–", "-")
        .replace("—", "-")
        .replace("\u200f", "")
        .replace("\u200e", "")
        .replace("\ufeff", "")
    )

    # Normalize each line but keep newlines for structural parsing.
    text = "\n".join(
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    )

    # Repair common joins created by PDF extraction.
    text = re.sub(r"([א-ת])(\d{4,})", r"\1 \2", text)
    text = re.sub(r"(\d{4,})([א-ת])", r"\1 \2", text)
    text = re.sub(r"([א-ת])([A-Za-z_]{2,})", r"\1 \2", text)
    text = re.sub(r"([A-Za-z_]{2,})([א-ת])", r"\1 \2", text)

    # Repair missing whitespace after terminal punctuation:
    # "PlayStation.הערכת" -> "PlayStation. הערכת"
    text = re.sub(r"([.!?؟])(?=[א-תA-Z])", r"\1 ", text)

    # Normalize punctuation spacing without modifying decimal points.
    text = re.sub(r"\s+([.!?؟,;])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _protect_special_patterns(text):
    """Protect dots / slashes / colons that are not sentence boundaries."""
    protected: Dict[str, str] = {}
    counter = 0

    patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\bhttps?://[^\s]+",
        r"\bwww\.[^\s]+",
        r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b",
        r"\b\d{1,2}:\d{2}\b",
        r"\b\d+\.\d+\b",
        r"\b[\w\-]+\.(?:pdf|docx|doc|xlsx|csv|json|txt|py|js|ts|env|zip)\b",
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|No)\.",
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


def _is_metadata_line(line):
    return bool(_METADATA_RE.match(str(line).strip()))


def _looks_like_heading_line(line):
    """
    Detect a heading only at a structural paragraph boundary.

    Unlike the previous splitter, this function is not run on arbitrary
    wrapped lines in the middle of paragraphs.
    """
    line = str(line).strip()
    if not line or _is_metadata_line(line):
        return False

    normalized = line.rstrip(":").strip()
    lowered = normalized.lower()

    if lowered in _COMMON_HEADING_WORDS:
        return True

    if line.endswith(":") and len(line) <= 80 and len(line.split()) <= 8:
        return True

    if _END_PUNCT_RE.search(line):
        return False

    words = line.split()
    if not 1 <= len(words) <= 6:
        return False
    if len(line) > 60 or re.search(r"\d", line):
        return False

    # A title-like phrase should not contain clause punctuation.
    return line.count(",") == 0 and line.count(";") == 0


def _raw_paragraphs(text):
    """Return paragraphs separated by real blank lines."""
    paragraphs = []
    current = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        elif current:
            paragraphs.append(current)
            current = []

    if current:
        paragraphs.append(current)

    return paragraphs


def _join_wrapped_lines_naturally(lines):
    """
    Join PDF-wrapped lines, while preserving meaningful line boundaries.

    A newline is considered meaningful when:
    - the previous line ends with terminal punctuation, semicolon or colon;
    - the next line is a heading, metadata field or list item;
    - the current line itself is a heading or metadata field.

    Ordinary visual wrapping inside a sentence is joined with a space.
    """
    if not lines:
        return ""

    parts = [str(lines[0]).strip()]

    for previous, current in zip(lines, lines[1:]):
        previous = str(previous).strip()
        current = str(current).strip()

        meaningful_break = (
            bool(re.search(r"[.!?؟;:]$", previous))
            or _looks_like_heading_line(current)
            or _is_metadata_line(current)
            or bool(_NUMBERED_ITEM_RE.match(current))
            or bool(_HEBREW_ITEM_RE.match(current))
            or bool(re.match(r"^[\u2022•●▪▫◦]\s+", current))
        )

        separator = BOUNDARY if meaningful_break else " "
        parts.append(separator + current)

    return "".join(parts).strip()


def _structure_paragraph(lines):
    """
    Convert one raw paragraph into structural units.

    Wrapped PDF lines are joined, but natural boundaries are preserved with
    a temporary marker so sentence splitting can use them later.
    """
    if not lines:
        return []

    units = []
    index = 0

    first_metadata_index = next(
        (
            position
            for position, line in enumerate(lines)
            if _is_metadata_line(line)
        ),
        None,
    )

    title_lines = []
    if first_metadata_index is not None:
        title_lines = lines[:first_metadata_index]
        index = first_metadata_index

        if title_lines:
            units.append(_join_wrapped_lines_naturally(title_lines))

    metadata = []
    while index < len(lines) and _is_metadata_line(lines[index]):
        metadata.append(lines[index])
        index += 1

    if metadata:
        # Metadata is compact, but each field keeps a natural boundary.
        units.append(BOUNDARY.join(metadata))

    remaining = lines[index:]
    if not remaining:
        if not units and title_lines:
            units.append(_join_wrapped_lines_naturally(title_lines))
        return [unit for unit in units if unit]

    if _looks_like_heading_line(remaining[0]) and len(remaining) > 1:
        heading = remaining[0]
        body = _join_wrapped_lines_naturally(remaining[1:])
        units.append(f"{heading}{BOUNDARY}{body}".strip())
    else:
        units.append(_join_wrapped_lines_naturally(remaining))

    return [unit for unit in units if unit]


def _insert_list_boundaries(text):
    """Insert boundaries for bullets and list items."""
    bullet_chars = r"[\u2022•●▪▫◦]"
    text = re.sub(rf"(?:^|\s){bullet_chars}\s+(?=\S)", BOUNDARY, text)

    text = re.sub(
        r"(?:^|\s)(\d{1,3})\s*[\.\)]\s*(?=[^\d\s])",
        lambda match: BOUNDARY + f"<LISTNUM_{match.group(1)}> ",
        text,
    )

    text = re.sub(
        r"(?:^|\s)([א-ת])\s*[\.\)]\s+(?=\S)",
        lambda match: BOUNDARY + f"<LISTHEB_{match.group(1)}> ",
        text,
    )

    text = re.sub(r"\s*---+\s*", BOUNDARY, text)
    return text


def _split_sentences(text):
    """Split punctuation boundaries after special patterns are protected."""
    protected_text, protected = _protect_special_patterns(text)
    protected_text = _insert_list_boundaries(protected_text)

    parts = re.split(
        r"\s*<SPLIT_BOUNDARY>\s*|"
        r"(?:(?<=[!?؟])|(?<=[א-תA-Za-z][.]))\s+(?=[א-תA-Za-z0-9])",
        protected_text,
    )

    sentences = []
    for part in parts:
        part = _restore_special_patterns(part.strip(), protected)
        part = re.sub(r"<LISTNUM_(\d+)>", r"\1.", part)
        part = re.sub(r"<LISTHEB_([א-ת])>", r"\1.", part)
        part = re.sub(r"\s+", " ", part).strip().strip('"').strip("'")

        if len(part) < MIN_UNIT_CHARS:
            continue
        if _ONLY_MARKERS_RE.fullmatch(part):
            continue

        sentences.append(part)

    return sentences


def _split_by_natural_separators(text):
    """
    Split text by descending boundary strength.

    Strong:
    - explicit structural boundary
    - terminal punctuation
    - semicolon
    - colon

    Soft, used only for an already long fragment:
    - comma
    """
    protected_text, protected = _protect_special_patterns(text)

    strong_parts = re.split(
        r"\s*<SPLIT_BOUNDARY>\s*|"
        r"(?<=[.!?؟;:])\s+(?=[א-תA-Za-z0-9])",
        protected_text,
    )

    output = []

    for part in strong_parts:
        restored = _restore_special_patterns(part.strip(), protected)
        restored = re.sub(r"\s+", " ", restored).strip()

        if not restored:
            continue

        if len(restored) <= MAX_UNIT_CHARS:
            output.append(restored)
            continue

        comma_parts = re.split(r"(?<=,)\s+", restored)
        output.extend(
            part.strip()
            for part in comma_parts
            if part.strip()
        )

    return output


def _pack_natural_parts(parts, max_chars=MAX_UNIT_CHARS):
    """
    Pack natural fragments without creating tiny or overly long units.
    """
    packed = []
    current = ""

    for part in parts:
        part = str(part).strip()
        if not part:
            continue

        candidate = part if not current else f"{current} {part}"

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            packed.append(current)

        current = part

    if current:
        packed.append(current)

    return packed


def _split_by_words(text, max_chars=MAX_UNIT_CHARS):
    """Last-resort split for a fragment with no usable punctuation."""
    words = str(text).split()
    chunks = []
    current_words = []

    for word in words:
        candidate = " ".join(current_words + [word])

        if len(candidate) <= max_chars:
            current_words.append(word)
            continue

        if current_words:
            chunks.append(" ".join(current_words))

        current_words = [word]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _rebalance_tiny_chunks(chunks, max_chars=MAX_UNIT_CHARS):
    """Merge tiny fragments into a neighbour when that stays within the cap."""
    balanced = []

    for chunk in chunks:
        chunk = str(chunk).strip()
        if not chunk:
            continue

        if (
            balanced
            and len(chunk) < MIN_NATURAL_CHUNK_CHARS
            and len(balanced[-1]) + 1 + len(chunk) <= max_chars
        ):
            balanced[-1] = f"{balanced[-1]} {chunk}"
        else:
            balanced.append(chunk)

    if (
        len(balanced) >= 2
        and len(balanced[-1]) < MIN_NATURAL_CHUNK_CHARS
        and len(balanced[-2]) + 1 + len(balanced[-1]) <= max_chars
    ):
        balanced[-2] = f"{balanced[-2]} {balanced[-1]}"
        balanced.pop()

    return balanced


def _soft_split_long_text(text, max_chars=MAX_UNIT_CHARS):
    """
    Split long text at natural boundaries before falling back to word count.

    The preferred unit size is around TARGET_UNIT_CHARS, with a hard cap of
    MAX_UNIT_CHARS.
    """
    text = str(text).strip()

    if len(text) <= max_chars:
        return [text]

    natural_parts = _split_by_natural_separators(text)

    # Pack toward the target rather than filling every unit to the hard cap.
    packed = []
    current = ""

    for part in natural_parts:
        candidate = part if not current else f"{current} {part}"

        if len(candidate) <= TARGET_UNIT_CHARS:
            current = candidate
            continue

        if current:
            packed.append(current)
            current = ""

        if len(part) <= max_chars:
            current = part
        else:
            word_chunks = _split_by_words(part, max_chars=max_chars)
            packed.extend(word_chunks)

    if current:
        packed.append(current)

    final = []

    for chunk in packed:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(_split_by_words(chunk, max_chars=max_chars))

    final = _rebalance_tiny_chunks(final, max_chars=max_chars)

    return [
        chunk.strip()
        for chunk in final
        if len(chunk.strip()) >= MIN_UNIT_CHARS
    ]


def _merge_short_heading_units(units):
    """
    Merge only a genuine standalone heading with the next unit.

    Most headings are already attached in _structure_paragraph. This is a
    conservative fallback for headings separated by an empty line.
    """
    merged = []
    index = 0

    while index < len(units):
        current = str(units[index]).strip()

        if (
            _looks_like_heading_line(current)
            and index + 1 < len(units)
            and not _NUMBERED_ITEM_RE.match(str(units[index + 1]).strip())
            and not _HEBREW_ITEM_RE.match(str(units[index + 1]).strip())
        ):
            candidate = f"{current} {str(units[index + 1]).strip()}".strip()
            if len(candidate) <= TARGET_UNIT_CHARS:
                merged.append(candidate)
                index += 2
                continue

        merged.append(current)
        index += 1

    return merged


def split_to_blocks(text):
    """Return structural blocks while respecting PDF paragraph layout."""
    text = normalize_text(text)
    if not text:
        return []

    blocks = []
    for paragraph_lines in _raw_paragraphs(text):
        blocks.extend(_structure_paragraph(paragraph_lines))

    cleaned_blocks = []
    for block in blocks:
        block = re.sub(r"[ \t]+", " ", block).strip()
        block = re.sub(rf"\s*{re.escape(BOUNDARY.strip())}\s*", BOUNDARY, block)
        if block:
            cleaned_blocks.append(block)

    return cleaned_blocks


def split_block_to_sentences(block):
    """Split one structural block into model-sized sentence-like units."""
    if not block or not str(block).strip():
        return []

    # Metadata fields are intentionally kept together.
    metadata_parts = re.split(
        r"(?=(?:נושא|תאריך|סיווג|מיועד\\s+עבור|עבור|אל|מאת|עותק|גרסה|מחבר|subject|date|classification|to|from|cc|version|author)\\s*[:：])",
        block,
        flags=re.IGNORECASE,
    )
    metadata_parts = [part.strip() for part in metadata_parts if part.strip()]
    if metadata_parts and all(_is_metadata_line(part) for part in metadata_parts):
        return [block]

    results = []
    for sentence in _split_sentences(str(block).strip()):
        results.extend(_soft_split_long_text(sentence))

    return results


def split_document(text):
    """
    Full segmentation pipeline.

    1. Preserve blank-line paragraph structure.
    2. Join wrapped PDF lines before identifying headings.
    3. Group consecutive metadata fields.
    4. Attach headings to their first content sentence.
    5. Repair missing spaces after punctuation.
    6. Protect emails, URLs, dates, times, decimals and filenames.
    7. Split at natural punctuation and meaningful line breaks.
    8. Keep each model unit below MAX_UNIT_CHARS.
    """
    text = normalize_text(text)
    if not text:
        return []

    units = []
    for block in split_to_blocks(text):
        units.extend(split_block_to_sentences(block))

    units = _merge_short_heading_units(units)

    return [
        re.sub(r"\s+", " ", unit).strip()
        for unit in units
        if len(re.sub(r"\s+", " ", unit).strip()) >= MIN_UNIT_CHARS
    ]
