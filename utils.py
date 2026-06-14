# ==========================================================
# Text Processing Utilities - generic structural segmentation
# ----------------------------------------------------------
# Segmentation is based only on document structure:
# - blank lines and paragraph boundaries
# - real line breaks
# - sentence-ending punctuation
# - bullets and numbered items
# - generic "label: value" structure
# - document-relative line statistics
#
# No document-specific headings, labels, company names,
# sentences, or keywords are hard-coded in this module.
# ==========================================================

from __future__ import annotations

import re
import statistics
from typing import Any, Callable, Dict, Iterable, List, Sequence


_TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?؟…][\"')\]}\u05F4\u05F3]*$")
_LIST_ITEM_RE = re.compile(
    r"^(?:"
    r"[\u2022•●▪▫◦‣⁃]\s+"
    r"|(?:\d+|[A-Za-z]|[א-ת])\s*[\.\)]\s+"
    r")"
)
_ONLY_MARKS_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?؟…])"
    r"(?=[\"')\]}\u05F4\u05F3]*\s+[^\s])"
)


def normalize_text(text: Any) -> str:
    """Normalize extraction artifacts while preserving document structure."""
    if text is None:
        return ""

    value = str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u00a0", " ")

    translations = {
        "״": '"',
        "“": '"',
        "”": '"',
        "’": "'",
        "׳": "'",
        "–": "-",
        "—": "-",
        "\u200f": "",
        "\u200e": "",
        "\ufeff": "",
        "\u202a": "",
        "\u202b": "",
        "\u202c": "",
    }
    for source, target in translations.items():
        value = value.replace(source, target)

    normalized_lines = []
    for raw_line in value.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        normalized_lines.append(line)

    value = "\n".join(normalized_lines)

    # Repair common PDF extraction joins without changing semantics.
    value = re.sub(r"([א-ת])(\d)", r"\1 \2", value)
    value = re.sub(r"(\d)([א-ת])", r"\1 \2", value)
    value = re.sub(r"([א-ת])([A-Za-z])", r"\1 \2", value)
    value = re.sub(r"([A-Za-z])([א-ת])", r"\1 \2", value)

    # Preserve paragraph structure but collapse excessive blank lines.
    value = re.sub(r"\n{3,}", "\n\n", value)

    # Normalize punctuation spacing.
    value = re.sub(r"\s+([,;.!?؟])", r"\1", value)

    return value.strip()


def _protect_inline_patterns(text: str) -> tuple[str, Dict[str, str]]:
    """
    Protect inline patterns whose punctuation is not a structural boundary.

    The rules describe generic syntax only; they do not contain
    document-specific words or labels.
    """
    protected: Dict[str, str] = {}

    patterns = (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\b(?:https?|ftp)://[^\s]+",
        r"\bwww\.[^\s]+",
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        r"\b\d{1,4}[/.]\d{1,2}[/.]\d{1,4}\b",
        r"\b\d+(?:[.,]\d+)+\b",
        r"\b(?:[A-Za-z0-9_-]+\.)+[A-Za-z]{2,}\b",
        r"\b[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]*",
    )

    def replace(match: re.Match[str]) -> str:
        key = f"\uFFF0{len(protected)}\uFFF1"
        protected[key] = match.group(0)
        return key

    result = text
    for pattern in patterns:
        result = re.sub(pattern, replace, result, flags=re.IGNORECASE)

    return result, protected


def _restore_inline_patterns(text: str, protected: Dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def _non_empty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def _document_line_statistics(lines: Sequence[str]) -> Dict[str, float]:
    """
    Calculate document-relative statistics.

    No fixed heading names or fixed character limits are used.
    """
    lengths = [len(line) for line in lines if line]
    word_counts = [len(line.split()) for line in lines if line]

    if not lengths:
        return {
            "median_length": 0.0,
            "lower_length": 0.0,
            "median_words": 0.0,
        }

    sorted_lengths = sorted(lengths)
    lower_half = sorted_lengths[: max(1, len(sorted_lengths) // 2)]

    return {
        "median_length": float(statistics.median(lengths)),
        "lower_length": float(statistics.median(lower_half)),
        "median_words": float(statistics.median(word_counts)),
    }


def _find_structural_colon(line: str) -> int | None:
    """
    Find a generic label/value colon while ignoring protected syntax.

    Examples of the supported structure:
        Label: value
        תווית: ערך

    The function does not know or list any label names.
    """
    protected_line, _ = _protect_inline_patterns(line)

    for index, character in enumerate(protected_line):
        if character not in {":", "："}:
            continue

        left = protected_line[:index].strip()
        right = protected_line[index + 1 :].strip()

        if not left or not right:
            continue
        if _TERMINAL_PUNCTUATION_RE.search(left):
            continue
        if "\n" in left or "\n" in right:
            continue

        # A label is structurally the prefix of the line rather than
        # a complete clause containing multiple punctuation boundaries.
        if any(mark in left for mark in (";", "?", "!", "؟")):
            continue

        return index

    return None


def _is_label_value_line(line: str) -> bool:
    return _find_structural_colon(str(line).strip()) is not None


def _is_list_item(line: str) -> bool:
    return bool(_LIST_ITEM_RE.match(str(line).strip()))


def _ends_sentence(line: str) -> bool:
    return bool(_TERMINAL_PUNCTUATION_RE.search(str(line).strip()))


def _looks_like_structural_heading(
    line: str,
    *,
    previous_line: str | None,
    next_line: str | None,
    previous_was_blank: bool,
    next_is_blank: bool,
    statistics_data: Dict[str, float],
) -> bool:
    """
    Infer a heading from layout and document-relative statistics only.

    No heading vocabulary is used.
    """
    candidate = str(line).strip()
    if not candidate:
        return False
    if _is_label_value_line(candidate):
        return False
    if _is_list_item(candidate):
        return False
    if _ends_sentence(candidate):
        return False
    if _ONLY_MARKS_RE.fullmatch(candidate):
        return False

    median_length = statistics_data["median_length"]
    lower_length = statistics_data["lower_length"]

    # A heading is relatively short inside its own document.
    relatively_short = (
        len(candidate) <= lower_length
        if lower_length > 0
        else True
    )

    # Structural evidence comes from layout, not vocabulary.
    separated_by_layout = previous_was_blank or next_is_blank
    followed_by_body = (
        next_line is not None
        and (
            len(next_line) > len(candidate)
            or _is_label_value_line(next_line)
            or _is_list_item(next_line)
        )
    )

    # Very first standalone lines often form a title block.
    starts_document = previous_line is None and next_line is not None

    return relatively_short and (
        separated_by_layout
        or followed_by_body
        or starts_document
    )


def _line_records(text: str) -> List[Dict[str, Any]]:
    raw_lines = text.split("\n")
    non_empty = [line.strip() for line in raw_lines if line.strip()]
    stats = _document_line_statistics(non_empty)

    records: List[Dict[str, Any]] = []

    for index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            records.append({"kind": "blank", "text": ""})
            continue

        previous_line = next(
            (
                raw_lines[position].strip()
                for position in range(index - 1, -1, -1)
                if raw_lines[position].strip()
            ),
            None,
        )
        next_line = next(
            (
                raw_lines[position].strip()
                for position in range(index + 1, len(raw_lines))
                if raw_lines[position].strip()
            ),
            None,
        )

        previous_was_blank = index == 0 or not raw_lines[index - 1].strip()
        next_is_blank = (
            index == len(raw_lines) - 1
            or not raw_lines[index + 1].strip()
        )

        if _is_label_value_line(line):
            kind = "metadata"
        elif _is_list_item(line):
            kind = "list_item"
        elif _looks_like_structural_heading(
            line,
            previous_line=previous_line,
            next_line=next_line,
            previous_was_blank=previous_was_blank,
            next_is_blank=next_is_blank,
            statistics_data=stats,
        ):
            kind = "heading"
        else:
            kind = "body"

        records.append({"kind": kind, "text": line})

    return records


def _join_wrapped_body_lines(lines: Sequence[str]) -> str:
    """
    Join lines that were wrapped only by the PDF layout.

    Sentence-ending punctuation remains a real boundary.
    """
    if not lines:
        return ""

    result = lines[0].strip()

    for line in lines[1:]:
        current = line.strip()
        if not current:
            continue

        if result.endswith(("-", "־")):
            result = result.rstrip("-־") + current
        else:
            result = f"{result} {current}"

    return re.sub(r"\s+", " ", result).strip()


def _split_sentences(text: str) -> List[str]:
    protected_text, protected = _protect_inline_patterns(text)

    pieces = re.split(
        r"(?<=[.!?؟…])\s+(?=[^\s])",
        protected_text,
    )

    results: List[str] = []
    for piece in pieces:
        restored = _restore_inline_patterns(piece.strip(), protected)
        restored = re.sub(r"\s+", " ", restored).strip()
        if restored and not _ONLY_MARKS_RE.fullmatch(restored):
            results.append(restored)

    return results


def _split_token_safe(
    text: str,
    *,
    tokenizer: Any,
    model_max_length: int | None = None,
) -> List[str]:
    """
    Split an unusually long natural unit by the active tokenizer.

    The limit is read from the tokenizer/model at runtime. No character
    threshold and no document-specific value is used.
    """
    if tokenizer is None:
        return [text]

    configured_limit = model_max_length
    if configured_limit is None:
        configured_limit = getattr(tokenizer, "model_max_length", None)

    if not isinstance(configured_limit, int) or configured_limit <= 0:
        return [text]

    special_count = len(
        tokenizer.build_inputs_with_special_tokens([])
    )
    available_tokens = configured_limit - special_count
    if available_tokens <= 0:
        return [text]

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )
    if len(token_ids) <= available_tokens:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(token_ids):
        end = min(start + available_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if chunk:
            chunks.append(chunk)

        start = end

    return chunks or [text]


def split_to_blocks(text: Any) -> List[str]:
    """
    Return generic structural blocks.

    Metadata, headings, list items and paragraphs are separated without
    exposing any internal marker in the returned text.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    records = _line_records(normalized)
    blocks: List[str] = []
    body_buffer: List[str] = []

    def flush_body() -> None:
        if not body_buffer:
            return

        joined = _join_wrapped_body_lines(body_buffer)
        if joined:
            blocks.append(joined)
        body_buffer.clear()

    for record in records:
        kind = record["kind"]
        value = record["text"]

        if kind == "blank":
            flush_body()
            continue

        if kind in {"metadata", "heading", "list_item"}:
            flush_body()
            blocks.append(value)
            continue

        body_buffer.append(value)

        if _ends_sentence(value):
            flush_body()

    flush_body()
    return blocks


def split_block_to_sentences(
    block: Any,
    *,
    tokenizer: Any = None,
    model_max_length: int | None = None,
) -> List[str]:
    """Split one structural block into natural, model-safe units."""
    value = normalize_text(block)
    if not value:
        return []

    natural_units = _split_sentences(value)
    output: List[str] = []

    for unit in natural_units:
        output.extend(
            _split_token_safe(
                unit,
                tokenizer=tokenizer,
                model_max_length=model_max_length,
            )
        )

    return [
        unit.strip()
        for unit in output
        if unit.strip() and not _ONLY_MARKS_RE.fullmatch(unit.strip())
    ]


def split_document(
    text: Any,
    *,
    tokenizer: Any = None,
    model_max_length: int | None = None,
) -> List[str]:
    """
    Segment a document using only generic structural rules.

    Optional tokenizer arguments allow the caller to enforce the active
    model's real token limit dynamically.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    results: List[str] = []

    for block in split_to_blocks(normalized):
        if _is_label_value_line(block):
            units = [block]
        elif _is_list_item(block):
            units = [block]
        elif "\n" not in block and not _ends_sentence(block):
            # Structurally detected standalone heading.
            units = [block]
        else:
            units = split_block_to_sentences(
                block,
                tokenizer=tokenizer,
                model_max_length=model_max_length,
            )

        for unit in units:
            cleaned = re.sub(r"\s+", " ", unit).strip()
            if not cleaned:
                continue
            if "<SPLIT_BOUNDARY>" in cleaned:
                raise RuntimeError(
                    "Internal segmentation marker leaked into output."
                )
            results.append(cleaned)

    return results
