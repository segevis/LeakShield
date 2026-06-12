# ==========================================================
# PDF Colored Report Generator
# ----------------------------------------------------------
# Creates a marked copy of the original PDF from JSON Schema V2.
# ==========================================================

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import fitz  # PyMuPDF


SCHEMA_VERSION = "2.0"
REGEX_HIGHLIGHT_COLOR = (1.0, 0.35, 0.35)
HEBERT_HIGHLIGHT_COLOR = (0.25, 0.75, 0.75)
REGEX_HIGHLIGHT_OPACITY = 0.35
HEBERT_HIGHLIGHT_OPACITY = 0.35
MIN_REGEX_VALUE_LENGTH = 2
CONFLICTING_SEQUENCE_STATES = {"INVALID_RESULT", "ERROR", "NOT_RUN", "VALID_NON_LEAK"}


def _build_summary(
    input_pdf_path: str,
    results_json_path: str,
    output_pdf_path: str,
) -> Dict[str, Any]:
    """Return an empty report summary."""
    return {
        "input_pdf": input_pdf_path,
        "results_json": results_json_path,
        "output_pdf": output_pdf_path,
        "schema_version": SCHEMA_VERSION,
        "leak_sentences": 0,
        "regex_findings": 0,
        "hebert_token_sentences": 0,
        "hebert_sequence_sentences": 0,
        "hebert_sentence_markings": 0,
        "marked_regex_rects": 0,
        "marked_sentence_rects": 0,
        "raw_sentence_rects": 0,
        "merged_sentence_rects": 0,
        "sentence_rects_merged": 0,
        "duplicate_regex_rects_skipped": 0,
        "duplicate_sentence_rects_skipped": 0,
        "not_found_regex_values": 0,
        "not_found_sentences": 0,
        "ambiguous_regex_matches": 0,
        "ambiguous_sentence_matches": 0,
        "invalid_regex_findings": 0,
        "invalid_token_findings": 0,
        "invalid_pages": 0,
        "not_found_examples": [],
        "warnings": [],
    }


def _record_warning(summary: Dict[str, Any], message: str) -> None:
    """Append a short warning to the report summary."""
    summary["warnings"].append(str(message))


def _record_not_found(summary: Dict[str, Any], example: Dict[str, Any]) -> None:
    """Store a bounded not-found example."""
    if len(summary["not_found_examples"]) < 20:
        summary["not_found_examples"].append(example)


def _load_v2_results(results_json_path: Path) -> Dict[str, Any]:
    """Load and validate a Schema V2 results JSON file."""
    if not results_json_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_json_path}")

    with results_json_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, dict):
        raise ValueError("results.json root must be a Schema V2 object.")

    if results.get("schema_version") != SCHEMA_VERSION:
        raise ValueError('results.json schema_version must be "2.0".')

    if "pages" not in results:
        raise ValueError("results.json must contain a pages list.")

    if not isinstance(results["pages"], list):
        raise ValueError("results.json pages must be a list.")

    for page in results["pages"]:
        if not isinstance(page, dict):
            raise ValueError("Each page entry must be an object.")

        if not isinstance(page.get("sentences"), list):
            raise ValueError("Each page entry must contain a sentences list.")

        for sentence_item in page["sentences"]:
            if not isinstance(sentence_item, dict):
                raise ValueError("Each sentence entry must be an object.")

    return results


def _iter_v2_sentences(results: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield page and sentence items in document order."""
    for page in results["pages"]:
        for sentence_item in page["sentences"]:
            yield page, sentence_item


def _normalize_search_text(text: str) -> str:
    """Normalize whitespace for PDF text search."""
    return " ".join(str(text).replace("\n", " ").split()).strip()


def _search_terms(text: str) -> List[str]:
    """Return safe search variants for the original text."""
    original = str(text).strip()
    normalized = _normalize_search_text(original)
    newline_replaced = str(text).replace("\n", " ").strip()

    terms: List[str] = []
    for candidate in (original, normalized, newline_replaced):
        if candidate and candidate not in terms:
            terms.append(candidate)

    return terms


def _search_regex_value_on_page(page: fitz.Page, value: str) -> List[fitz.Rect]:
    """Search a regex value on one page."""
    for term in _search_terms(value):
        rects = page.search_for(term)
        if rects:
            return list(rects)

    return []


def _search_sentence_on_page(page: fitz.Page, sentence: str) -> List[fitz.Rect]:
    """Search only full-sentence variants on one page."""
    for term in _search_terms(sentence):
        rects = page.search_for(term)
        if rects:
            return list(rects)

    return []


def _rects_share_line(rect: fitz.Rect, line_rect: fitz.Rect, y_tolerance: float) -> bool:
    """Return whether two rectangles likely belong to the same text line."""
    close_y_edges = (
        abs(rect.y0 - line_rect.y0) <= y_tolerance
        and abs(rect.y1 - line_rect.y1) <= y_tolerance
    )
    if close_y_edges:
        return True

    vertical_overlap = max(0.0, min(rect.y1, line_rect.y1) - max(rect.y0, line_rect.y0))
    min_height = max(0.1, min(rect.height, line_rect.height))
    return (vertical_overlap / min_height) >= 0.70


def _merge_sentence_rects_by_line(rects: List[fitz.Rect]) -> List[fitz.Rect]:
    """Merge sentence search fragments into one rectangle per text line."""
    if not rects:
        return []

    sorted_rects = sorted((fitz.Rect(rect) for rect in rects), key=lambda rect: (rect.y0, rect.x0))
    heights = sorted(rect.height for rect in sorted_rects if rect.height > 0)
    median_height = heights[len(heights) // 2] if heights else 4.0
    y_tolerance = max(1.0, median_height * 0.25)

    line_rects: List[fitz.Rect] = []
    for rect in sorted_rects:
        for index, line_rect in enumerate(line_rects):
            if _rects_share_line(rect, line_rect, y_tolerance):
                line_rects[index] = fitz.Rect(
                    min(line_rect.x0, rect.x0),
                    min(line_rect.y0, rect.y0),
                    max(line_rect.x1, rect.x1),
                    max(line_rect.y1, rect.y1),
                )
                break
        else:
            line_rects.append(fitz.Rect(rect))

    return sorted(line_rects, key=lambda rect: (rect.y0, rect.x0))


def _rect_key(page_index: int, rect: fitz.Rect, annotation_type: str) -> Tuple[Any, ...]:
    """Return a rounded key for deduplicating annotations."""
    return (
        page_index,
        annotation_type,
        round(rect.x0, 1),
        round(rect.y0, 1),
        round(rect.x1, 1),
        round(rect.y1, 1),
    )


def _deduplicate_rects(
    page_index: int,
    rects: Iterable[fitz.Rect],
    annotation_type: str,
    seen: set[Tuple[Any, ...]],
) -> Tuple[List[fitz.Rect], int]:
    """Return new rectangles and the number of skipped duplicates."""
    unique: List[fitz.Rect] = []
    skipped = 0

    for rect in rects:
        key = _rect_key(page_index, rect, annotation_type)
        if key in seen:
            skipped += 1
            continue

        seen.add(key)
        unique.append(rect)

    return unique, skipped


def _add_regex_annotation(page: fitz.Page, rect: fitz.Rect) -> None:
    """Add one point-specific regex annotation."""
    annot = page.add_highlight_annot(rect)
    annot.set_colors(stroke=REGEX_HIGHLIGHT_COLOR)
    if hasattr(annot, "set_opacity"):
        annot.set_opacity(REGEX_HIGHLIGHT_OPACITY)
    annot.update()


def _add_sentence_annotation(page: fitz.Page, rect: fitz.Rect) -> None:
    """Add one subtle sentence-level HeBERT highlight."""
    annot = page.add_highlight_annot(rect)
    annot.set_colors(stroke=HEBERT_HIGHLIGHT_COLOR)
    if hasattr(annot, "set_opacity"):
        annot.set_opacity(HEBERT_HIGHLIGHT_OPACITY)
    annot.update()


def _valid_regex_findings(sentence_item: Dict[str, Any], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return valid regex findings for one sentence item."""
    findings = sentence_item.get("regex_results", [])
    valid: List[Dict[str, Any]] = []

    if not isinstance(findings, list):
        return valid

    for finding in findings:
        if not isinstance(finding, dict):
            summary["invalid_regex_findings"] += 1
            continue

        value = finding.get("value")
        if not isinstance(value, str):
            summary["invalid_regex_findings"] += 1
            continue

        cleaned = _normalize_search_text(value)
        if len(cleaned) < MIN_REGEX_VALUE_LENGTH:
            summary["invalid_regex_findings"] += 1
            continue

        valid.append(
            {
                "value": cleaned,
                "type": finding.get("type"),
                "source": finding.get("source", "regex"),
                "start": finding.get("start"),
                "end": finding.get("end"),
            }
        )

    return valid


def _finite_score(value: Any) -> float | None:
    """Return a finite score in the inclusive 0-1 range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None

    return score


def _valid_token_finding(finding: Any) -> bool:
    """Return whether a token finding is structurally valid."""
    if not isinstance(finding, dict):
        return False

    finding_type = finding.get("type")
    value = finding.get("value")
    score = _finite_score(finding.get("score"))
    start = finding.get("start")
    end = finding.get("end")

    if not isinstance(finding_type, str) or not finding_type.strip():
        return False

    if not isinstance(value, str) or not value.strip():
        return False

    if score is None:
        return False

    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        return False

    if isinstance(end, bool) or not isinstance(end, int) or end < 0:
        return False

    return end > start


def _has_token_evidence(sentence_item: Dict[str, Any]) -> bool:
    """Return whether token analysis provided valid LEAK evidence."""
    token_analysis = sentence_item.get("token_analysis", {})
    if not isinstance(token_analysis, dict):
        return False

    if token_analysis.get("status") != "OK":
        return False

    results = token_analysis.get("results")
    return isinstance(results, list) and any(_valid_token_finding(item) for item in results)


def _count_invalid_token_findings(sentence_item: Dict[str, Any]) -> int:
    """Return the number of invalid token findings in a token results list."""
    token_analysis = sentence_item.get("token_analysis", {})
    if not isinstance(token_analysis, dict) or token_analysis.get("status") != "OK":
        return 0

    results = token_analysis.get("results")
    if not isinstance(results, list):
        return 0

    return sum(1 for item in results if not _valid_token_finding(item))


def _has_sequence_evidence(sentence_item: Dict[str, Any], summary: Dict[str, Any]) -> bool:
    """Return whether sequence analysis provided valid LEAK evidence."""
    decision = sentence_item.get("decision", {})
    if not isinstance(decision, dict):
        return False

    sequence_support = decision.get("sequence_support", {})
    validity = None
    if isinstance(sequence_support, dict):
        validity = sequence_support.get("validity")

    evidence_sources = decision.get("evidence_sources", [])
    if isinstance(evidence_sources, list) and "hebert_sequence" in evidence_sources:
        if validity in CONFLICTING_SEQUENCE_STATES:
            _record_warning(
                summary,
                f"Inconsistent sequence evidence: evidence_sources contains hebert_sequence but validity is {validity}.",
            )
            return False
        return True

    if not isinstance(sequence_support, dict):
        return False

    return (
        validity == "VALID_LEAK"
        and sequence_support.get("used_for_final_label") is True
    )


def _has_hebert_evidence(sentence_item: Dict[str, Any], summary: Dict[str, Any]) -> Tuple[bool, bool]:
    """Return token and sequence evidence flags."""
    return _has_token_evidence(sentence_item), _has_sequence_evidence(sentence_item, summary)


def _is_final_leak(sentence_item: Dict[str, Any]) -> bool:
    """Return whether the final V2 decision is LEAK."""
    decision = sentence_item.get("decision", {})
    return isinstance(decision, dict) and decision.get("final_label") == "LEAK"


def _page_index_from_sentence(
    doc: fitz.Document,
    page_item: Dict[str, Any],
    sentence_item: Dict[str, Any],
    summary: Dict[str, Any],
) -> int | None:
    """Convert a V2 one-based page number into a PyMuPDF page index."""
    container_page = page_item.get("page")
    sentence_page = sentence_item.get("page")

    def valid_page_number(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 1

    has_container_page = "page" in page_item
    has_sentence_page = "page" in sentence_item

    if has_container_page and not valid_page_number(container_page):
        summary["invalid_pages"] += 1
        _record_warning(summary, f"Invalid page number: {container_page!r}")
        _record_not_found(summary, {"page": container_page, "reason": "invalid_page"})
        return None

    if has_sentence_page and not valid_page_number(sentence_page):
        summary["invalid_pages"] += 1
        _record_warning(summary, f"Invalid page number: {sentence_page!r}")
        _record_not_found(summary, {"page": sentence_page, "reason": "invalid_page"})
        return None

    if has_container_page and has_sentence_page and container_page != sentence_page:
        summary["invalid_pages"] += 1
        _record_warning(
            summary,
            f"Page mismatch: container page {container_page} does not match sentence page {sentence_page}.",
        )
        _record_not_found(
            summary,
            {
                "container_page": container_page,
                "sentence_page": sentence_page,
                "reason": "page_mismatch",
            },
        )
        return None

    if has_sentence_page:
        page_number = sentence_page
    elif has_container_page:
        page_number = container_page
    else:
        summary["invalid_pages"] += 1
        _record_warning(summary, "Missing page number.")
        _record_not_found(summary, {"reason": "missing_page"})
        return None

    page_index = page_number - 1
    if not 0 <= page_index < len(doc):
        summary["invalid_pages"] += 1
        _record_warning(summary, f"Page number out of range: {page_number}")
        return None

    return page_index


def create_colored_pdf_report(
    input_pdf_path: str,
    results_json_path: str,
    output_pdf_path: str,
) -> Dict[str, Any]:
    """Create a marked copy of the original PDF from Schema V2 results."""
    input_pdf = Path(input_pdf_path)
    results_path = Path(results_json_path)
    output_pdf = Path(output_pdf_path)

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    if input_pdf.resolve() == output_pdf.resolve():
        raise ValueError("Output PDF path must be different from the input PDF path.")

    results = _load_v2_results(results_path)
    summary = _build_summary(str(input_pdf), str(results_path), str(output_pdf))

    doc = fitz.open(str(input_pdf))
    regex_rect_keys: set[Tuple[Any, ...]] = set()
    sentence_rect_keys: set[Tuple[Any, ...]] = set()

    try:
        for page_item, sentence_item in _iter_v2_sentences(results):
            if not _is_final_leak(sentence_item):
                continue

            summary["leak_sentences"] += 1
            page_index = _page_index_from_sentence(doc, page_item, sentence_item, summary)
            if page_index is None:
                continue

            page = doc[page_index]
            sentence = str(sentence_item.get("sentence", "")).strip()
            regex_findings = _valid_regex_findings(sentence_item, summary)
            summary["invalid_token_findings"] += _count_invalid_token_findings(sentence_item)
            token_evidence, sequence_evidence = _has_hebert_evidence(sentence_item, summary)

            summary["regex_findings"] += len(regex_findings)
            if token_evidence:
                summary["hebert_token_sentences"] += 1
            if sequence_evidence:
                summary["hebert_sequence_sentences"] += 1

            for finding in regex_findings:
                rects = _search_regex_value_on_page(page, finding["value"])
                if len(rects) > 1:
                    summary["ambiguous_regex_matches"] += 1

                if not rects:
                    summary["not_found_regex_values"] += 1
                    _record_not_found(
                        summary,
                        {
                            "page": page_index + 1,
                            "value": finding["value"],
                            "type": finding.get("type"),
                            "reason": "regex_value_not_found",
                        },
                    )
                    continue

                unique_rects, skipped = _deduplicate_rects(
                    page_index,
                    rects,
                    "regex",
                    regex_rect_keys,
                )
                summary["duplicate_regex_rects_skipped"] += skipped

                for rect in unique_rects:
                    _add_regex_annotation(page, rect)
                    summary["marked_regex_rects"] += 1

            if token_evidence or sequence_evidence:
                summary["hebert_sentence_markings"] += 1
                rects = _search_sentence_on_page(page, sentence)
                if len(rects) > 1:
                    summary["ambiguous_sentence_matches"] += 1

                if not rects:
                    summary["not_found_sentences"] += 1
                    _record_warning(summary, "LEAK sentence could not be located on its page.")
                    _record_not_found(
                        summary,
                        {
                            "page": page_index + 1,
                            "sentence": sentence,
                            "reason": "sentence_not_found",
                        },
                    )
                    continue

                summary["raw_sentence_rects"] += len(rects)
                merged_rects = _merge_sentence_rects_by_line(rects)
                summary["merged_sentence_rects"] += len(merged_rects)
                summary["sentence_rects_merged"] += max(0, len(rects) - len(merged_rects))

                unique_rects, skipped = _deduplicate_rects(
                    page_index,
                    merged_rects,
                    "sentence",
                    sentence_rect_keys,
                )
                summary["duplicate_sentence_rects_skipped"] += skipped

                for rect in unique_rects:
                    _add_sentence_annotation(page, rect)
                    summary["marked_sentence_rects"] += 1

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_pdf), garbage=4, deflate=True)
    finally:
        doc.close()

    return summary


if __name__ == "__main__":
    from config import OUTPUT_JSON, PDF_PATH

    output_path = "output/marked_leaks_report.pdf"

    report_summary = create_colored_pdf_report(
        input_pdf_path=PDF_PATH,
        results_json_path=OUTPUT_JSON,
        output_pdf_path=output_path,
    )

    print(json.dumps(report_summary, ensure_ascii=False, indent=2))
