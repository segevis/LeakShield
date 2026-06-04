# ==========================================================
# PDF Colored Report Generator
# ----------------------------------------------------------
# Creates a new PDF based on the original input PDF.
#
# The original PDF layout is preserved.
# Detected leakage text is marked in red.
#
# Important:
# We do not rebuild the PDF from extracted text, because that
# would destroy layout, tables, Hebrew alignment, fonts, etc.
# ==========================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import fitz  # PyMuPDF


RED = (1, 0, 0)
LIGHT_RED = (1, 0.85, 0.85)


def _iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Recursively iterate through dictionaries inside a JSON object.
    This makes the report generator robust to different results.json structures.
    """
    if isinstance(obj, dict):
        yield obj

        for value in obj.values():
            yield from _iter_dicts(value)

    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_dicts(item)


def _get_text_from_result(item: Dict[str, Any]) -> str:
    """
    Try several common field names used by the pipeline.
    """
    possible_keys = [
        "text",
        "sentence",
        "text_unit",
        "segment",
        "value",
        "original_text",
    ]

    for key in possible_keys:
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    # Sometimes the text is inside ml_result.
    ml_result = item.get("ml_result") or item.get("svm_result") or item.get("ml_res")

    if isinstance(ml_result, dict):
        value = ml_result.get("text")

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _get_page_number_from_result(item: Dict[str, Any]) -> Optional[int]:
    """
    Return zero-based page index if possible.
    Supports both:
    - page: 1
    - page_number: 1
    - page_index: 0
    """
    for key in ["page_index"]:
        value = item.get(key)

        if isinstance(value, int):
            return value

    for key in ["page", "page_number", "page_num"]:
        value = item.get(key)

        if isinstance(value, int):
            return max(0, value - 1)

        if isinstance(value, str) and value.isdigit():
            return max(0, int(value) - 1)

    return None


def _is_leak_result(item: Dict[str, Any]) -> bool:
    """
    Detect whether a JSON result item represents a final leak.
    """
    label_keys = [
        "final_label",
        "label",
        "classification",
        "decision",
    ]

    for key in label_keys:
        value = item.get(key)

        if isinstance(value, str) and value.upper() == "LEAK":
            return True

    final_decision = item.get("final_decision")

    if isinstance(final_decision, dict):
        value = final_decision.get("final_label")

        if isinstance(value, str) and value.upper() == "LEAK":
            return True

    return False


def _extract_leak_items(results_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract leak items from results.json in a structure-tolerant way.
    """
    leak_items = []

    for item in _iter_dicts(results_json):
        if not _is_leak_result(item):
            continue

        text = _get_text_from_result(item)

        if not text:
            continue

        leak_items.append(
            {
                "text": text,
                "page_index": _get_page_number_from_result(item),
            }
        )

    # Remove duplicates.
    unique = []
    seen = set()

    for item in leak_items:
        key = (item["page_index"], item["text"])

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def _search_text_variants(page: fitz.Page, text: str) -> List[fitz.Rect]:
    """
    Search text in a PDF page.

    PDF extraction can slightly change spaces, punctuation, or line breaks.
    We try several variants before giving up.
    """
    candidates = []

    cleaned = " ".join(str(text).split()).strip()

    if cleaned:
        candidates.append(cleaned)

    # Try shorter search if full sentence is too long.
    if len(cleaned) > 120:
        candidates.append(cleaned[:120].strip())

    if len(cleaned) > 80:
        candidates.append(cleaned[:80].strip())

    # Try removing numbering at the beginning.
    import re

    no_number = re.sub(r"^\d{1,3}\s*[\.\)]\s*", "", cleaned).strip()

    if no_number and no_number not in candidates:
        candidates.append(no_number)

    if len(no_number) > 100:
        candidates.append(no_number[:100].strip())

    # Try important phrase chunks.
    parts = re.split(r"[,;:]", no_number)

    for part in parts:
        part = part.strip()

        if len(part) >= 25 and part not in candidates:
            candidates.append(part)

    rects: List[fitz.Rect] = []

    for candidate in candidates:
        if not candidate:
            continue

        found = page.search_for(candidate)

        if found:
            rects.extend(found)
            break

    return rects


def create_colored_pdf_report(
    input_pdf_path: str,
    results_json_path: str,
    output_pdf_path: str,
) -> Dict[str, Any]:
    """
    Create a marked PDF:
    - original document remains unchanged
    - detected leak text is highlighted/underlined in red
    """
    input_pdf = Path(input_pdf_path)
    results_path = Path(results_json_path)
    output_pdf = Path(output_pdf_path)

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    with results_path.open("r", encoding="utf-8") as f:
        results_json = json.load(f)

    leak_items = _extract_leak_items(results_json)

    doc = fitz.open(str(input_pdf))

    marked_count = 0
    not_found = []

    for item in leak_items:
        text = item["text"]
        page_index = item["page_index"]

        candidate_pages = []

        if page_index is not None and 0 <= page_index < len(doc):
            candidate_pages = [page_index]
        else:
            candidate_pages = list(range(len(doc)))

        found_any = False

        for idx in candidate_pages:
            page = doc[idx]
            rects = _search_text_variants(page, text)

            if not rects:
                continue

            found_any = True

            for rect in rects:
                # Red highlight.
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=RED)
                annot.update()

                # Red underline to make it obvious even when highlight is subtle.
                underline = page.add_underline_annot(rect)
                underline.set_colors(stroke=RED)
                underline.update()

                marked_count += 1

            # If page was known or we found it, stop searching other pages.
            break

        if not found_any:
            not_found.append(text)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf), garbage=4, deflate=True)
    doc.close()

    return {
        "input_pdf": str(input_pdf),
        "results_json": str(results_path),
        "output_pdf": str(output_pdf),
        "leak_items": len(leak_items),
        "marked_count": marked_count,
        "not_found_count": len(not_found),
        "not_found_examples": not_found[:20],
    }


if __name__ == "__main__":
    from config import OUTPUT_JSON, PDF_PATH

    output_path = "output/marked_leaks_report.pdf"

    summary = create_colored_pdf_report(
        input_pdf_path=PDF_PATH,
        results_json_path=OUTPUT_JSON,
        output_pdf_path=output_path,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))