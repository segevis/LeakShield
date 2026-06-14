from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import fitz
import pytest

import pdf_colored_report
from pdf_colored_report import create_colored_pdf_report


FORBIDDEN_FIELDS = {
    "svm",
    "ml",
    "ml_result",
    "svm_result",
    "hebert_typed",
    "hebert",
}


def make_pdf(path: Path, pages: List[List[str]]) -> Path:
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=12)
            y += 28
    doc.save(str(path))
    doc.close()
    return path


def count_annotations(path: Path) -> int:
    doc = fitz.open(str(path))
    try:
        return sum(1 for page in doc for _ in (page.annots() or []))
    finally:
        doc.close()


def load_annotations_by_page(path: Path) -> List[int]:
    doc = fitz.open(str(path))
    try:
        return [sum(1 for _ in (page.annots() or [])) for page in doc]
    finally:
        doc.close()


def load_annotation_details(path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(str(path))
    try:
        details: List[Dict[str, Any]] = []
        for page_index, page in enumerate(doc):
            for annot in page.annots() or []:
                details.append(
                    {
                        "page": page_index + 1,
                        "type": annot.type[1],
                        "opacity": annot.opacity,
                        "rect": fitz.Rect(annot.rect),
                    }
                )
        return details
    finally:
        doc.close()


def write_results(path: Path, pages: List[Dict[str, Any]], schema_version: str = "2.0") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "metadata": {},
                "pages": pages,
                "summary": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def regex_finding(value: str = "secret@example.com") -> Dict[str, Any]:
    return {
        "type": "EMAIL",
        "value": value,
        "start": 0,
        "end": len(value),
        "source": "regex",
    }


def token_analysis(results: List[Dict[str, Any]] | None = None, status: str = "OK") -> Dict[str, Any]:
    return {
        "status": status,
        "results": results if results is not None else [],
        "source": "hebert_token_classifier_v3",
        "error": None,
    }


def token_finding(value: str = "token") -> Dict[str, Any]:
    return {
        "type": "TOKEN_SECRET",
        "value": value,
        "score": 0.9,
        "start": 0,
        "end": len(value),
        "source": "hebert_token_classifier_v3",
    }


def invalid_token_finding(**overrides) -> Dict[str, Any]:
    finding = token_finding()
    finding.update(overrides)
    return finding


def sequence_support(
    validity: str = "VALID_NON_LEAK",
    used: bool = False,
) -> Dict[str, Any]:
    return {
        "status": "OK",
        "validity": validity,
        "label": "LEAK" if validity == "VALID_LEAK" else "NON_LEAK",
        "prediction": 1 if validity == "VALID_LEAK" else 0,
        "confidence": 0.8,
        "probability_leak": 0.8 if validity == "VALID_LEAK" else 0.2,
        "threshold": 0.3,
        "used_for_final_label": used,
        "used_for_risk_level": used,
    }


def decision(
    label: str,
    evidence_sources: List[str] | None = None,
    validity: str = "VALID_NON_LEAK",
    sequence_used: bool = False,
) -> Dict[str, Any]:
    return {
        "final_label": label,
        "risk_level": "MEDIUM" if label == "LEAK" else "LOW",
        "confidence_value": None,
        "confidence_source": "none",
        "reasons": [],
        "evidence_sources": evidence_sources or [],
        "analysis_status": "COMPLETE",
        "regex_types": [],
        "token_types": [],
        "sequence_support": sequence_support(validity, sequence_used),
        "model_errors": [],
    }


def sentence_item(
    sentence: str,
    *,
    page: int = 1,
    label: str = "LEAK",
    regex_results: List[Any] | None = None,
    token_results: List[Dict[str, Any]] | None = None,
    token_status: str = "OK",
    evidence_sources: List[str] | None = None,
    sequence_validity: str = "VALID_NON_LEAK",
    sequence_used: bool = False,
) -> Dict[str, Any]:
    return {
        "page": page,
        "sentence": sentence,
        "regex_results": regex_results if regex_results is not None else [],
        "token_analysis": token_analysis(token_results, token_status),
        "sequence_analysis": {
            "status": "OK",
            "result": None,
            "source": "hebert_sequence_classifier_v3",
            "error": None,
        },
        "decision": decision(label, evidence_sources, sequence_validity, sequence_used),
        "model_errors": [],
    }


def sentence_item_without_page(sentence: str, **kwargs) -> Dict[str, Any]:
    item = sentence_item(sentence, **kwargs)
    item.pop("page")
    return item


def page_item_without_page(sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"sentences": sentences}


def page_item(page: int, sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"page": page, "sentences": sentences}


def run_report(tmp_path: Path, pdf_pages: List[List[str]], pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_pdf = make_pdf(tmp_path / "input.pdf", pdf_pages)
    results_json = write_results(tmp_path / "results.json", pages)
    output_pdf = tmp_path / "marked.pdf"
    return create_colored_pdf_report(str(input_pdf), str(results_json), str(output_pdf))


def test_merge_sentence_rects_by_line_merges_many_fragments_on_one_line():
    rects = [fitz.Rect(x, 10, x + 4, 20) for x in range(100, 50, -5)]

    merged = pdf_colored_report._merge_sentence_rects_by_line(rects)

    assert len(merged) == 1
    assert merged[0] == fitz.Rect(55, 10, 104, 20)


def test_merge_sentence_rects_by_line_supports_rtl_order_with_min_max_x():
    rects = [
        fitz.Rect(180, 20, 200, 30),
        fitz.Rect(130, 20.4, 160, 30.4),
        fitz.Rect(90, 20.2, 120, 30.2),
    ]

    merged = pdf_colored_report._merge_sentence_rects_by_line(rects)

    assert len(merged) == 1
    assert merged[0].x0 == 90
    assert merged[0].x1 == 200


def test_merge_sentence_rects_by_line_keeps_two_lines_separate():
    rects = [
        fitz.Rect(10, 10, 20, 20),
        fitz.Rect(25, 10, 35, 20),
        fitz.Rect(10, 30, 20, 40),
        fitz.Rect(25, 30, 35, 40),
    ]

    merged = pdf_colored_report._merge_sentence_rects_by_line(rects)

    assert len(merged) == 2


def test_merge_sentence_rects_by_line_keeps_three_lines_separate():
    rects = [
        fitz.Rect(10, 10, 20, 20),
        fitz.Rect(10, 26, 20, 36),
        fitz.Rect(10, 42, 20, 52),
    ]

    merged = pdf_colored_report._merge_sentence_rects_by_line(rects)

    assert len(merged) == 3


def test_merge_sentence_rects_by_line_allows_small_vertical_drift():
    rects = [
        fitz.Rect(10, 10, 20, 20),
        fitz.Rect(25, 11, 35, 21),
        fitz.Rect(40, 9.5, 50, 19.5),
    ]

    merged = pdf_colored_report._merge_sentence_rects_by_line(rects)

    assert len(merged) == 1


def test_merge_sentence_rects_by_line_rejects_large_vertical_gap():
    rects = [
        fitz.Rect(10, 10, 20, 20),
        fitz.Rect(25, 24, 35, 34),
    ]

    merged = pdf_colored_report._merge_sentence_rects_by_line(rects)

    assert len(merged) == 2


def test_merge_sentence_rects_by_line_handles_single_and_empty_lists():
    rect = fitz.Rect(1, 2, 3, 4)

    assert pdf_colored_report._merge_sentence_rects_by_line([rect]) == [rect]
    assert pdf_colored_report._merge_sentence_rects_by_line([]) == []


def test_json_v2_valid_loads_and_output_pdf_created(tmp_path):
    summary = run_report(
        tmp_path,
        [["No leaks here"]],
        [page_item(1, [sentence_item("No leaks here", label="NON_LEAK")])],
    )

    assert summary["schema_version"] == "2.0"
    assert Path(summary["output_pdf"]).exists()


@pytest.mark.parametrize(
    "payload, error",
    [
        ([], "root"),
        ({"schema_version": "1.0", "pages": []}, "schema_version"),
        ({"schema_version": "2.0"}, "pages"),
        ({"schema_version": "2.0", "pages": {}}, "pages"),
        ({"schema_version": "2.0", "pages": [{"page": 1, "sentences": {}}]}, "sentences"),
    ],
)
def test_invalid_json_schema_raises_value_error(tmp_path, payload, error):
    input_pdf = make_pdf(tmp_path / "input.pdf", [["text"]])
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        create_colored_pdf_report(str(input_pdf), str(results_json), str(tmp_path / "out.pdf"))


def test_non_leak_and_undetermined_create_no_annotations(tmp_path):
    summary = run_report(
        tmp_path,
        [["regex secret@example.com", "token sentence"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "regex secret@example.com",
                        label="NON_LEAK",
                        regex_results=[regex_finding()],
                    ),
                    sentence_item(
                        "token sentence",
                        label="UNDETERMINED",
                        token_results=[token_finding("token")],
                    ),
                ],
            )
        ],
    )

    assert summary["marked_regex_rects"] == 0
    assert summary["marked_sentence_rects"] == 0
    assert count_annotations(Path(summary["output_pdf"])) == 0


def test_regex_only_marks_value_not_sentence(tmp_path):
    summary = run_report(
        tmp_path,
        [["The email is secret@example.com in this sentence"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "The email is secret@example.com in this sentence",
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    )
                ],
            )
        ],
    )

    assert summary["leak_sentences"] == 1
    assert summary["regex_findings"] == 1
    assert summary["marked_regex_rects"] == 1
    assert summary["marked_sentence_rects"] == 0
    assert count_annotations(Path(summary["output_pdf"])) == 1


def test_token_only_marks_sentence_not_token_value(tmp_path):
    summary = run_report(
        tmp_path,
        [["This full token sentence contains token"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "This full token sentence contains token",
                        token_results=[token_finding("token")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["hebert_token_sentences"] == 1
    assert summary["hebert_sentence_markings"] == 1
    assert summary["marked_regex_rects"] == 0
    assert summary["marked_sentence_rects"] == 1
    assert count_annotations(Path(summary["output_pdf"])) == 1


def test_sequence_only_marks_sentence(tmp_path):
    summary = run_report(
        tmp_path,
        [["Sequence classifier marks this whole sentence"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Sequence classifier marks this whole sentence",
                        evidence_sources=["hebert_sequence"],
                        sequence_validity="VALID_LEAK",
                        sequence_used=True,
                    )
                ],
            )
        ],
    )

    assert summary["hebert_sequence_sentences"] == 1
    assert summary["marked_sentence_rects"] == 1


def test_token_and_sequence_mark_sentence_once(tmp_path):
    summary = run_report(
        tmp_path,
        [["Both models mark this sentence once"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Both models mark this sentence once",
                        token_results=[token_finding("models")],
                        evidence_sources=["hebert_token", "hebert_sequence"],
                        sequence_validity="VALID_LEAK",
                        sequence_used=True,
                    )
                ],
            )
        ],
    )

    assert summary["hebert_token_sentences"] == 1
    assert summary["hebert_sequence_sentences"] == 1
    assert summary["hebert_sentence_markings"] == 1
    assert summary["marked_sentence_rects"] == 1
    assert count_annotations(Path(summary["output_pdf"])) == 1


@pytest.mark.parametrize(
    "token_results, evidence_sources, sequence_validity, sequence_used",
    [
        ([token_finding("token")], ["regex", "hebert_token"], "VALID_NON_LEAK", False),
        ([], ["regex", "hebert_sequence"], "VALID_LEAK", True),
        ([token_finding("token")], ["regex", "hebert_token", "hebert_sequence"], "VALID_LEAK", True),
    ],
)
def test_regex_and_hebert_mark_value_and_sentence_once(
    tmp_path,
    token_results,
    evidence_sources,
    sequence_validity,
    sequence_used,
):
    summary = run_report(
        tmp_path,
        [["Combined sentence contains secret@example.com and token"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Combined sentence contains secret@example.com and token",
                        regex_results=[regex_finding()],
                        token_results=token_results,
                        evidence_sources=evidence_sources,
                        sequence_validity=sequence_validity,
                        sequence_used=sequence_used,
                    )
                ],
            )
        ],
    )

    assert summary["marked_regex_rects"] == 1
    assert summary["marked_sentence_rects"] == 1
    assert count_annotations(Path(summary["output_pdf"])) == 2


def test_regex_and_hebert_annotations_are_highlights_with_distinct_opacity(tmp_path):
    summary = run_report(
        tmp_path,
        [["Combined sentence contains secret@example.com and token"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Combined sentence contains secret@example.com and token",
                        regex_results=[regex_finding()],
                        token_results=[token_finding("token")],
                        evidence_sources=["regex", "hebert_token"],
                    )
                ],
            )
        ],
    )

    details = load_annotation_details(Path(summary["output_pdf"]))
    opacities = sorted(round(item["opacity"], 2) for item in details)

    assert [item["type"] for item in details] == ["Highlight", "Highlight"]
    assert "Square" not in {item["type"] for item in details}
    assert "Underline" not in {item["type"] for item in details}
    assert opacities[0] == pytest.approx(pdf_colored_report.HEBERT_HIGHLIGHT_OPACITY, abs=0.02)
    assert opacities[1] == pytest.approx(pdf_colored_report.REGEX_HIGHLIGHT_OPACITY, abs=0.02)
    assert pdf_colored_report.REGEX_HIGHLIGHT_OPACITY > pdf_colored_report.HEBERT_HIGHLIGHT_OPACITY


def test_hebert_sentence_fragments_are_merged_by_line_before_annotation(tmp_path, monkeypatch):
    raw_rects = [
        fitz.Rect(10, 10, 14, 20),
        fitz.Rect(16, 10.2, 22, 20.2),
        fitz.Rect(24, 9.8, 30, 19.8),
        fitz.Rect(32, 10, 38, 20),
        fitz.Rect(40, 10.3, 46, 20.3),
        fitz.Rect(10, 32, 16, 42),
        fitz.Rect(18, 32.1, 24, 42.1),
        fitz.Rect(26, 31.8, 32, 41.8),
        fitz.Rect(34, 32, 40, 42),
        fitz.Rect(42, 32.2, 48, 42.2),
    ]
    calls = 0

    def fake_search_sentence(page, sentence):
        nonlocal calls
        calls += 1
        return raw_rects

    monkeypatch.setattr(pdf_colored_report, "_search_sentence_on_page", fake_search_sentence)

    summary = run_report(
        tmp_path,
        [["A sentence that will be located by a mocked search"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "A sentence that will be located by a mocked search",
                        token_results=[token_finding("located")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert calls == 1
    assert summary["raw_sentence_rects"] == 10
    assert summary["merged_sentence_rects"] == 2
    assert summary["sentence_rects_merged"] == 8
    assert summary["marked_sentence_rects"] == 2
    assert count_annotations(Path(summary["output_pdf"])) == 2
    assert {item["type"] for item in load_annotation_details(Path(summary["output_pdf"]))} == {"Highlight"}


def test_token_and_sequence_share_one_sentence_search_and_merge(tmp_path, monkeypatch):
    calls = 0

    def fake_search_sentence(page, sentence):
        nonlocal calls
        calls += 1
        return [
            fitz.Rect(10, 10, 14, 20),
            fitz.Rect(16, 10, 20, 20),
            fitz.Rect(10, 32, 14, 42),
            fitz.Rect(16, 32, 20, 42),
        ]

    monkeypatch.setattr(pdf_colored_report, "_search_sentence_on_page", fake_search_sentence)

    summary = run_report(
        tmp_path,
        [["Both models mark this sentence once"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Both models mark this sentence once",
                        token_results=[token_finding("models")],
                        evidence_sources=["hebert_token", "hebert_sequence"],
                        sequence_validity="VALID_LEAK",
                        sequence_used=True,
                    )
                ],
            )
        ],
    )

    assert calls == 1
    assert summary["raw_sentence_rects"] == 4
    assert summary["merged_sentence_rects"] == 2
    assert summary["marked_sentence_rects"] == 2
    assert count_annotations(Path(summary["output_pdf"])) == 2


def test_two_token_findings_same_sentence_mark_once(tmp_path):
    summary = run_report(
        tmp_path,
        [["Token sentence has first and second findings"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Token sentence has first and second findings",
                        token_results=[token_finding("first"), token_finding("second")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["hebert_sentence_markings"] == 1
    assert summary["marked_sentence_rects"] == 1


def test_multiple_regex_findings_are_marked_separately(tmp_path):
    summary = run_report(
        tmp_path,
        [["Emails a@example.com and b@example.com are present"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Emails a@example.com and b@example.com are present",
                        regex_results=[
                            regex_finding("a@example.com"),
                            regex_finding("b@example.com"),
                        ],
                        evidence_sources=["regex"],
                    )
                ],
            )
        ],
    )

    assert summary["regex_findings"] == 2
    assert summary["marked_regex_rects"] == 2


def test_duplicate_regex_rect_is_skipped(tmp_path):
    summary = run_report(
        tmp_path,
        [["Duplicate secret@example.com appears once"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Duplicate secret@example.com appears once",
                        regex_results=[regex_finding(), regex_finding()],
                        evidence_sources=["regex"],
                    )
                ],
            )
        ],
    )

    assert summary["regex_findings"] == 2
    assert summary["marked_regex_rects"] == 1
    assert summary["duplicate_regex_rects_skipped"] == 1


def test_token_finding_non_leak_is_ignored(tmp_path):
    summary = run_report(
        tmp_path,
        [["Token exists but final decision is clean"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Token exists but final decision is clean",
                        label="NON_LEAK",
                        token_results=[token_finding("Token")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["marked_sentence_rects"] == 0
    assert count_annotations(Path(summary["output_pdf"])) == 0


@pytest.mark.parametrize(
    "token_results",
    [
        [{}],
        [invalid_token_finding(type="")],
        [invalid_token_finding(value="")],
        [invalid_token_finding(score=None)],
        [invalid_token_finding(score=float("nan"))],
        [invalid_token_finding(score=float("inf"))],
        [invalid_token_finding(score=1.5)],
        [invalid_token_finding(start=True)],
        [invalid_token_finding(end=True)],
        [invalid_token_finding(start=5, end=5)],
    ],
)
def test_invalid_token_findings_do_not_create_evidence(tmp_path, token_results):
    summary = run_report(
        tmp_path,
        [["Invalid token should not mark sentence"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Invalid token should not mark sentence",
                        token_results=token_results,
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["hebert_token_sentences"] == 0
    assert summary["marked_sentence_rects"] == 0
    assert summary["invalid_token_findings"] == 1


def test_token_results_not_list_does_not_create_evidence(tmp_path):
    item = sentence_item(
        "Token results object should not mark",
        token_results=[],
        evidence_sources=["hebert_token"],
    )
    item["token_analysis"]["results"] = {}

    summary = run_report(tmp_path, [["Token results object should not mark"]], [page_item(1, [item])])

    assert summary["hebert_token_sentences"] == 0
    assert summary["marked_sentence_rects"] == 0


@pytest.mark.parametrize("missing_key", ["type", "value", "score", "start", "end"])
def test_token_finding_missing_required_field_is_invalid(tmp_path, missing_key):
    finding = token_finding("required")
    finding.pop(missing_key)

    summary = run_report(
        tmp_path,
        [["Missing token field should not mark"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Missing token field should not mark",
                        token_results=[finding],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["hebert_token_sentences"] == 0
    assert summary["marked_sentence_rects"] == 0
    assert summary["invalid_token_findings"] == 1


def test_one_valid_token_finding_is_enough_for_sentence_marking(tmp_path):
    summary = run_report(
        tmp_path,
        [["One valid token is enough"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "One valid token is enough",
                        token_results=[{}, token_finding("valid")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["invalid_token_findings"] == 1
    assert summary["hebert_token_sentences"] == 1
    assert summary["marked_sentence_rects"] == 1


def test_invalid_sequence_does_not_mark_sentence(tmp_path):
    summary = run_report(
        tmp_path,
        [["Invalid sequence should not mark sentence"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Invalid sequence should not mark sentence",
                        sequence_validity="INVALID_RESULT",
                        sequence_used=False,
                    )
                ],
            )
        ],
    )

    assert summary["hebert_sequence_sentences"] == 0
    assert summary["marked_sentence_rects"] == 0


def test_valid_sequence_support_without_evidence_source_marks_sentence(tmp_path):
    summary = run_report(
        tmp_path,
        [["Valid sequence support marks this sentence"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Valid sequence support marks this sentence",
                        evidence_sources=[],
                        sequence_validity="VALID_LEAK",
                        sequence_used=True,
                    )
                ],
            )
        ],
    )

    assert summary["hebert_sequence_sentences"] == 1
    assert summary["marked_sentence_rects"] == 1


@pytest.mark.parametrize("validity", ["INVALID_RESULT", "VALID_NON_LEAK", "ERROR", "NOT_RUN"])
def test_conflicting_sequence_evidence_source_does_not_mark(tmp_path, validity):
    summary = run_report(
        tmp_path,
        [["Conflicting sequence should not mark"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Conflicting sequence should not mark",
                        evidence_sources=["hebert_sequence"],
                        sequence_validity=validity,
                        sequence_used=False,
                    )
                ],
            )
        ],
    )

    assert summary["hebert_sequence_sentences"] == 0
    assert summary["marked_sentence_rects"] == 0
    assert summary["warnings"]


def test_valid_leak_sequence_not_used_for_final_label_does_not_mark(tmp_path):
    summary = run_report(
        tmp_path,
        [["Unused valid leak sequence should not mark"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Unused valid leak sequence should not mark",
                        evidence_sources=[],
                        sequence_validity="VALID_LEAK",
                        sequence_used=False,
                    )
                ],
            )
        ],
    )

    assert summary["hebert_sequence_sentences"] == 0
    assert summary["marked_sentence_rects"] == 0


def test_valid_token_still_marks_when_sequence_evidence_is_conflicting(tmp_path):
    summary = run_report(
        tmp_path,
        [["Token marks despite conflicting sequence"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Token marks despite conflicting sequence",
                        token_results=[token_finding("Token")],
                        evidence_sources=["hebert_token", "hebert_sequence"],
                        sequence_validity="INVALID_RESULT",
                        sequence_used=False,
                    )
                ],
            )
        ],
    )

    assert summary["hebert_token_sentences"] == 1
    assert summary["hebert_sequence_sentences"] == 0
    assert summary["marked_sentence_rects"] == 1
    assert summary["warnings"]


def test_sequence_probability_without_evidence_does_not_mark(tmp_path):
    item = sentence_item(
        "Probability alone must not mark",
        evidence_sources=[],
        sequence_validity="VALID_LEAK",
        sequence_used=False,
    )

    summary = run_report(tmp_path, [["Probability alone must not mark"]], [page_item(1, [item])])

    assert summary["hebert_sequence_sentences"] == 0
    assert summary["marked_sentence_rects"] == 0


def test_hebert_sentence_not_found_does_not_fallback_to_token_value(tmp_path):
    summary = run_report(
        tmp_path,
        [["The token value appears here but sentence does not"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "A completely different sentence with token",
                        token_results=[token_finding("token")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["not_found_sentences"] == 1
    assert summary["marked_sentence_rects"] == 0
    assert summary["marked_regex_rects"] == 0
    assert summary["warnings"]


def test_regex_value_not_found_and_invalid_findings_are_reported(tmp_path):
    summary = run_report(
        tmp_path,
        [["Only one real value is present"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Only one real value is present",
                        regex_results=[
                            regex_finding("missing@example.com"),
                            {"type": "BAD"},
                            "not a dict",
                            regex_finding("x"),
                        ],
                        evidence_sources=["regex"],
                    )
                ],
            )
        ],
    )

    assert summary["regex_findings"] == 1
    assert summary["not_found_regex_values"] == 1
    assert summary["invalid_regex_findings"] == 3


def test_page_one_maps_to_first_pdf_page_and_invalid_page_is_not_searched(tmp_path):
    summary = run_report(
        tmp_path,
        [["first secret@example.com"], ["second other@example.com"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "first secret@example.com",
                        page=1,
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    ),
                    sentence_item(
                        "second other@example.com",
                        page=99,
                        regex_results=[regex_finding("other@example.com")],
                        evidence_sources=["regex"],
                    ),
                ],
            )
        ],
    )

    assert summary["marked_regex_rects"] == 1
    assert summary["invalid_pages"] == 1
    assert load_annotations_by_page(Path(summary["output_pdf"])) == [1, 0]


@pytest.mark.parametrize(
    ("container_page", "sentence_page"),
    [(1, 2), (2, 1)],
)
def test_page_mismatch_is_reported_without_marking(tmp_path, container_page, sentence_page):
    summary = run_report(
        tmp_path,
        [["first secret@example.com"], ["second secret@example.com"]],
        [
            page_item(
                container_page,
                [
                    sentence_item(
                        "first secret@example.com",
                        page=sentence_page,
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    )
                ],
            )
        ],
    )

    assert summary["invalid_pages"] == 1
    assert summary["marked_regex_rects"] == 0
    assert summary["marked_sentence_rects"] == 0
    assert "Page mismatch" in summary["warnings"][0]
    assert summary["not_found_examples"][0]["reason"] == "page_mismatch"


def test_matching_container_and_sentence_page_marks(tmp_path):
    summary = run_report(
        tmp_path,
        [["matching secret@example.com"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "matching secret@example.com",
                        page=1,
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    )
                ],
            )
        ],
    )

    assert summary["invalid_pages"] == 0
    assert summary["marked_regex_rects"] == 1


def test_missing_sentence_page_uses_container_page(tmp_path):
    summary = run_report(
        tmp_path,
        [["container secret@example.com"]],
        [
            page_item(
                1,
                [
                    sentence_item_without_page(
                        "container secret@example.com",
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    )
                ],
            )
        ],
    )

    assert summary["marked_regex_rects"] == 1


def test_missing_container_page_uses_sentence_page(tmp_path):
    summary = run_report(
        tmp_path,
        [["sentence secret@example.com"]],
        [
            page_item_without_page(
                [
                    sentence_item(
                        "sentence secret@example.com",
                        page=1,
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    )
                ]
            )
        ],
    )

    assert summary["marked_regex_rects"] == 1


@pytest.mark.parametrize("bad_page", [True, False, 0, -1, "1", 1.0, None])
def test_invalid_page_values_are_rejected(tmp_path, bad_page):
    summary = run_report(
        tmp_path,
        [["bad page secret@example.com"]],
        [
            page_item_without_page(
                [
                    sentence_item(
                        "bad page secret@example.com",
                        page=bad_page,
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    )
                ]
            )
        ],
    )

    assert summary["invalid_pages"] == 1
    assert summary["marked_regex_rects"] == 0


def test_two_pages_mark_correct_pages_and_empty_page_is_ok(tmp_path):
    summary = run_report(
        tmp_path,
        [["first secret@example.com"], [], ["third other@example.com"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "first secret@example.com",
                        page=1,
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    )
                ],
            ),
            page_item(2, []),
            page_item(
                3,
                [
                    sentence_item(
                        "third other@example.com",
                        page=3,
                        regex_results=[regex_finding("other@example.com")],
                        evidence_sources=["regex"],
                    )
                ],
            ),
        ],
    )

    assert summary["marked_regex_rects"] == 2
    assert load_annotations_by_page(Path(summary["output_pdf"])) == [1, 0, 1]


def test_whitespace_normalized_sentence_can_be_found(tmp_path):
    summary = run_report(
        tmp_path,
        [["Whitespace sentence can be found"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Whitespace   sentence\ncan be found",
                        token_results=[token_finding("Whitespace")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["marked_sentence_rects"] == 1


def test_line_break_sentence_not_found_is_reported_without_wrong_marking(tmp_path):
    summary = run_report(
        tmp_path,
        [["Line break part one"], ["part two is separate"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Line break part one part two is separate",
                        token_results=[token_finding("part")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["marked_sentence_rects"] == 0
    assert summary["not_found_sentences"] == 1


def test_long_sentence_part_alone_is_not_used_as_fallback(tmp_path):
    summary = run_report(
        tmp_path,
        [["This long generic sentence part is present elsewhere"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "This long generic sentence part is present elsewhere but the full sentence is missing",
                        token_results=[token_finding("generic")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["marked_sentence_rects"] == 0
    assert summary["not_found_sentences"] == 1


def test_general_part_elsewhere_is_not_marked_as_sentence(tmp_path):
    summary = run_report(
        tmp_path,
        [["The document contains sensitive information"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "The document contains sensitive information and more text that is absent",
                        token_results=[token_finding("sensitive")],
                        evidence_sources=["hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["marked_sentence_rects"] == 0
    assert summary["not_found_sentences"] == 1


def test_ambiguous_regex_and_sentence_matches_are_counted(tmp_path):
    summary = run_report(
        tmp_path,
        [["repeat@example.com repeat@example.com"], ["Same sentence", "Same sentence"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "repeat@example.com repeat@example.com",
                        regex_results=[regex_finding("repeat@example.com")],
                        evidence_sources=["regex"],
                    ),
                ],
            ),
            page_item(
                2,
                [
                    sentence_item(
                        "Same sentence",
                        page=2,
                        token_results=[token_finding("Same")],
                        evidence_sources=["hebert_token"],
                    ),
                ],
            )
        ],
    )

    assert summary["ambiguous_regex_matches"] == 1
    assert summary["ambiguous_sentence_matches"] == 1


def test_summary_counts_are_consistent_with_pdf_annotations(tmp_path):
    summary = run_report(
        tmp_path,
        [["Combined secret@example.com token sentence"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Combined secret@example.com token sentence",
                        regex_results=[regex_finding()],
                        token_results=[token_finding("token")],
                        evidence_sources=["regex", "hebert_token"],
                    )
                ],
            )
        ],
    )

    assert summary["leak_sentences"] == 1
    assert summary["marked_regex_rects"] + summary["marked_sentence_rects"] == count_annotations(
        Path(summary["output_pdf"])
    )


def test_output_pdf_can_be_opened(tmp_path):
    summary = run_report(
        tmp_path,
        [["Openable output secret@example.com"]],
        [
            page_item(
                1,
                [
                    sentence_item(
                        "Openable output secret@example.com",
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    )
                ],
            )
        ],
    )

    doc = fitz.open(summary["output_pdf"])
    try:
        assert len(doc) == 1
    finally:
        doc.close()


def test_input_and_output_same_path_raise_value_error(tmp_path):
    input_pdf = make_pdf(tmp_path / "input.pdf", [["text"]])
    results_json = write_results(tmp_path / "results.json", [])

    with pytest.raises(ValueError, match="different"):
        create_colored_pdf_report(str(input_pdf), str(results_json), str(input_pdf))

    doc = fitz.open(str(input_pdf))
    try:
        assert len(doc) == 1
    finally:
        doc.close()


def test_resolved_same_input_and_output_path_raise_value_error(tmp_path):
    input_pdf = make_pdf(tmp_path / "input.pdf", [["text"]])
    results_json = write_results(tmp_path / "results.json", [])
    output_pdf = tmp_path / "nested" / ".." / "input.pdf"

    with pytest.raises(ValueError, match="different"):
        create_colored_pdf_report(str(input_pdf), str(results_json), str(output_pdf))


class FakeAnnot:
    def __init__(self, *, update_error: Exception | None = None):
        self.update_error = update_error

    def set_colors(self, stroke):
        return None

    def set_opacity(self, opacity):
        return None

    def update(self):
        if self.update_error:
            raise self.update_error


class FakePage:
    def __init__(self, *, annotation_error: Exception | None = None):
        self.annotation_error = annotation_error

    def search_for(self, text):
        return [fitz.Rect(1, 1, 10, 10)]

    def add_highlight_annot(self, rect):
        return FakeAnnot(update_error=self.annotation_error)

    def add_rect_annot(self, rect):
        return FakeAnnot(update_error=self.annotation_error)


class FakeDoc:
    def __init__(self, *, save_error: Exception | None = None, annotation_error: Exception | None = None):
        self.closed = False
        self.save_error = save_error
        self.page = FakePage(annotation_error=annotation_error)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.page

    def save(self, *args, **kwargs):
        if self.save_error:
            raise self.save_error

    def close(self):
        self.closed = True


def test_document_is_closed_when_annotation_raises(tmp_path, monkeypatch):
    input_pdf = make_pdf(tmp_path / "input.pdf", [["secret@example.com"]])
    results_json = write_results(
        tmp_path / "results.json",
        [
            page_item(
                1,
                [
                    sentence_item(
                        "secret@example.com",
                        regex_results=[regex_finding()],
                        evidence_sources=["regex"],
                    )
                ],
            )
        ],
    )
    fake_doc = FakeDoc(annotation_error=RuntimeError("annotation failed"))
    monkeypatch.setattr(pdf_colored_report.fitz, "open", lambda path: fake_doc)

    with pytest.raises(RuntimeError, match="annotation failed"):
        create_colored_pdf_report(str(input_pdf), str(results_json), str(tmp_path / "out.pdf"))

    assert fake_doc.closed is True


def test_document_is_closed_when_save_raises(tmp_path, monkeypatch):
    input_pdf = make_pdf(tmp_path / "input.pdf", [["text"]])
    results_json = write_results(tmp_path / "results.json", [])
    fake_doc = FakeDoc(save_error=RuntimeError("save failed"))
    monkeypatch.setattr(pdf_colored_report.fitz, "open", lambda path: fake_doc)

    with pytest.raises(RuntimeError, match="save failed"):
        create_colored_pdf_report(str(input_pdf), str(results_json), str(tmp_path / "out.pdf"))

    assert fake_doc.closed is True


def test_no_dependency_on_legacy_fields(tmp_path):
    item = sentence_item(
        "Legacy fields should be ignored",
        label="NON_LEAK",
    )
    item["svm"] = {"label": "LEAK"}
    item["ml_result"] = {"label": "LEAK"}
    item["svm_result"] = {"label": "LEAK"}
    item["hebert_typed"] = [token_finding("Legacy")]
    item["hebert"] = [token_finding("Legacy")]

    summary = run_report(tmp_path, [["Legacy fields should be ignored"]], [page_item(1, [item])])

    assert summary["marked_regex_rects"] == 0
    assert summary["marked_sentence_rects"] == 0


def test_public_function_signature_is_preserved():
    signature = inspect.signature(create_colored_pdf_report)
    assert list(signature.parameters) == [
        "input_pdf_path",
        "results_json_path",
        "output_pdf_path",
    ]


def test_missing_files_raise_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_colored_pdf_report(
            str(tmp_path / "missing.pdf"),
            str(tmp_path / "missing.json"),
            str(tmp_path / "out.pdf"),
        )

    input_pdf = make_pdf(tmp_path / "input.pdf", [["text"]])
    with pytest.raises(FileNotFoundError):
        create_colored_pdf_report(
            str(input_pdf),
            str(tmp_path / "missing.json"),
            str(tmp_path / "out.pdf"),
        )
