from __future__ import annotations

import csv
import importlib
import json
from datetime import datetime
from typing import Any, Dict, List

import pytest


FORBIDDEN_FIELDS = {
    "svm",
    "svm_label",
    "svm_confidence",
    "svm_margin",
    "svm_source",
    "ml",
    "ml_label",
    "ml_result",
    "margin",
    "final_confidence",
}

CSV_HEADERS_V2 = [
    "schema_version",
    "page",
    "sentence",
    "analysis_status",
    "final_label",
    "risk_level",
    "confidence_value",
    "confidence_source",
    "reasons",
    "evidence_sources",
    "regex_types",
    "regex_values",
    "token_status",
    "token_types",
    "token_values",
    "token_scores",
    "sequence_status",
    "sequence_label",
    "sequence_prediction",
    "sequence_confidence",
    "sequence_probability_leak",
    "sequence_temperature",
    "sequence_threshold",
    "sequence_source",
    "sequence_validity",
    "model_errors",
]


@pytest.fixture()
def pipeline():
    return importlib.reload(importlib.import_module("analysis_pipeline"))


def token_analysis(status: str = "OK", spans: bool = False, error=None):
    return {
        "status": status,
        "results": [
            {
                "type": "TOKEN_SECRET",
                "value": "סוד",
                "score": 0.91,
                "start": 0,
                "end": 3,
                "source": "hebert_token_classifier_v3",
            }
        ]
        if spans
        else [],
        "source": "hebert_token_classifier_v3",
        "error": error,
    }


def sequence_analysis(
    status: str = "OK",
    label: str = "NON_LEAK",
    prediction: int = 0,
    probability: float = 0.2,
    threshold: float = 0.3,
    confidence: float = 0.8,
    temperature: float = 8.06174373626709,
    error=None,
):
    result = None
    if status == "OK":
        result = {
            "label": label,
            "prediction": prediction,
            "confidence": confidence,
            "probability_leak": probability,
            "temperature": temperature,
            "threshold": threshold,
            "source": "hebert_sequence_classifier_v3",
        }

    return {
        "status": status,
        "result": result,
        "source": "hebert_sequence_classifier_v3",
        "error": error,
    }


def ai_analysis(
    status: str = "COMPLETE",
    token=None,
    sequence=None,
    errors=None,
):
    return {
        "analysis_status": status,
        "token_analysis": token or token_analysis(),
        "sequence_analysis": sequence or sequence_analysis(),
        "model_errors": errors or [],
    }


def decision(
    label: str = "NON_LEAK",
    risk: str = "LOW",
    status: str = "COMPLETE",
    evidence=None,
    errors=None,
    sequence_validity: str = "VALID_NON_LEAK",
):
    return {
        "final_label": label,
        "risk_level": risk,
        "confidence_value": 0.8 if label != "UNDETERMINED" else None,
        "confidence_source": "hebert_sequence" if label != "UNDETERMINED" else "none",
        "reasons": ["סיבה, עם פסיק"],
        "evidence_sources": evidence or [],
        "analysis_status": status,
        "regex_types": ["EMAIL"] if "regex" in (evidence or []) else [],
        "token_types": ["TOKEN_SECRET"] if "hebert_token" in (evidence or []) else [],
        "sequence_support": {
            "status": "OK",
            "validity": sequence_validity,
            "label": "LEAK" if "hebert_sequence" in (evidence or []) else "NON_LEAK",
            "prediction": 1 if "hebert_sequence" in (evidence or []) else 0,
            "confidence": 0.8,
            "probability_leak": 0.8,
            "threshold": 0.3,
            "used_for_final_label": "hebert_sequence" in (evidence or []),
            "used_for_risk_level": "hebert_sequence" in (evidence or []),
        },
        "model_errors": errors or [],
    }


def install_pipeline_mocks(
    monkeypatch,
    pipeline,
    tmp_path,
    *,
    pages,
    ai_by_sentence,
    decision_by_sentence,
    regex_sentences=None,
):
    calls = {"analyzer": [], "decision": []}
    regex_sentences = set(regex_sentences or [])
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_text("not a real pdf", encoding="utf-8")
    output_json = tmp_path / "results.json"
    output_csv = tmp_path / "results.csv"

    monkeypatch.setattr(pipeline, "extract_text", lambda path, debug=False: pages)
    monkeypatch.setattr(pipeline, "normalize_text", lambda text: text)
    monkeypatch.setattr(
        pipeline,
        "split_document",
        lambda text: [part for part in text.split("|") if part],
    )

    def fake_regex(sentence):
        if sentence in regex_sentences:
            return [
                {
                    "type": "EMAIL",
                    "value": "a,b@example.com",
                    "start": 0,
                    "end": 15,
                    "source": "regex",
                }
            ]
        return []

    def fake_analyzer(sentence):
        calls["analyzer"].append(sentence)
        return ai_by_sentence[sentence]

    def fake_decision(**kwargs):
        sentence = calls["analyzer"][-1]
        calls["decision"].append(kwargs)
        return decision_by_sentence[sentence]

    monkeypatch.setattr(pipeline, "detect_regex", fake_regex)
    monkeypatch.setattr(pipeline, "analyze_sensitive_text", fake_analyzer)
    monkeypatch.setattr(pipeline, "make_final_decision_v2", fake_decision)

    return input_pdf, output_json, output_csv, calls


def run_pipeline(
    monkeypatch,
    pipeline,
    tmp_path,
    *,
    pages,
    ai_by_sentence,
    decision_by_sentence,
    regex_sentences=None,
):
    input_pdf, output_json, output_csv, calls = install_pipeline_mocks(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=pages,
        ai_by_sentence=ai_by_sentence,
        decision_by_sentence=decision_by_sentence,
        regex_sentences=regex_sentences,
    )
    summary = pipeline.analyze_pdf(
        str(input_pdf),
        output_json=str(output_json),
        output_csv=str(output_csv),
    )
    with output_json.open("r", encoding="utf-8") as file:
        results_json = json.load(file)
    with output_csv.open("r", encoding="utf-8-sig", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    return summary, results_json, csv_rows, calls


def assert_no_forbidden_fields(value):
    if isinstance(value, dict):
        assert not (set(value) & FORBIDDEN_FIELDS)
        for item in value.values():
            assert_no_forbidden_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_forbidden_fields(item)


def assert_summary_invariants(summary: Dict[str, int]) -> None:
    assert summary["total_sentences"] == (
        summary["total_leaks"]
        + summary["total_non_leaks"]
        + summary["total_undetermined"]
    )
    assert summary["total_sentences"] == (
        summary["high_risk"]
        + summary["medium_risk"]
        + summary["low_risk"]
        + summary["unknown_risk"]
    )
    assert summary["total_sentences"] == (
        summary["complete_analyses"]
        + summary["partial_analyses"]
        + summary["failed_analyses"]
    )


def test_analyzer_called_once_per_sentence(pipeline, monkeypatch, tmp_path):
    sentences = ["one", "two", "three"]
    ai = {sentence: ai_analysis() for sentence in sentences}
    decisions = {sentence: decision() for sentence in sentences}

    _, _, _, calls = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "one|two"), (2, "three")],
        ai_by_sentence=ai,
        decision_by_sentence=decisions,
    )

    assert calls["analyzer"] == sentences
    assert len(calls["decision"]) == 3


def test_no_svm_or_direct_model_wrapper_runtime_imports(pipeline):
    assert not hasattr(pipeline, "predict_svm_final")
    assert not hasattr(pipeline, "detect_hebert_typed_leaks")
    assert not hasattr(pipeline, "predict_hebert_sequence")


def test_sentence_item_matches_schema_v2(pipeline, monkeypatch, tmp_path):
    summary, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "שלום")],
        ai_by_sentence={"שלום": ai_analysis()},
        decision_by_sentence={"שלום": decision()},
    )

    item = results["pages"][0]["sentences"][0]
    assert summary["schema_version"] == "2.0"
    assert set(item) == {
        "page",
        "sentence",
        "regex_results",
        "token_analysis",
        "sequence_analysis",
        "decision",
        "model_errors",
    }


def test_json_root_is_dict_and_schema_version_is_v2(pipeline, monkeypatch, tmp_path):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={"sentence": decision()},
    )

    assert isinstance(results, dict)
    assert results["schema_version"] == "2.0"


def test_metadata_contains_required_keys(pipeline, monkeypatch, tmp_path):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={"sentence": decision()},
    )

    assert set(results["metadata"]) == {
        "pipeline",
        "generated_at",
        "input_file",
        "multitask_model",
        "multitask_config",
        "model_architecture",
        "active_detection_layers",
        "sequence_temperature",
        "sequence_threshold",
        "analysis_status",
        "model_errors_count",
    }


def test_summary_counts_all_labels_risks_and_statuses(pipeline, monkeypatch, tmp_path):
    ai = {
        "leak": ai_analysis("COMPLETE"),
        "non": ai_analysis("PARTIAL"),
        "unknown": ai_analysis("FAILED"),
    }
    decisions = {
        "leak": decision("LEAK", "HIGH", "COMPLETE", ["hebert_sequence"]),
        "non": decision("NON_LEAK", "LOW", "PARTIAL"),
        "unknown": decision("UNDETERMINED", "UNKNOWN", "FAILED"),
    }

    summary, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "leak|non|unknown")],
        ai_by_sentence=ai,
        decision_by_sentence=decisions,
    )

    expected = {
        "total_pages": 1,
        "total_sentences": 3,
        "total_leaks": 1,
        "total_non_leaks": 1,
        "total_undetermined": 1,
        "high_risk": 1,
        "medium_risk": 0,
        "low_risk": 1,
        "unknown_risk": 1,
        "complete_analyses": 1,
        "partial_analyses": 1,
        "failed_analyses": 1,
        "model_errors_count": 0,
    }
    assert results["summary"] == expected
    assert_summary_invariants(results["summary"])
    assert summary["total_leaks"] == 1


def test_summary_fallbacks_for_unknown_values(pipeline, monkeypatch, tmp_path):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={
            "sentence": decision("MAYBE", "STRANGE", "BROKEN")
        },
    )

    summary = results["summary"]
    assert summary["total_sentences"] == 1
    assert summary["total_undetermined"] == 1
    assert summary["unknown_risk"] == 1
    assert summary["failed_analyses"] == 1
    assert_summary_invariants(summary)
    assert results["metadata"]["analysis_status"] == "FAILED"


def test_summary_fallbacks_for_missing_values(pipeline, monkeypatch, tmp_path):
    incomplete_decision = decision()
    incomplete_decision.pop("final_label")
    incomplete_decision.pop("risk_level")
    incomplete_decision.pop("analysis_status")

    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={"sentence": incomplete_decision},
    )

    summary = results["summary"]
    assert summary["total_undetermined"] == 1
    assert summary["unknown_risk"] == 1
    assert summary["failed_analyses"] == 1
    assert_summary_invariants(summary)
    assert results["metadata"]["analysis_status"] == "FAILED"


def test_summary_invariants_with_mixed_known_and_unknown_values(
    pipeline,
    monkeypatch,
    tmp_path,
):
    decisions = {
        "leak": decision("LEAK", "MEDIUM", "COMPLETE", ["hebert_token"]),
        "unknown_label": decision("ODD", "LOW", "PARTIAL"),
        "unknown_risk": decision("NON_LEAK", "ODD", "COMPLETE"),
        "unknown_status": decision("UNDETERMINED", "UNKNOWN", "ODD"),
    }

    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "leak|unknown_label|unknown_risk|unknown_status")],
        ai_by_sentence={sentence: ai_analysis() for sentence in decisions},
        decision_by_sentence=decisions,
    )

    assert results["summary"]["total_sentences"] == 4
    assert_summary_invariants(results["summary"])


def test_sequence_only_leak_is_preserved_in_sentence_decision(
    pipeline,
    monkeypatch,
    tmp_path,
):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sequence leak")],
        ai_by_sentence={
            "sequence leak": ai_analysis(
                sequence=sequence_analysis(
                    label="LEAK",
                    prediction=1,
                    probability=0.9,
                    threshold=0.3,
                )
            )
        },
        decision_by_sentence={
            "sequence leak": decision(
                "LEAK",
                "MEDIUM",
                "COMPLETE",
                ["hebert_sequence"],
                sequence_validity="VALID_LEAK",
            )
        },
    )

    item = results["pages"][0]["sentences"][0]
    assert item["decision"]["final_label"] == "LEAK"
    assert item["decision"]["evidence_sources"] == ["hebert_sequence"]
    assert item["regex_results"] == []
    assert item["token_analysis"]["results"] == []


def test_total_pages_includes_empty_page(pipeline, monkeypatch, tmp_path):
    _, results, _, calls = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "first"), (2, "")],
        ai_by_sentence={"first": ai_analysis()},
        decision_by_sentence={"first": decision()},
    )

    assert calls["analyzer"] == ["first"]
    assert results["summary"]["total_pages"] == 2
    assert len(results["pages"]) == 2
    assert results["pages"][1]["sentences"] == []


def test_page_and_sentence_order_is_preserved(pipeline, monkeypatch, tmp_path):
    _, results, _, calls = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "a|b"), (2, "c")],
        ai_by_sentence={sentence: ai_analysis() for sentence in ["a", "b", "c"]},
        decision_by_sentence={sentence: decision() for sentence in ["a", "b", "c"]},
    )

    assert calls["analyzer"] == ["a", "b", "c"]
    assert [page["page"] for page in results["pages"]] == [1, 2]
    assert [item["sentence"] for item in results["pages"][0]["sentences"]] == ["a", "b"]
    assert [item["sentence"] for item in results["pages"][1]["sentences"]] == ["c"]


def test_model_errors_are_merged_without_duplicates(pipeline, monkeypatch, tmp_path):
    err = {
        "component": "sequence",
        "source": "hebert_sequence_classifier_v3",
        "error_type": "INVALID_RESULT",
        "message": "bad result",
    }
    ai = {"sentence": ai_analysis("PARTIAL", errors=[err])}
    decisions = {"sentence": decision("UNDETERMINED", "UNKNOWN", "PARTIAL", errors=[err])}

    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence=ai,
        decision_by_sentence=decisions,
    )

    item = results["pages"][0]["sentences"][0]
    assert item["model_errors"] == [err]
    assert results["summary"]["model_errors_count"] == 1
    assert results["metadata"]["model_errors_count"] == 1


def test_csv_headers_match_schema_v2(pipeline, monkeypatch, tmp_path):
    _, _, rows, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={"sentence": decision()},
    )

    assert rows
    assert list(rows[0]) == CSV_HEADERS_V2


def test_no_svm_or_ml_fields_in_json_or_csv(pipeline, monkeypatch, tmp_path):
    _, results, rows, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={"sentence": decision()},
    )

    assert_no_forbidden_fields(results)
    assert not (set(rows[0]) & FORBIDDEN_FIELDS)


def test_csv_list_fields_are_json_strings(pipeline, monkeypatch, tmp_path):
    _, _, rows, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "regex")],
        ai_by_sentence={"regex": ai_analysis(token=token_analysis(spans=True))},
        decision_by_sentence={"regex": decision("LEAK", "HIGH", "COMPLETE", ["regex", "hebert_token"])},
        regex_sentences={"regex"},
    )

    row = rows[0]
    assert json.loads(row["reasons"]) == ["סיבה, עם פסיק"]
    assert json.loads(row["regex_values"]) == ["a,b@example.com"]
    assert json.loads(row["token_values"]) == ["סוד"]
    assert json.loads(row["evidence_sources"]) == ["regex", "hebert_token"]


def test_document_without_sentences_is_complete_with_empty_summary(
    pipeline,
    monkeypatch,
    tmp_path,
):
    summary, results, rows, calls = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "")],
        ai_by_sentence={},
        decision_by_sentence={},
    )

    assert calls["analyzer"] == []
    assert rows == []
    assert results["metadata"]["analysis_status"] == "COMPLETE"
    assert results["summary"]["total_sentences"] == 0
    assert summary["analysis_status"] == "COMPLETE"


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["COMPLETE", "COMPLETE"], "COMPLETE"),
        (["COMPLETE", "PARTIAL"], "PARTIAL"),
        (["COMPLETE", "FAILED"], "PARTIAL"),
        (["FAILED", "FAILED"], "FAILED"),
    ],
)
def test_document_analysis_status_rules(
    pipeline,
    monkeypatch,
    tmp_path,
    statuses,
    expected,
):
    sentences = [f"s{index}" for index in range(len(statuses))]
    ai = {sentence: ai_analysis(status) for sentence, status in zip(sentences, statuses)}
    decisions = {
        sentence: decision("NON_LEAK", "LOW", status)
        for sentence, status in zip(sentences, statuses)
    }

    summary, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "|".join(sentences))],
        ai_by_sentence=ai,
        decision_by_sentence=decisions,
    )

    assert results["metadata"]["analysis_status"] == expected
    assert summary["analysis_status"] == expected


def test_temperature_and_threshold_metadata_from_valid_sequence(
    pipeline,
    monkeypatch,
    tmp_path,
):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={"sentence": decision()},
    )

    assert results["metadata"]["sequence_temperature"] == 8.06174373626709
    assert results["metadata"]["sequence_threshold"] == 0.3


def test_temperature_and_threshold_metadata_from_valid_leak_sequence(
    pipeline,
    monkeypatch,
    tmp_path,
):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={
            "sentence": ai_analysis(
                sequence=sequence_analysis(
                    label="LEAK",
                    prediction=1,
                    probability=0.91,
                    threshold=0.3,
                    temperature=5.5,
                )
            )
        },
        decision_by_sentence={
            "sentence": decision(
                "LEAK",
                "MEDIUM",
                "COMPLETE",
                ["hebert_sequence"],
                sequence_validity="VALID_LEAK",
            )
        },
    )

    assert results["metadata"]["sequence_temperature"] == 5.5
    assert results["metadata"]["sequence_threshold"] == 0.3


def test_temperature_and_threshold_metadata_null_without_valid_sequence(
    pipeline,
    monkeypatch,
    tmp_path,
):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={
            "sentence": ai_analysis(
                "FAILED",
                sequence=sequence_analysis(status="ERROR", error="failed"),
            )
        },
        decision_by_sentence={
            "sentence": decision("UNDETERMINED", "UNKNOWN", "FAILED")
        },
    )

    assert results["metadata"]["sequence_temperature"] is None
    assert results["metadata"]["sequence_threshold"] is None


@pytest.mark.parametrize(
    ("sequence", "sequence_validity"),
    [
        (
            sequence_analysis(
                label="LEAK",
                prediction=1,
                probability=0.25,
                threshold=0.3,
            ),
            "INVALID_RESULT",
        ),
        (sequence_analysis(status="ERROR", error="failed"), "ERROR"),
        (sequence_analysis(status="NOT_RUN"), "NOT_RUN"),
        (sequence_analysis(temperature=float("nan")), "VALID_NON_LEAK"),
        (sequence_analysis(temperature=float("inf")), "VALID_NON_LEAK"),
        (sequence_analysis(temperature=0.0), "VALID_NON_LEAK"),
        (sequence_analysis(threshold=1.5), "VALID_NON_LEAK"),
        (sequence_analysis(threshold=True), "VALID_NON_LEAK"),
    ],
)
def test_sequence_metadata_ignores_invalid_or_unusable_results(
    pipeline,
    monkeypatch,
    tmp_path,
    sequence,
    sequence_validity,
):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis(sequence=sequence)},
        decision_by_sentence={
            "sentence": decision(sequence_validity=sequence_validity)
        },
    )

    assert results["metadata"]["sequence_temperature"] is None
    assert results["metadata"]["sequence_threshold"] is None


def test_sequence_metadata_uses_first_later_valid_result(
    pipeline,
    monkeypatch,
    tmp_path,
):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "invalid|valid")],
        ai_by_sentence={
            "invalid": ai_analysis(
                sequence=sequence_analysis(
                    label="LEAK",
                    prediction=1,
                    probability=0.2,
                    threshold=0.3,
                    temperature=9.9,
                )
            ),
            "valid": ai_analysis(
                sequence=sequence_analysis(
                    label="LEAK",
                    prediction=1,
                    probability=0.8,
                    threshold=0.4,
                    temperature=6.2,
                )
            ),
        },
        decision_by_sentence={
            "invalid": decision("UNDETERMINED", "UNKNOWN", "PARTIAL", sequence_validity="INVALID_RESULT"),
            "valid": decision(
                "LEAK",
                "MEDIUM",
                "COMPLETE",
                ["hebert_sequence"],
                sequence_validity="VALID_LEAK",
            ),
        },
    )

    assert results["metadata"]["sequence_temperature"] == 6.2
    assert results["metadata"]["sequence_threshold"] == 0.4


def test_metadata_model_paths_use_environment_overrides(
    pipeline,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HEBERT_MULTITASK_MODEL_PATH", "custom/multitask/model")
    monkeypatch.setenv("HEBERT_MULTITASK_CONFIG_PATH", "custom/production.json")

    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={"sentence": decision()},
    )

    assert results["metadata"]["multitask_model"] == "custom/multitask/model"
    assert results["metadata"]["multitask_config"] == "custom/production.json"
    assert results["metadata"]["active_detection_layers"] == ["regex", "hebert_multitask_v2"]


def test_metadata_model_paths_use_defaults_without_environment(
    pipeline,
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("HEBERT_MULTITASK_MODEL_PATH", raising=False)
    monkeypatch.delenv("HEBERT_MULTITASK_CONFIG_PATH", raising=False)

    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={"sentence": decision()},
    )

    assert results["metadata"]["multitask_model"] == "models/hebert_multitask_v2/best"
    assert results["metadata"]["multitask_config"] == "config/hebert_multitask_v2_production.json"
    assert results["metadata"]["active_detection_layers"] == ["regex", "hebert_multitask_v2"]


def test_generated_at_is_iso8601_parseable(pipeline, monkeypatch, tmp_path):
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis()},
        decision_by_sentence={"sentence": decision()},
    )

    datetime.fromisoformat(results["metadata"]["generated_at"])


def test_token_error_is_preserved(pipeline, monkeypatch, tmp_path):
    err = {
        "component": "token",
        "source": "hebert_token_classifier_v3",
        "error_type": "RuntimeError",
        "message": "token failed",
    }
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={
            "sentence": ai_analysis(
                "PARTIAL",
                token=token_analysis("ERROR", error="token failed"),
                errors=[err],
            )
        },
        decision_by_sentence={"sentence": decision("NON_LEAK", "LOW", "PARTIAL")},
    )

    assert results["pages"][0]["sentences"][0]["model_errors"] == [err]


def test_sequence_error_is_preserved(pipeline, monkeypatch, tmp_path):
    err = {
        "component": "sequence",
        "source": "hebert_sequence_classifier_v3",
        "error_type": "RuntimeError",
        "message": "sequence failed",
    }
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={
            "sentence": ai_analysis(
                "PARTIAL",
                sequence=sequence_analysis("ERROR", error="sequence failed"),
                errors=[err],
            )
        },
        decision_by_sentence={"sentence": decision("NON_LEAK", "LOW", "PARTIAL")},
    )

    assert results["pages"][0]["sentences"][0]["model_errors"] == [err]


def test_invalid_sequence_error_from_decision_is_preserved(
    pipeline,
    monkeypatch,
    tmp_path,
):
    err = {
        "component": "sequence",
        "source": "hebert_sequence_classifier_v3",
        "error_type": "INVALID_RESULT",
        "message": "invalid",
    }
    _, results, _, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "sentence")],
        ai_by_sentence={"sentence": ai_analysis("COMPLETE")},
        decision_by_sentence={
            "sentence": decision(
                "UNDETERMINED",
                "UNKNOWN",
                "PARTIAL",
                errors=[err],
                sequence_validity="INVALID_RESULT",
            )
        },
    )

    item = results["pages"][0]["sentences"][0]
    assert item["model_errors"] == [err]
    assert item["decision"]["sequence_support"]["validity"] == "INVALID_RESULT"


def test_hebrew_is_preserved_in_json_and_csv(pipeline, monkeypatch, tmp_path):
    _, results, rows, _ = run_pipeline(
        monkeypatch,
        pipeline,
        tmp_path,
        pages=[(1, "שלום סודי")],
        ai_by_sentence={
            "שלום סודי": ai_analysis(token=token_analysis(spans=True))
        },
        decision_by_sentence={
            "שלום סודי": decision("LEAK", "MEDIUM", "COMPLETE", ["hebert_token"])
        },
    )

    assert results["pages"][0]["sentences"][0]["sentence"] == "שלום סודי"
    assert rows[0]["sentence"] == "שלום סודי"
    assert json.loads(rows[0]["token_values"]) == ["סוד"]


def test_main_and_gui_import_after_pipeline_change():
    importlib.import_module("main")
    importlib.import_module("gui")
