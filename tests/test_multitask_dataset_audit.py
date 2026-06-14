from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from training.audit_multitask_datasets import (
    CSV_HEADERS,
    near_duplicate_candidates,
    normalize_sequence_label,
    normalize_text,
    run_audit,
)


def write_schema(data_dir: Path) -> None:
    (data_dir / "hebert_label_schema.json").write_text(
        json.dumps(
            {
                "labels": [
                    "O",
                    "B-SECRET_LEAK",
                    "I-SECRET_LEAK",
                    "B-PERSON_LEAK",
                    "I-PERSON_LEAK",
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict], bom: bool = False) -> None:
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding=encoding)


def write_raw(path: Path, text: str, bom: bool = False) -> None:
    path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")


def token_record(
    text: str,
    labels: list[str] | None = None,
    spans: list[dict] | None = None,
    group_id: str = "",
) -> dict:
    labels = labels if labels is not None else ["O"]
    return {
        "text": text,
        "tokens": text.split() or [text],
        "tag_names": labels,
        "typed_spans": spans if spans is not None else [],
        "group_id": group_id,
        "category": "token_cat",
    }


def sequence_record(text: str, label=1, group_id: str = "", category: str = "seq_cat") -> dict:
    return {
        "text": text,
        "label": label,
        "label_name": "LEAK" if label in (1, "1", "LEAK") else "NON_LEAK" if label in (0, "0", "NON_LEAK") else label,
        "group_id": group_id,
        "category": category,
    }


def write_csv_dataset(path: Path, rows: list[dict]) -> None:
    headers = ["text", "label", "category", "case_id", "expected_reason"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def run_small_audit(tmp_path: Path) -> tuple[Path, dict]:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    summary = run_audit(data_dir, out_dir)
    return out_dir, summary


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_exact_duplicate_inside_file_and_between_train_test(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(
        data_dir / "hebert_sequence_train.jsonl",
        [sequence_record("same text", 1), sequence_record("same text", 1)],
    )
    write_jsonl(data_dir / "hebert_sequence_test.jsonl", [sequence_record("same text", 1)])

    summary = run_audit(data_dir, out_dir)
    duplicates = read_csv_rows(out_dir / "exact_duplicates.csv")
    leakage = read_csv_rows(out_dir / "split_leakage.csv")

    assert summary["exact_duplicates"] >= 1
    assert duplicates
    assert leakage


def test_normalized_duplicate_and_unicode_nfc(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("שלום\t עולם", 0)])
    write_jsonl(data_dir / "hebert_sequence_test.jsonl", [sequence_record("שלום עולם", 0)])
    assert normalize_text("e\u0301") == "é"

    run_audit(data_dir, out_dir)

    assert read_csv_rows(out_dir / "normalized_duplicates.csv")


def test_sequence_and_token_label_conflicts(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(
        data_dir / "hebert_sequence_train.jsonl",
        [sequence_record("conflict text", 1), sequence_record("conflict text", 0)],
    )
    write_jsonl(
        data_dir / "hebert_token_classification_train_v3_targeted.jsonl",
        [
            token_record("token conflict", ["B-SECRET_LEAK"], [{"start": 0, "end": 5, "text": "token", "type": "SECRET_LEAK"}]),
            token_record("token conflict", ["O"], []),
        ],
    )

    run_audit(data_dir, out_dir)
    conflicts = read_csv_rows(out_dir / "label_conflicts.csv")

    assert {row["conflict_type"] for row in conflicts} >= {"SEQUENCE_LABEL_CONFLICT", "TOKEN_LABEL_CONFLICT"}


def test_group_leakage_and_protected_detection(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("train", 1, group_id="g1")])
    write_jsonl(data_dir / "hebert_sequence_calibration_v3.jsonl", [sequence_record("calib", 1, group_id="g1")])

    run_audit(data_dir, out_dir)
    inventory = json.loads((out_dir / "dataset_inventory.json").read_text(encoding="utf-8"))

    assert read_csv_rows(out_dir / "group_leakage.csv")
    assert next(item for item in inventory if item["path"].endswith("calibration_v3.jsonl"))["protected"] is True


@pytest.mark.parametrize(
    "record,error_type",
    [
        ({}, "MissingText"),
        ({"text": ""}, "EmptyText"),
        ({"text": "bad label", "label": 7}, "InvalidSequenceLabel"),
    ],
)
def test_malformed_text_and_sequence_label_are_reported(tmp_path, record, error_type):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [record])

    run_audit(data_dir, out_dir)

    assert any(row["error_type"] == error_type for row in read_csv_rows(out_dir / "malformed_records.csv"))


def test_malformed_jsonl_and_empty_file_are_handled(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_raw(data_dir / "hebert_sequence_train.jsonl", '{"text": "ok", "label": 1}\n{bad json')
    write_raw(data_dir / "hebert_sequence_validation.jsonl", "")

    summary = run_audit(data_dir, out_dir)

    assert summary["malformed_records"] >= 1
    assert (out_dir / "dataset_inventory.json").exists()


@pytest.mark.parametrize(
    "record,error_prefix",
    [
        (token_record("a b", ["I-SECRET_LEAK", "O"]), "illegal_i_without_b"),
        (token_record("a", ["B-UNKNOWN"]), "unknown_label"),
        (token_record("abc", ["B-SECRET_LEAK"], [{"start": 0, "end": 9, "text": "abc", "type": "SECRET_LEAK"}]), "span_out_of_range"),
        (token_record("abc", ["B-SECRET_LEAK"], [{"start": 2, "end": 2, "text": "", "type": "SECRET_LEAK"}]), "span_end_not_after_start"),
        (token_record("abc", ["B-SECRET_LEAK"], [{"start": 0, "end": 2, "text": "zz", "type": "SECRET_LEAK"}]), "span_value_mismatch"),
    ],
)
def test_token_bio_schema_and_span_errors(tmp_path, record, error_prefix):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(data_dir / "hebert_token_classification_train_v3_targeted.jsonl", [record])

    run_audit(data_dir, out_dir)

    assert any(row["error_type"] == error_prefix for row in read_csv_rows(out_dir / "malformed_records.csv"))


def test_cross_task_overlap_classifications(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(
        data_dir / "hebert_token_classification_train_v3_targeted.jsonl",
        [
            token_record("consistent", ["B-SECRET_LEAK"], [{"start": 0, "end": 10, "text": "consistent", "type": "SECRET_LEAK"}]),
            token_record("possible", ["B-SECRET_LEAK"], [{"start": 0, "end": 8, "text": "possible", "type": "SECRET_LEAK"}]),
            token_record("review", ["B-SECRET_LEAK"], [{"start": 0, "end": 6, "text": "review", "type": "SECRET_LEAK"}]),
            token_record("review", ["O"], []),
        ],
    )
    write_jsonl(
        data_dir / "hebert_sequence_train.jsonl",
        [sequence_record("consistent", 1), sequence_record("possible", 0), sequence_record("review", 1)],
    )

    run_audit(data_dir, out_dir)
    classes = {row["text_preview"]: row["classification"] for row in read_csv_rows(out_dir / "cross_task_overlaps.csv")}

    assert classes["consistent"] == "CONSISTENT"
    assert classes["possible"] == "POSSIBLY_VALID_DIFFERENCE"
    assert classes["review"] == "CONFLICT_REVIEW_REQUIRED"


def test_manifest_keeps_protected_sets_out_of_train(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(data_dir / "hebert_token_classification_test_v3_targeted.jsonl", [token_record("test")])
    write_jsonl(data_dir / "hebert_sequence_calibration_v3.jsonl", [sequence_record("calib", 1)])
    write_csv_dataset(data_dir / "svm_blind_test_v6.csv", [{"text": "blind", "label": 1, "category": "c", "case_id": "b", "expected_reason": "protected"}])
    write_jsonl(data_dir / "hebert_targeted_challenge_v3.jsonl", [sequence_record("challenge", 1)])

    run_audit(data_dir, out_dir)
    manifest = json.loads((out_dir / "proposed_splits_manifest.json").read_text(encoding="utf-8"))
    train_sources = manifest["multitask_train"]["token_sources"] + manifest["multitask_train"]["sequence_sources"]

    assert not any("test" in source or "calibration" in source or "blind" in source or "challenge" in source for source in train_sources)


def test_bom_and_utf8_jsonl_load_and_hebrew_is_not_escaped(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("שלום", "LEAK")], bom=True)
    write_jsonl(data_dir / "hebert_sequence_validation.jsonl", [sequence_record("עולם", "NON_LEAK")])

    run_audit(data_dir, out_dir)
    text = (out_dir / "dataset_inventory.json").read_text(encoding="utf-8")

    assert "שלום" in text
    assert "\\u05" not in text


def test_all_output_files_and_headers_are_created(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("hello", 1)])

    run_audit(data_dir, out_dir)

    for filename, headers in CSV_HEADERS.items():
        assert (out_dir / filename).exists()
        with (out_dir / filename).open("r", encoding="utf-8-sig", newline="") as file:
            assert next(csv.reader(file)) == headers
    assert (out_dir / "dataset_inventory.json").exists()
    assert (out_dir / "proposed_splits_manifest.json").exists()
    assert (out_dir / "audit_summary.txt").exists()


def test_source_datasets_are_not_modified(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    source = data_dir / "hebert_sequence_train.jsonl"
    write_jsonl(source, [sequence_record("immutable", 1)])
    before = source.read_bytes()

    run_audit(data_dir, out_dir)

    assert source.read_bytes() == before


def test_near_duplicate_output_is_limited_and_blocked(tmp_path):
    from training.audit_multitask_datasets import AuditRecord

    train = [
        AuditRecord(
            file="train",
            line_number=index,
            task="sequence",
            split="train",
            role="MULTITASK_TRAIN_SEQUENCE_SOURCE",
            protected=False,
            text=f"abcdefghij {index} sensitive sentence",
            normalized_text=f"abcdefghij {index} sensitive sentence",
            exact_hash=str(index),
            normalized_hash=f"t{index}",
            label="LEAK",
        )
        for index in range(30)
    ]
    protected = [
        AuditRecord(
            file="test",
            line_number=index,
            task="sequence",
            split="test",
            role="INTERNAL_TEST_ONLY",
            protected=True,
            text=f"abcdefghij {index} sensitive sentenca",
            normalized_text=f"abcdefghij {index} sensitive sentenca",
            exact_hash=f"p{index}",
            normalized_hash=f"p{index}",
            label="LEAK",
        )
        for index in range(30)
    ]

    rows, comparisons = near_duplicate_candidates(train + protected)

    assert len(rows) <= 100
    assert comparisons < 30 * 30


def test_malformed_csv_row_and_csv_blind_test_load(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_raw(data_dir / "svm_blind_test_v6.csv", "text,label,category\nשלום,1,c,extra\n")

    run_audit(data_dir, out_dir)
    inventory = json.loads((out_dir / "dataset_inventory.json").read_text(encoding="utf-8"))

    assert any(item["path"].endswith("svm_blind_test_v6.csv") for item in inventory)
    assert any(row["error_type"] == "MalformedCSVRow" for row in read_csv_rows(out_dir / "malformed_records.csv"))


def test_label_normalization_and_task_detection_by_structure(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(data_dir / "custom_unknown_name.jsonl", [token_record("token by structure", ["O"])])
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("one", "1"), sequence_record("zero", "0")])

    assert normalize_sequence_label("1") == "LEAK"
    assert normalize_sequence_label("NON_LEAK") == "NON_LEAK"
    summary = run_audit(data_dir, out_dir)

    assert summary["sequence_records"] == 2


def test_unknown_role_is_not_train(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    write_schema(data_dir)
    write_jsonl(data_dir / "hebert_sequence_extra_unknown.jsonl", [sequence_record("unknown", 1)])

    run_audit(data_dir, out_dir)
    manifest = json.loads((out_dir / "proposed_splits_manifest.json").read_text(encoding="utf-8"))

    assert manifest["multitask_train"]["sequence_sources"] == []
    assert manifest["excluded_or_unknown"]
