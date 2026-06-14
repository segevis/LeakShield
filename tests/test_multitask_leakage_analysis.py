from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from training.analyze_multitask_leakage import (
    ACTIVE_PROTECTED,
    TOKEN_TRAIN,
    TOKEN_VALIDATION,
    cross_task_classification,
    option_a_projection,
    option_b_projection,
    run_analysis,
)
from training.audit_multitask_datasets import AuditRecord


def write_schema(data_dir: Path) -> None:
    (data_dir / "hebert_label_schema.json").write_text(
        json.dumps({"labels": ["O", "B-SECRET_LEAK", "I-SECRET_LEAK"]}),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")


def token_record(text: str, group_id: str = "", label: bool = True) -> dict:
    labels = ["B-SECRET_LEAK"] if label else ["O"]
    span = [{"start": 0, "end": min(4, len(text)), "text": text[: min(4, len(text))], "type": "SECRET_LEAK"}] if label else []
    return {"text": text, "tokens": [text], "tag_names": labels, "typed_spans": span, "group_id": group_id}


def sequence_record(text: str, label: str = "LEAK", group_id: str = "") -> dict:
    return {"text": text, "label": label, "label_name": label, "group_id": group_id}


def write_csv_dataset(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_schema(data_dir)
    return data_dir


def run_small(tmp_path: Path) -> tuple[Path, dict]:
    out_dir = tmp_path / "out"
    summary = run_analysis(tmp_path / "audit", tmp_path / "data", out_dir)
    return out_dir, summary


def audit_record(file: str, task: str, split: str, text: str, label: str, group_id: str = "", entity_count: int = 0) -> AuditRecord:
    from training.audit_multitask_datasets import normalize_text, stable_hash

    norm = normalize_text(text)
    return AuditRecord(
        file=file,
        line_number=1,
        task=task,
        split=split,
        role="",
        protected=file in ACTIVE_PROTECTED,
        text=text,
        normalized_text=norm,
        exact_hash=stable_hash(text),
        normalized_hash=stable_hash(norm),
        label=label,
        group_id=group_id,
        entity_count=entity_count,
        entity_types=("SECRET_LEAK",) if entity_count else (),
    )


def test_historical_overlap_is_not_active_blocker(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_token_classification_train.jsonl", [token_record("same", "g-old")])
    write_jsonl(data_dir / "hebert_token_classification_train_v3_targeted.jsonl", [token_record("same", "g-new")])

    out_dir, summary = run_small(tmp_path)
    rows = read_csv_rows(out_dir / "leakage_by_file_pair.csv")

    assert any(row["severity"] == "HISTORICAL_ONLY" for row in rows)
    assert summary["active_summary"]["active_train_to_protected_leakage"] == 0


def test_train_to_protected_exact_overlap_is_high_or_critical(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("secret", "LEAK")])
    write_jsonl(data_dir / "hebert_sequence_calibration_v3.jsonl", [sequence_record("secret", "LEAK")])

    out_dir, summary = run_small(tmp_path)
    rows = read_csv_rows(out_dir / "leakage_by_file_pair.csv")

    pair = next(row for row in rows if "calibration" in row["file_a"] or "calibration" in row["file_b"])
    assert pair["severity"] == "CRITICAL"
    assert pair["blocker"] == "True"
    assert summary["active_summary"]["active_train_to_protected_leakage"] > 0


def test_train_to_protected_normalized_overlap_is_detected(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("same\t text", "LEAK")])
    write_jsonl(data_dir / "hebert_sequence_test.jsonl", [sequence_record("same text", "LEAK")])

    out_dir, _ = run_small(tmp_path)
    protected_rows = read_csv_rows(out_dir / "protected_overlap_records.csv")

    assert protected_rows
    assert protected_rows[0]["overlap_type"] == "NORMALIZED_TEXT"


def test_train_to_validation_is_medium_and_train_duplicate_low(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("dup", "LEAK"), sequence_record("dup", "LEAK")])
    write_jsonl(data_dir / "hebert_sequence_validation.jsonl", [sequence_record("dup", "LEAK")])

    out_dir, _ = run_small(tmp_path)
    rows = read_csv_rows(out_dir / "leakage_by_file_pair.csv")

    assert any(row["severity"] == "MEDIUM" for row in rows)
    assert any(row["severity"] == "LOW" and row["file_a"] == row["file_b"] for row in rows)


def test_group_overlap_without_text_overlap_is_reported_separately(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("train text", "LEAK", "g1")])
    write_jsonl(data_dir / "hebert_sequence_test.jsonl", [sequence_record("test text", "LEAK", "g1")])

    out_dir, _ = run_small(tmp_path)
    group_rows = read_csv_rows(out_dir / "group_overlap_summary.csv")

    assert group_rows
    assert int(group_rows[0]["shared_group_count"]) == 1


def test_cross_task_safe_both_and_split_conflict():
    token_train = audit_record(TOKEN_TRAIN, "token", "train", "same", "LEAK", entity_count=1)
    sequence_train = audit_record("data/hebert_sequence_train.jsonl", "sequence", "train", "same", "LEAK")
    sequence_test = audit_record("data/hebert_sequence_test.jsonl", "sequence", "test", "same", "LEAK")

    assert cross_task_classification(token_train, sequence_train) == "SAFE_BOTH_TASKS"
    assert cross_task_classification(token_train, sequence_test) == "SPLIT_CONFLICT"


def test_option_a_removes_protected_overlaps_and_keeps_protected_unchanged():
    train = audit_record("data/hebert_sequence_train.jsonl", "sequence", "train", "secret", "LEAK", "g1")
    protected = audit_record("data/hebert_sequence_test.jsonl", "sequence", "test", "secret", "LEAK", "g1")
    safe = audit_record("data/hebert_sequence_train.jsonl", "sequence", "train", "safe", "NON_LEAK", "g2")

    projection = option_a_projection([train, protected, safe])

    assert projection["train_records"] == 1
    assert projection["protected_sets_unchanged"]
    assert projection["writes_datasets"] is False


def test_option_b_splits_by_group_and_keeps_keys_in_one_split():
    records = [
        audit_record("data/hebert_sequence_train.jsonl", "sequence", "train", f"text {index}", "LEAK", f"g{index // 2}")
        for index in range(10)
    ]

    projection = option_b_projection(records)

    assert projection["train_records"] + projection["validation_records"] == 10
    assert projection["train_groups"] + projection["validation_groups"] == 5


def test_calibration_and_blind_never_enter_train(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("train", "LEAK")])
    write_jsonl(data_dir / "hebert_sequence_calibration_v3.jsonl", [sequence_record("calib", "LEAK")])
    write_csv_dataset(data_dir / "svm_blind_test_v6.csv", [{"text": "blind", "label": "LEAK"}])

    out_dir, _ = run_small(tmp_path)
    plan = json.loads((out_dir / "recommended_clean_split_plan.json").read_text(encoding="utf-8"))

    assert all("calibration" not in path for path in plan["token_train_sources"] + plan["sequence_train_sources"])
    assert all("blind" not in path for path in plan["token_train_sources"] + plan["sequence_train_sources"])


def test_projection_outputs_do_not_create_datasets_and_sources_unchanged(tmp_path):
    data_dir = make_data_dir(tmp_path)
    source = data_dir / "hebert_sequence_train.jsonl"
    write_jsonl(source, [sequence_record("שלום", "LEAK")])
    before = source.read_bytes()

    out_dir, _ = run_small(tmp_path)

    assert source.read_bytes() == before
    assert not any(path.name.startswith("multitask_") and path.suffix == ".jsonl" for path in out_dir.iterdir())


def test_outputs_have_no_stack_traces_and_keep_hebrew(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("שלום עולם", "LEAK")])

    out_dir, _ = run_small(tmp_path)
    summary = (out_dir / "leakage_analysis_summary.txt").read_text(encoding="utf-8")
    plan_text = (out_dir / "recommended_clean_split_plan.json").read_text(encoding="utf-8")

    assert "Traceback" not in summary
    assert "\\u05" not in plan_text


def test_recommended_plan_safe_false_with_blocker_and_true_after_rules(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("secret", "LEAK", "g1")])
    write_jsonl(data_dir / "hebert_sequence_test.jsonl", [sequence_record("secret", "LEAK", "g1")])

    out_dir, summary = run_small(tmp_path)
    plan = json.loads((out_dir / "recommended_clean_split_plan.json").read_text(encoding="utf-8"))

    assert summary["active_summary"]["active_train_to_protected_leakage"] > 0
    assert plan["safe_to_build"] is True


def test_label_balance_and_token_type_distribution_are_computed(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("a", "LEAK"), sequence_record("b", "NON_LEAK")])
    write_jsonl(data_dir / "hebert_token_classification_train_v3_targeted.jsonl", [token_record("secret")])

    out_dir, _ = run_small(tmp_path)
    option = json.loads((out_dir / "option_a_projection.json").read_text(encoding="utf-8"))

    assert option["sequence_label_balance"]["LEAK"] == 1
    assert option["sequence_label_balance"]["NON_LEAK"] == 1
    assert option["token_entity_type_distribution"]["SECRET_LEAK"] == 1


def test_bool_group_malformed_values_and_empty_dataset(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [{"text": "x", "label": "LEAK", "group_id": True}])
    write_jsonl(data_dir / "hebert_sequence_validation.jsonl", [])

    out_dir, summary = run_small(tmp_path)

    assert summary["records_loaded"] == 1
    assert (out_dir / "leakage_by_file_pair.csv").exists()


@pytest.mark.parametrize(
    "filename",
    [
        "hebert_sequence_calibration_v3.jsonl",
        "hebert_sequence_test.jsonl",
        "hebert_sequence_development_v3.jsonl",
        "hebert_sequence_real_world_test_v3.jsonl",
        "hebert_targeted_challenge_v3.jsonl",
        "svm_blind_test_v6.csv",
        "real_world_definition_challenge.csv",
    ],
)
def test_protected_sources_are_classified_as_protected(tmp_path, filename):
    data_dir = make_data_dir(tmp_path)
    if filename.endswith(".csv"):
        write_csv_dataset(data_dir / filename, [{"text": "protected", "label": "LEAK"}])
    elif "challenge" in filename:
        write_jsonl(data_dir / filename, [token_record("protected challenge")])
    else:
        write_jsonl(data_dir / filename, [sequence_record("protected", "LEAK")])

    out_dir, _ = run_small(tmp_path)
    plan = json.loads((out_dir / "recommended_clean_split_plan.json").read_text(encoding="utf-8"))

    assert f"data/{filename}" in plan["protected_sets"]


@pytest.mark.parametrize(
    "train_name,validation_name",
    [
        ("hebert_token_classification_train_v3_targeted.jsonl", "hebert_token_classification_validation_v3_targeted.jsonl"),
        ("hebert_sequence_train.jsonl", "hebert_sequence_validation.jsonl"),
    ],
)
def test_train_validation_normalized_text_does_not_remain_in_option_a_train(tmp_path, train_name, validation_name):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / train_name, [token_record("same\ttext") if "token" in train_name else sequence_record("same\ttext")])
    write_jsonl(data_dir / validation_name, [token_record("same text") if "token" in validation_name else sequence_record("same text")])

    out_dir, _ = run_small(tmp_path)
    option = json.loads((out_dir / "option_a_projection.json").read_text(encoding="utf-8"))

    assert option["train_records"] == 0


@pytest.mark.parametrize(
    "text,label,expected",
    [
        ("secret", "LEAK", {"LEAK": 1}),
        ("public", "NON_LEAK", {"NON_LEAK": 1}),
    ],
)
def test_sequence_label_balance_single_label_cases(tmp_path, text, label, expected):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record(text, label)])

    out_dir, _ = run_small(tmp_path)
    option = json.loads((out_dir / "option_a_projection.json").read_text(encoding="utf-8"))

    assert option["sequence_label_balance"] == expected


def test_all_required_outputs_are_created(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("x", "LEAK")])

    out_dir, _ = run_small(tmp_path)

    expected = {
        "leakage_by_file_pair.csv",
        "active_leakage_summary.json",
        "historical_overlap_summary.json",
        "protected_overlap_records.csv",
        "group_overlap_summary.csv",
        "cross_task_overlap_classification.csv",
        "option_a_projection.json",
        "option_b_projection.json",
        "recommended_clean_split_plan.json",
        "leakage_analysis_summary.txt",
    }
    assert expected <= {path.name for path in out_dir.iterdir()}


def test_recommended_plan_safe_true_when_no_blockers(tmp_path):
    data_dir = make_data_dir(tmp_path)
    write_jsonl(data_dir / "hebert_sequence_train.jsonl", [sequence_record("train only", "LEAK")])

    out_dir, _ = run_small(tmp_path)
    plan = json.loads((out_dir / "recommended_clean_split_plan.json").read_text(encoding="utf-8"))

    assert plan["safe_to_build"] is True
