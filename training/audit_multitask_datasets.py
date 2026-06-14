from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROTECTED_KEYWORDS = (
    "test",
    "blind",
    "calibration",
    "development",
    "challenge",
    "real_world_test",
    "final_test",
    "hard_test",
)
MAX_PAIR_ROWS = 100_000
MAX_NEAR_DUPLICATE_COMPARISONS = 50_000

CSV_HEADERS = {
    "exact_duplicates.csv": [
        "normalized_hash",
        "text_preview",
        "file_a",
        "role_a",
        "split_a",
        "label_a",
        "file_b",
        "role_b",
        "split_b",
        "label_b",
        "conflict",
    ],
    "normalized_duplicates.csv": [
        "normalized_hash",
        "text_preview",
        "file_a",
        "role_a",
        "split_a",
        "label_a",
        "file_b",
        "role_b",
        "split_b",
        "label_b",
        "conflict",
    ],
    "split_leakage.csv": [
        "normalized_hash",
        "text_preview",
        "train_file",
        "protected_file",
        "train_label",
        "protected_label",
        "leakage_type",
        "group_id",
        "severity",
    ],
    "group_leakage.csv": ["group_id", "file_a", "split_a", "file_b", "split_b", "severity"],
    "label_conflicts.csv": [
        "task",
        "normalized_hash",
        "text_preview",
        "file_a",
        "label_a",
        "file_b",
        "label_b",
        "conflict_type",
    ],
    "cross_task_overlaps.csv": [
        "normalized_hash",
        "text_preview",
        "token_file",
        "token_split",
        "token_entity_count",
        "token_types",
        "sequence_file",
        "sequence_split",
        "sequence_label",
        "protected_overlap",
        "classification",
        "notes",
    ],
    "near_duplicate_candidates.csv": [
        "similarity_score",
        "text_a_preview",
        "file_a",
        "split_a",
        "text_b_preview",
        "file_b",
        "split_b",
        "protected_overlap",
    ],
    "token_label_distribution.csv": ["dataset", "label", "count", "percentage"],
    "token_entity_type_distribution.csv": ["dataset", "entity_type", "count", "percentage"],
    "sequence_label_distribution.csv": ["dataset", "label", "count", "percentage"],
    "dataset_length_statistics.csv": [
        "dataset",
        "records",
        "min_length",
        "max_length",
        "mean_length",
        "median_length",
        "p95_length",
        "p99_length",
    ],
    "malformed_records.csv": ["file", "line_number", "error_type", "error_message", "record_preview"],
}


@dataclass
class DatasetInfo:
    path: str
    task: str
    proposed_role: str
    format: str
    encoding: str = "utf-8-sig"
    records: int = 0
    labels: Dict[str, int] = field(default_factory=dict)
    groups: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    normalized_duplicate_count: int = 0
    malformed_count: int = 0
    empty_text_count: int = 0
    protected: bool = False
    notes: List[str] = field(default_factory=list)
    split: str = "unknown"
    fields: List[str] = field(default_factory=list)
    sample_text_preview: str = ""


@dataclass
class AuditRecord:
    file: str
    line_number: int
    task: str
    split: str
    role: str
    protected: bool
    text: str
    normalized_text: str
    exact_hash: str
    normalized_hash: str
    label: str
    group_id: str = ""
    category: str = ""
    fields: Tuple[str, ...] = ()
    entity_count: int = 0
    entity_types: Tuple[str, ...] = ()
    token_signature: str = ""


def normalize_text(text: str) -> str:
    """Return conservative normalized text for duplicate checks."""
    return " ".join(unicodedata.normalize("NFC", str(text)).replace("\n", " ").replace("\t", " ").split()).strip()


def stable_hash(text: str) -> str:
    """Return a stable SHA256 hash for report joins."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preview(text: str, limit: int = 120) -> str:
    """Return a short single-line text preview."""
    value = normalize_text(text)
    return value[:limit] + ("..." if len(value) > limit else "")


def read_label_schema(schema_path: Path) -> set[str]:
    """Load token labels from a schema file if present."""
    if not schema_path.exists():
        return {"O"}
    with schema_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    labels = data.get("labels") if isinstance(data, dict) else None
    return set(labels) if isinstance(labels, list) else {"O"}


def detect_task(record: Any) -> str:
    """Infer task from record structure before filename hints."""
    if not isinstance(record, dict):
        return "unknown"
    if any(key in record for key in ("tokens", "ner_tags", "tag_names", "typed_spans", "entities")):
        return "token"
    if "label" in record or "label_name" in record:
        return "sequence"
    return "unknown"


def infer_split(path: Path) -> str:
    """Infer split from filename."""
    name = path.name.lower()
    if "hard_train" in name:
        return "train"
    if "train" in name:
        return "train"
    if "validation" in name or "dev" in name and "development" not in name:
        return "validation"
    if "calibration" in name:
        return "calibration"
    if "development" in name:
        return "development"
    if "challenge" in name:
        return "challenge"
    if "blind" in name:
        return "blind"
    if "real_world_test" in name:
        return "real_world_test"
    if "test" in name:
        return "test"
    return "unknown"


def is_protected_dataset(path: Path, split: str) -> bool:
    """Return whether a dataset must be excluded from training."""
    name = path.name.lower()
    if "hard_train" in name:
        return False
    return split in PROTECTED_KEYWORDS or any(keyword in name for keyword in PROTECTED_KEYWORDS)


def propose_role(path: Path, task: str, split: str, protected: bool) -> str:
    """Propose a safe dataset role from known filenames and structure."""
    name = path.name
    if name == "hebert_token_classification_train_v3_targeted.jsonl":
        return "MULTITASK_TRAIN_TOKEN_SOURCE"
    if name == "hebert_token_classification_validation_v3_targeted.jsonl":
        return "MULTITASK_VALIDATION_TOKEN_SOURCE"
    if name == "hebert_token_classification_test_v3_targeted.jsonl":
        return "INTERNAL_TEST_ONLY"
    if name in {"hebert_sequence_train.jsonl", "hebert_sequence_hard_train_v3.jsonl"}:
        return "MULTITASK_TRAIN_SEQUENCE_SOURCE"
    if name == "hebert_sequence_validation.jsonl":
        return "MULTITASK_VALIDATION_SEQUENCE_SOURCE"
    if name == "hebert_sequence_calibration_v3.jsonl":
        return "CALIBRATION_ONLY"
    if name in {"svm_blind_test_v6.csv", "real_world_definition_challenge.csv"}:
        return "FINAL_BLIND_TEST_ONLY"
    if protected and split in {"test", "development"}:
        return "INTERNAL_TEST_ONLY"
    if protected and split in {"challenge", "real_world_test"}:
        return "HARD_TEST_ONLY"
    if protected:
        return "PROTECTED_EVALUATION"
    if task == "token" and name.startswith("hebert_token_classification_"):
        return "SUPERSEDED"
    if task == "sequence" and name.endswith("_v2.jsonl"):
        return "BASELINE_REFERENCE_ONLY"
    return "UNKNOWN"


def normalize_sequence_label(value: Any, label_name: Any = None) -> Optional[str]:
    """Normalize sequence label values for audit only."""
    if isinstance(label_name, str) and label_name.upper() in {"LEAK", "NON_LEAK"}:
        return label_name.upper()
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value == 1:
            return "LEAK"
        if value == 0:
            return "NON_LEAK"
    if isinstance(value, str):
        upper = value.strip().upper()
        if upper in {"LEAK", "NON_LEAK"}:
            return upper
        if upper == "1":
            return "LEAK"
        if upper == "0":
            return "NON_LEAK"
    return None


def entity_type_from_label(label: str) -> str:
    """Strip BIO prefix from a label."""
    if label == "O":
        return "O"
    return label[2:] if label.startswith(("B-", "I-")) else label


def validate_bio_labels(labels: Sequence[str], allowed_labels: set[str]) -> List[str]:
    """Validate BIO labels against a schema."""
    errors: List[str] = []
    previous_type = ""
    previous_was_entity = False
    for index, label in enumerate(labels):
        if label not in allowed_labels:
            errors.append(f"unknown_label:{label}@{index}")
        if label == "O":
            previous_type = ""
            previous_was_entity = False
            continue
        if not label.startswith(("B-", "I-")):
            errors.append(f"invalid_bio_label:{label}@{index}")
            previous_type = ""
            previous_was_entity = False
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "I" and (not previous_was_entity or previous_type != entity_type):
            errors.append(f"illegal_i_without_b:{label}@{index}")
        previous_type = entity_type
        previous_was_entity = True
    return errors


def validate_spans(record: Dict[str, Any], text: str) -> Tuple[List[str], List[str]]:
    """Validate span boundaries and return errors and entity types."""
    spans = record.get("typed_spans", record.get("entities", []))
    if spans is None:
        spans = []
    if not isinstance(spans, list):
        return ["spans_not_list"], []
    errors: List[str] = []
    ranges: List[Tuple[int, int]] = []
    types: List[str] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            errors.append(f"span_not_object:{index}")
            continue
        start = span.get("start")
        end = span.get("end")
        value = span.get("value", span.get("text"))
        entity_type = span.get("type", "")
        if isinstance(entity_type, str) and entity_type:
            types.append(entity_type)
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"span_bounds_not_int:{index}")
            continue
        if start < 0 or end > len(text):
            errors.append(f"span_out_of_range:{index}")
        if end <= start:
            errors.append(f"span_end_not_after_start:{index}")
        if isinstance(value, str) and 0 <= start < end <= len(text) and text[start:end] != value:
            errors.append(f"span_value_mismatch:{index}")
        ranges.append((start, end))
    for index, (start, end) in enumerate(sorted(ranges)):
        if index and start < sorted(ranges)[index - 1][1]:
            errors.append("overlapping_entities")
            break
    return errors, types


def iter_jsonl(path: Path) -> Tuple[List[Tuple[int, Any]], List[Dict[str, Any]]]:
    """Read JSONL records and collect malformed rows."""
    records: List[Tuple[int, Any]] = []
    malformed: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append((line_number, json.loads(raw)))
            except json.JSONDecodeError as exc:
                malformed.append(
                    {
                        "file": str(path),
                        "line_number": line_number,
                        "error_type": "JSONDecodeError",
                        "error_message": str(exc),
                        "record_preview": raw[:160],
                    }
                )
    return records, malformed


def iter_csv(path: Path) -> Tuple[List[Tuple[int, Any]], List[Dict[str, Any]]]:
    """Read CSV records and collect malformed rows."""
    records: List[Tuple[int, Any]] = []
    malformed: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        try:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                return records, malformed
            width = len(reader.fieldnames)
            for line_number, row in enumerate(reader, 2):
                if None in row or len(row) != width:
                    malformed.append(
                        {
                            "file": str(path),
                            "line_number": line_number,
                            "error_type": "MalformedCSVRow",
                            "error_message": "CSV row does not match header width.",
                            "record_preview": str(row)[:160],
                        }
                    )
                    continue
                records.append((line_number, row))
        except csv.Error as exc:
            malformed.append(
                {
                    "file": str(path),
                    "line_number": 0,
                    "error_type": "CSVError",
                    "error_message": str(exc),
                    "record_preview": "",
                }
            )
    return records, malformed


def list_datasets(data_dir: Path) -> List[Path]:
    """Return relevant token and sequence dataset files."""
    candidates: List[Path] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jsonl", ".csv"}:
            continue
        name = path.name.lower()
        if (
            "hebert_token_classification" in name
            or "hebert_sequence" in name
            or "hebert_targeted_challenge" in name
            or name in {"svm_blind_test_v6.csv", "real_world_definition_challenge.csv"}
        ):
            candidates.append(path)
    return candidates


def analyze_dataset(path: Path, data_dir: Path, allowed_labels: set[str]) -> Tuple[DatasetInfo, List[AuditRecord], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Analyze one dataset file."""
    rel = path.relative_to(data_dir.parent).as_posix()
    split = infer_split(path)
    raw_records, malformed = iter_csv(path) if path.suffix.lower() == ".csv" else iter_jsonl(path)
    task = "unknown"
    for _, record in raw_records:
        task = detect_task(record)
        if task != "unknown":
            break
    protected = is_protected_dataset(path, split)
    role = propose_role(path, task, split, protected)
    info = DatasetInfo(path=rel, task=task, proposed_role=role, format=path.suffix.lower().lstrip("."), split=split, protected=protected)
    info.malformed_count = len(malformed)
    audit_records: List[AuditRecord] = []
    malformed_extra: List[Dict[str, Any]] = []
    token_label_rows: List[Dict[str, Any]] = []
    token_entity_rows: List[Dict[str, Any]] = []
    sequence_label_rows: List[Dict[str, Any]] = []
    seen_exact: Dict[str, int] = {}
    seen_norm: Dict[str, int] = {}
    group_values: set[str] = set()
    category_counts: Dict[str, int] = {}
    field_names: set[str] = set()
    label_counts: Dict[str, int] = {}
    token_label_counts: Dict[str, int] = {}
    token_entity_counts: Dict[str, int] = {}
    sequence_counts: Dict[str, int] = {}

    for line_number, record in raw_records:
        info.records += 1
        if not isinstance(record, dict):
            malformed_extra.append(row_error(rel, line_number, "RecordNotObject", "Record is not a JSON/CSV object.", record))
            continue
        field_names.update(str(key) for key in record)
        text = record.get("text")
        if not isinstance(text, str):
            info.empty_text_count += 1
            malformed_extra.append(row_error(rel, line_number, "MissingText", "text is missing or not a string.", record))
            continue
        if not text.strip():
            info.empty_text_count += 1
            malformed_extra.append(row_error(rel, line_number, "EmptyText", "text is empty.", record))
        norm = normalize_text(text)
        if not info.sample_text_preview and norm:
            info.sample_text_preview = preview(norm)
        exact_h = stable_hash(text)
        norm_h = stable_hash(norm)
        seen_exact[exact_h] = seen_exact.get(exact_h, 0) + 1
        seen_norm[norm_h] = seen_norm.get(norm_h, 0) + 1
        group_id = str(record.get("group_id", record.get("pair_id", "")) or "")
        category = str(record.get("category", "") or "")
        if group_id:
            group_values.add(group_id)
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1

        entity_types: List[str] = []
        entity_count = 0
        label = ""
        token_signature = ""
        current_task = detect_task(record)
        if current_task == "token":
            labels = token_labels(record, allowed_labels)
            label_errors = validate_bio_labels(labels, allowed_labels)
            span_errors, span_types = validate_spans(record, text)
            for error in label_errors + span_errors:
                malformed_extra.append(row_error(rel, line_number, error.split(":")[0], error, record))
            for label_name in labels:
                token_label_counts[label_name] = token_label_counts.get(label_name, 0) + 1
            entity_types = sorted(set([entity_type_from_label(item) for item in labels if item != "O"] + span_types))
            for entity_type in entity_types:
                token_entity_counts[entity_type] = token_entity_counts.get(entity_type, 0) + 1
            entity_count = len(record.get("typed_spans", record.get("entities", [])) or [])
            label = "LEAK" if entity_types else "NON_LEAK"
            token_signature = json.dumps({"labels": labels, "spans": normalized_spans(record)}, ensure_ascii=False, sort_keys=True)
        elif current_task == "sequence":
            normalized_label = normalize_sequence_label(record.get("label"), record.get("label_name"))
            if normalized_label is None:
                malformed_extra.append(row_error(rel, line_number, "InvalidSequenceLabel", "Sequence label is not LEAK/NON_LEAK or 1/0.", record))
                label = "INVALID"
            else:
                label = normalized_label
                sequence_counts[label] = sequence_counts.get(label, 0) + 1
        else:
            malformed_extra.append(row_error(rel, line_number, "UnknownTask", "Could not infer token or sequence task.", record))
            label = "UNKNOWN"

        label_counts[label] = label_counts.get(label, 0) + 1
        audit_records.append(
            AuditRecord(
                file=rel,
                line_number=line_number,
                task=current_task,
                split=split,
                role=role,
                protected=protected,
                text=text,
                normalized_text=norm,
                exact_hash=exact_h,
                normalized_hash=norm_h,
                label=label,
                group_id=group_id,
                category=category,
                fields=tuple(sorted(str(key) for key in record)),
                entity_count=entity_count,
                entity_types=tuple(entity_types),
                token_signature=token_signature,
            )
        )

    info.duplicate_count = sum(count - 1 for count in seen_exact.values() if count > 1)
    info.normalized_duplicate_count = sum(count - 1 for count in seen_norm.values() if count > 1)
    info.groups = len(group_values)
    info.categories = category_counts
    info.fields = sorted(field_names)
    info.labels = label_counts
    info.malformed_count += len(malformed_extra)
    if split == "unknown":
        info.notes.append("Split could not be inferred from filename.")
    if task == "unknown":
        info.notes.append("Task could not be inferred from record structure.")
    if protected and role.startswith("MULTITASK_TRAIN"):
        info.notes.append("Protected dataset was not allowed into train.")

    token_label_rows.extend(distribution_rows(rel, token_label_counts, "label"))
    token_entity_rows.extend(distribution_rows(rel, token_entity_counts, "entity_type"))
    sequence_label_rows.extend(distribution_rows(rel, sequence_counts, "label"))
    return info, audit_records, malformed + malformed_extra, token_label_rows, token_entity_rows, sequence_label_rows


def token_labels(record: Dict[str, Any], allowed_labels: set[str]) -> List[str]:
    """Return token labels as label names."""
    tag_names = record.get("tag_names")
    if isinstance(tag_names, list):
        return [str(item) for item in tag_names]
    ner_tags = record.get("ner_tags")
    id_to_label = {index: label for index, label in enumerate(sorted(allowed_labels, key=lambda item: 0 if item == "O" else 1))}
    if isinstance(ner_tags, list):
        return [id_to_label.get(item, f"UNKNOWN_ID_{item}") if isinstance(item, int) else str(item) for item in ner_tags]
    return []


def normalized_spans(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return span fields used for token label conflict checks."""
    spans = record.get("typed_spans", record.get("entities", []))
    if not isinstance(spans, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for span in spans:
        if isinstance(span, dict):
            normalized.append(
                {
                    "start": span.get("start"),
                    "end": span.get("end"),
                    "type": span.get("type"),
                    "text": span.get("text", span.get("value")),
                }
            )
    return normalized


def row_error(file: str, line_number: int, error_type: str, message: str, record: Any) -> Dict[str, Any]:
    """Build a public malformed-record row."""
    return {
        "file": file,
        "line_number": line_number,
        "error_type": error_type,
        "error_message": message,
        "record_preview": preview(json.dumps(record, ensure_ascii=False) if not isinstance(record, str) else record),
    }


def distribution_rows(dataset: str, counts: Dict[str, int], field_name: str) -> List[Dict[str, Any]]:
    """Return distribution rows with percentages."""
    total = sum(counts.values())
    rows: List[Dict[str, Any]] = []
    for key, count in sorted(counts.items()):
        rows.append({"dataset": dataset, field_name: key, "count": count, "percentage": round((count / total * 100.0) if total else 0.0, 4)})
    return rows


def build_duplicate_rows(records: List[AuditRecord], use_normalized: bool, max_rows: int = MAX_PAIR_ROWS) -> List[Dict[str, Any]]:
    """Build duplicate rows from exact or normalized hash maps."""
    by_hash: Dict[str, List[AuditRecord]] = {}
    for record in records:
        key = record.normalized_hash if use_normalized else record.exact_hash
        by_hash.setdefault(key, []).append(record)
    rows: List[Dict[str, Any]] = []
    for key, items in by_hash.items():
        if len(items) < 2:
            continue
        for left, right in pair_limited(items):
            if len(rows) >= max_rows:
                return rows
            conflict = left.label != right.label or (left.task == "token" and right.task == "token" and left.token_signature != right.token_signature)
            rows.append(
                {
                    "normalized_hash": key,
                    "text_preview": preview(left.normalized_text),
                    "file_a": left.file,
                    "role_a": left.role,
                    "split_a": left.split,
                    "label_a": left.label,
                    "file_b": right.file,
                    "role_b": right.role,
                    "split_b": right.split,
                    "label_b": right.label,
                    "conflict": str(bool(conflict)),
                }
            )
    return rows


def pair_limited(items: List[AuditRecord], limit: int = 200) -> Iterable[Tuple[AuditRecord, AuditRecord]]:
    """Yield bounded pairs for report size safety."""
    produced = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]
            produced += 1
            if produced >= limit:
                return


def label_conflicts(records: List[AuditRecord], max_rows: int = MAX_PAIR_ROWS) -> List[Dict[str, Any]]:
    """Find same-task label conflicts by normalized text."""
    rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], List[AuditRecord]] = {}
    for record in records:
        by_key.setdefault((record.task, record.normalized_hash), []).append(record)
    for (task, key), items in by_key.items():
        labels = {(item.label, item.token_signature if task == "token" else "") for item in items}
        if len(labels) <= 1:
            continue
        for left, right in pair_limited(items, 50):
            conflict = left.label != right.label or (task == "token" and left.token_signature != right.token_signature)
            if conflict:
                if len(rows) >= max_rows:
                    return rows
                rows.append(
                    {
                        "task": task,
                        "normalized_hash": key,
                        "text_preview": preview(left.normalized_text),
                        "file_a": left.file,
                        "label_a": left.label,
                        "file_b": right.file,
                        "label_b": right.label,
                        "conflict_type": "TOKEN_LABEL_CONFLICT" if task == "token" else "SEQUENCE_LABEL_CONFLICT",
                    }
                )
    return rows


def split_leakage(records: List[AuditRecord], max_rows: int = MAX_PAIR_ROWS) -> List[Dict[str, Any]]:
    """Find train records that appear in protected sets."""
    train_by_hash: Dict[str, List[AuditRecord]] = {}
    protected_by_hash: Dict[str, List[AuditRecord]] = {}
    for record in records:
        if record.split == "train" and not record.protected:
            train_by_hash.setdefault(record.normalized_hash, []).append(record)
        if record.protected:
            protected_by_hash.setdefault(record.normalized_hash, []).append(record)
    rows: List[Dict[str, Any]] = []
    for key, train_items in train_by_hash.items():
        for train_record in train_items:
            for protected_record in protected_by_hash.get(key, []):
                if len(rows) >= max_rows:
                    return rows
                rows.append(
                    {
                        "normalized_hash": key,
                        "text_preview": preview(train_record.normalized_text),
                        "train_file": train_record.file,
                        "protected_file": protected_record.file,
                        "train_label": train_record.label,
                        "protected_label": protected_record.label,
                        "leakage_type": "NORMALIZED_TEXT",
                        "group_id": train_record.group_id or protected_record.group_id,
                        "severity": "HIGH",
                    }
                )
    return rows


def group_leakage(records: List[AuditRecord], max_rows: int = MAX_PAIR_ROWS) -> List[Dict[str, Any]]:
    """Find group IDs appearing across multiple splits."""
    by_group: Dict[str, List[AuditRecord]] = {}
    for record in records:
        if record.group_id:
            by_group.setdefault(record.group_id, []).append(record)
    rows: List[Dict[str, Any]] = []
    for group_id, items in by_group.items():
        splits = {item.split for item in items}
        if len(splits) <= 1:
            continue
        for left, right in pair_limited(items, 20):
            if left.split != right.split:
                if len(rows) >= max_rows:
                    return rows
                severity = "HIGH" if "train" in {left.split, right.split} and (left.protected or right.protected) else "MEDIUM"
                rows.append({"group_id": group_id, "file_a": left.file, "split_a": left.split, "file_b": right.file, "split_b": right.split, "severity": severity})
    return rows


def cross_task_overlaps(records: List[AuditRecord], conflict_hashes: set[str], max_rows: int = MAX_PAIR_ROWS) -> List[Dict[str, Any]]:
    """Find normalized text overlaps between token and sequence tasks."""
    token_by_hash: Dict[str, List[AuditRecord]] = {}
    sequence_by_hash: Dict[str, List[AuditRecord]] = {}
    for record in records:
        if record.task == "token":
            token_by_hash.setdefault(record.normalized_hash, []).append(record)
        elif record.task == "sequence":
            sequence_by_hash.setdefault(record.normalized_hash, []).append(record)
    rows: List[Dict[str, Any]] = []
    for key, token_items in token_by_hash.items():
        for token_record in token_items:
            for sequence_record in sequence_by_hash.get(key, []):
                if len(rows) >= max_rows:
                    return rows
                if key in conflict_hashes:
                    classification = "CONFLICT_REVIEW_REQUIRED"
                    notes = "Same text has conflicting labels within at least one task."
                elif token_record.entity_count > 0 and sequence_record.label == "LEAK":
                    classification = "CONSISTENT"
                    notes = "Token entities and sequence LEAK agree."
                elif token_record.entity_count == 0 and sequence_record.label == "NON_LEAK":
                    classification = "CONSISTENT"
                    notes = "No token entities and sequence NON_LEAK agree."
                else:
                    classification = "POSSIBLY_VALID_DIFFERENCE"
                    notes = "Task semantics may differ; manual review recommended."
                rows.append(
                    {
                        "normalized_hash": key,
                        "text_preview": preview(token_record.normalized_text),
                        "token_file": token_record.file,
                        "token_split": token_record.split,
                        "token_entity_count": token_record.entity_count,
                        "token_types": json.dumps(list(token_record.entity_types), ensure_ascii=False),
                        "sequence_file": sequence_record.file,
                        "sequence_split": sequence_record.split,
                        "sequence_label": sequence_record.label,
                        "protected_overlap": str(token_record.protected or sequence_record.protected),
                        "classification": classification,
                        "notes": notes,
                    }
                )
    return rows


def near_duplicate_candidates(records: List[AuditRecord], max_rows: int = 100, max_comparisons: int = MAX_NEAR_DUPLICATE_COMPARISONS) -> Tuple[List[Dict[str, Any]], int]:
    """Find bounded lexical near-duplicates between train and protected records."""
    train = [record for record in records if record.split == "train" and not record.protected and len(record.normalized_text) >= 20]
    protected = [record for record in records if record.protected and len(record.normalized_text) >= 20]
    buckets: Dict[Tuple[int, str], List[AuditRecord]] = {}
    for record in protected:
        key = (len(record.normalized_text) // 40, record.normalized_text[:16])
        buckets.setdefault(key, []).append(record)
    candidates: List[Dict[str, Any]] = []
    comparisons = 0
    for train_record in train:
        length_bucket = len(train_record.normalized_text) // 40
        for bucket in (length_bucket - 1, length_bucket, length_bucket + 1):
            for protected_record in buckets.get((bucket, train_record.normalized_text[:16]), []):
                if comparisons >= max_comparisons:
                    candidates.sort(key=lambda row: row["similarity_score"], reverse=True)
                    return candidates[:max_rows], comparisons
                comparisons += 1
                if train_record.normalized_hash == protected_record.normalized_hash:
                    continue
                score = SequenceMatcher(None, train_record.normalized_text, protected_record.normalized_text).ratio()
                if score >= 0.88:
                    candidates.append(
                        {
                            "similarity_score": round(score, 4),
                            "text_a_preview": preview(train_record.normalized_text),
                            "file_a": train_record.file,
                            "split_a": train_record.split,
                            "text_b_preview": preview(protected_record.normalized_text),
                            "file_b": protected_record.file,
                            "split_b": protected_record.split,
                            "protected_overlap": "True",
                        }
                    )
    candidates.sort(key=lambda row: row["similarity_score"], reverse=True)
    return candidates[:max_rows], comparisons


def length_statistics(records: List[AuditRecord]) -> List[Dict[str, Any]]:
    """Return length statistics per dataset."""
    by_file: Dict[str, List[int]] = {}
    for record in records:
        by_file.setdefault(record.file, []).append(len(record.text))
    rows: List[Dict[str, Any]] = []
    for file, lengths in sorted(by_file.items()):
        sorted_lengths = sorted(lengths)
        rows.append(
            {
                "dataset": file,
                "records": len(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "mean_length": round(statistics.mean(lengths), 4),
                "median_length": round(statistics.median(lengths), 4),
                "p95_length": percentile(sorted_lengths, 95),
                "p99_length": percentile(sorted_lengths, 99),
            }
        )
    return rows


def percentile(sorted_values: List[int], pct: int) -> int:
    """Return nearest-rank percentile."""
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, round((pct / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[index]


def build_manifest(inventory: List[DatasetInfo], blockers: List[str]) -> Dict[str, Any]:
    """Build a proposed split manifest without creating a dataset."""
    manifest: Dict[str, Any] = {
        "multitask_train": {"token_sources": [], "sequence_sources": []},
        "multitask_validation": {"token_sources": [], "sequence_sources": []},
        "calibration_only": [],
        "internal_test_only": [],
        "hard_test_only": [],
        "final_blind_test_only": [],
        "baseline_reference_only": [],
        "superseded": [],
        "excluded_or_unknown": [],
    }
    if blockers:
        for info in inventory:
            manifest["excluded_or_unknown"].append({"path": info.path, "reason": "Blockers exist; train split requires manual review."})
        return manifest
    for info in inventory:
        if info.protected and info.proposed_role.startswith("MULTITASK_TRAIN"):
            manifest["excluded_or_unknown"].append({"path": info.path, "reason": "Protected dataset cannot be train."})
        elif info.proposed_role == "MULTITASK_TRAIN_TOKEN_SOURCE":
            manifest["multitask_train"]["token_sources"].append(info.path)
        elif info.proposed_role == "MULTITASK_TRAIN_SEQUENCE_SOURCE":
            manifest["multitask_train"]["sequence_sources"].append(info.path)
        elif info.proposed_role == "MULTITASK_VALIDATION_TOKEN_SOURCE":
            manifest["multitask_validation"]["token_sources"].append(info.path)
        elif info.proposed_role == "MULTITASK_VALIDATION_SEQUENCE_SOURCE":
            manifest["multitask_validation"]["sequence_sources"].append(info.path)
        elif info.proposed_role == "CALIBRATION_ONLY":
            manifest["calibration_only"].append(info.path)
        elif info.proposed_role == "INTERNAL_TEST_ONLY":
            manifest["internal_test_only"].append(info.path)
        elif info.proposed_role == "HARD_TEST_ONLY":
            manifest["hard_test_only"].append(info.path)
        elif info.proposed_role == "FINAL_BLIND_TEST_ONLY":
            manifest["final_blind_test_only"].append(info.path)
        elif info.proposed_role == "BASELINE_REFERENCE_ONLY":
            manifest["baseline_reference_only"].append(info.path)
        elif info.proposed_role == "SUPERSEDED":
            manifest["superseded"].append(info.path)
        else:
            manifest["excluded_or_unknown"].append({"path": info.path, "reason": info.proposed_role})
    return manifest


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """Write CSV with stable headers."""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    """Write UTF-8 JSON without escaping Hebrew."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_audit(data_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run the full dataset audit and write report files."""
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_labels = read_label_schema(data_dir / "hebert_label_schema.json")
    inventory: List[DatasetInfo] = []
    records: List[AuditRecord] = []
    malformed_rows: List[Dict[str, Any]] = []
    token_label_rows: List[Dict[str, Any]] = []
    token_entity_rows: List[Dict[str, Any]] = []
    sequence_label_rows: List[Dict[str, Any]] = []
    for dataset in list_datasets(data_dir):
        info, dataset_records, malformed, token_labels_rows, token_entities_rows, sequence_rows = analyze_dataset(dataset, data_dir, allowed_labels)
        inventory.append(info)
        records.extend(dataset_records)
        malformed_rows.extend(malformed)
        token_label_rows.extend(token_labels_rows)
        token_entity_rows.extend(token_entities_rows)
        sequence_label_rows.extend(sequence_rows)

    exact_rows = build_duplicate_rows(records, use_normalized=False)
    normalized_rows = build_duplicate_rows(records, use_normalized=True)
    leakage_rows = split_leakage(records)
    group_rows = group_leakage(records)
    conflict_rows = label_conflicts(records)
    conflict_hashes = {row["normalized_hash"] for row in conflict_rows}
    cross_rows = cross_task_overlaps(records, conflict_hashes)
    near_rows, near_comparisons = near_duplicate_candidates(records)
    length_rows = length_statistics(records)
    blockers: List[str] = []
    warnings: List[str] = []
    if leakage_rows:
        blockers.append("Train/protected normalized text leakage found.")
    if group_rows:
        blockers.append("group_id appears in multiple splits.")
    if conflict_rows:
        warnings.append("Label conflicts require manual review.")
    if malformed_rows:
        warnings.append("Malformed or structurally invalid records found.")
    manifest = build_manifest(inventory, blockers)

    write_json(output_dir / "dataset_inventory.json", [info_to_dict(info) for info in inventory])
    write_csv(output_dir / "exact_duplicates.csv", CSV_HEADERS["exact_duplicates.csv"], exact_rows)
    write_csv(output_dir / "normalized_duplicates.csv", CSV_HEADERS["normalized_duplicates.csv"], normalized_rows)
    write_csv(output_dir / "split_leakage.csv", CSV_HEADERS["split_leakage.csv"], leakage_rows)
    write_csv(output_dir / "group_leakage.csv", CSV_HEADERS["group_leakage.csv"], group_rows)
    write_csv(output_dir / "label_conflicts.csv", CSV_HEADERS["label_conflicts.csv"], conflict_rows)
    write_csv(output_dir / "cross_task_overlaps.csv", CSV_HEADERS["cross_task_overlaps.csv"], cross_rows)
    write_csv(output_dir / "near_duplicate_candidates.csv", CSV_HEADERS["near_duplicate_candidates.csv"], near_rows)
    write_csv(output_dir / "token_label_distribution.csv", CSV_HEADERS["token_label_distribution.csv"], token_label_rows)
    write_csv(output_dir / "token_entity_type_distribution.csv", CSV_HEADERS["token_entity_type_distribution.csv"], token_entity_rows)
    write_csv(output_dir / "sequence_label_distribution.csv", CSV_HEADERS["sequence_label_distribution.csv"], sequence_label_rows)
    write_csv(output_dir / "dataset_length_statistics.csv", CSV_HEADERS["dataset_length_statistics.csv"], length_rows)
    write_csv(output_dir / "malformed_records.csv", CSV_HEADERS["malformed_records.csv"], malformed_rows)
    write_json(output_dir / "proposed_splits_manifest.json", manifest)
    elapsed = time.perf_counter() - started
    summary = {
        "datasets": len(inventory),
        "records": len(records),
        "token_records": sum(1 for record in records if record.task == "token"),
        "sequence_records": sum(1 for record in records if record.task == "sequence"),
        "exact_duplicates": len(exact_rows),
        "normalized_duplicates": len(normalized_rows),
        "split_leakage": len(leakage_rows),
        "group_leakage": len(group_rows),
        "label_conflicts": len(conflict_rows),
        "cross_task_overlaps": len(cross_rows),
        "near_duplicate_candidates": len(near_rows),
        "near_duplicate_comparisons": near_comparisons,
        "protected_datasets": [info.path for info in inventory if info.protected],
        "malformed_records": len(malformed_rows),
        "blockers": blockers,
        "warnings": warnings,
        "safe_to_build_multitask_dataset": not blockers,
        "elapsed_seconds": round(elapsed, 3),
    }
    write_summary_text(output_dir / "audit_summary.txt", summary, manifest)
    return summary


def info_to_dict(info: DatasetInfo) -> Dict[str, Any]:
    """Convert dataset info to JSON-safe dict."""
    return {
        "path": info.path,
        "task": info.task,
        "proposed_role": info.proposed_role,
        "format": info.format,
        "encoding": info.encoding,
        "records": info.records,
        "labels": info.labels,
        "groups": info.groups,
        "categories": info.categories,
        "duplicate_count": info.duplicate_count,
        "normalized_duplicate_count": info.normalized_duplicate_count,
        "malformed_count": info.malformed_count,
        "empty_text_count": info.empty_text_count,
        "protected": info.protected,
        "split": info.split,
        "fields": info.fields,
        "sample_text_preview": info.sample_text_preview,
        "notes": info.notes,
    }


def write_summary_text(path: Path, summary: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    """Write a concise human-readable summary."""
    lines = [
        "Multi-Task Dataset Audit",
        "========================",
        f"Datasets checked: {summary['datasets']}",
        f"Total records: {summary['records']}",
        f"Token records: {summary['token_records']}",
        f"Sequence records: {summary['sequence_records']}",
        f"Exact duplicates: {summary['exact_duplicates']}",
        f"Normalized duplicates: {summary['normalized_duplicates']}",
        f"Split leakage: {summary['split_leakage']}",
        f"Group leakage: {summary['group_leakage']}",
        f"Label conflicts: {summary['label_conflicts']}",
        f"Cross-task overlaps: {summary['cross_task_overlaps']}",
        f"Near-duplicate candidates: {summary['near_duplicate_candidates']}",
        f"Malformed records: {summary['malformed_records']}",
        f"Protected datasets: {len(summary['protected_datasets'])}",
        f"Safe to build Multi-Task dataset: {summary['safe_to_build_multitask_dataset']}",
        "",
        "Blockers:",
        *[f"- {item}" for item in summary["blockers"]],
        "",
        "Warnings:",
        *[f"- {item}" for item in summary["warnings"]],
        "",
        "Proposed splits:",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Audit Token/Sequence datasets for Multi-Task HeBERT preparation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/multitask_data_audit"))
    args = parser.parse_args()
    summary = run_audit(args.data_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
