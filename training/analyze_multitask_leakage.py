from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.audit_multitask_datasets import (
    AuditRecord,
    analyze_dataset,
    list_datasets,
    normalize_text,
    preview,
    read_label_schema,
)


TOKEN_TRAIN = "data/hebert_token_classification_train_v3_targeted.jsonl"
TOKEN_VALIDATION = "data/hebert_token_classification_validation_v3_targeted.jsonl"
TOKEN_TEST = "data/hebert_token_classification_test_v3_targeted.jsonl"
SEQUENCE_TRAIN = "data/hebert_sequence_train.jsonl"
SEQUENCE_HARD_TRAIN = "data/hebert_sequence_hard_train_v3.jsonl"
SEQUENCE_VALIDATION = "data/hebert_sequence_validation.jsonl"
SEQUENCE_PROTECTED = {
    "data/hebert_sequence_test.jsonl",
    "data/hebert_sequence_development_v3.jsonl",
    "data/hebert_sequence_calibration_v3.jsonl",
    "data/hebert_sequence_real_world_test_v3.jsonl",
    "data/hebert_targeted_challenge_v3.jsonl",
    "data/svm_blind_test_v6.csv",
    "data/real_world_definition_challenge.csv",
}

ACTIVE_TRAIN = {TOKEN_TRAIN, SEQUENCE_TRAIN, SEQUENCE_HARD_TRAIN}
ACTIVE_VALIDATION = {TOKEN_VALIDATION, SEQUENCE_VALIDATION}
ACTIVE_PROTECTED = {TOKEN_TEST, *SEQUENCE_PROTECTED}
ACTIVE_FILES = ACTIVE_TRAIN | ACTIVE_VALIDATION | ACTIVE_PROTECTED
FINAL_OR_CRITICAL = {
    "data/hebert_sequence_calibration_v3.jsonl",
    "data/svm_blind_test_v6.csv",
    "data/real_world_definition_challenge.csv",
    "data/hebert_sequence_real_world_test_v3.jsonl",
    "data/hebert_targeted_challenge_v3.jsonl",
}

PAIR_HEADERS = [
    "file_a",
    "role_a",
    "file_b",
    "role_b",
    "exact_overlap_count",
    "normalized_overlap_count",
    "group_overlap_count",
    "unique_texts_a",
    "unique_texts_b",
    "percentage_of_a_affected",
    "percentage_of_b_affected",
    "severity",
    "blocker",
    "reason",
]


@dataclass(frozen=True)
class DatasetRole:
    path: str
    task: str
    role: str
    split: str
    active: bool
    protected: bool


def load_records(data_dir: Path) -> Tuple[List[AuditRecord], Dict[str, DatasetRole]]:
    """Load all audit records without changing source datasets."""
    labels = read_label_schema(data_dir / "hebert_label_schema.json")
    records: List[AuditRecord] = []
    roles: Dict[str, DatasetRole] = {}
    for path in list_datasets(data_dir):
        info, dataset_records, _, _, _, _ = analyze_dataset(path, data_dir, labels)
        active = info.path in ACTIVE_FILES or (
            info.path.startswith("data/hebert_targeted_challenge_") and info.path.endswith(".jsonl")
        )
        protected = info.path in ACTIVE_PROTECTED or (
            info.path.startswith("data/hebert_targeted_challenge_") and info.path.endswith(".jsonl")
        )
        role = active_role(info.path, info.proposed_role, protected)
        roles[info.path] = DatasetRole(
            path=info.path,
            task=info.task,
            role=role,
            split=info.split,
            active=active,
            protected=protected,
        )
        records.extend(dataset_records)
    return records, roles


def active_role(path: str, proposed: str, protected: bool) -> str:
    """Return the role used for focused leakage analysis."""
    if path in ACTIVE_TRAIN:
        return "TRAIN_CANDIDATE"
    if path in ACTIVE_VALIDATION:
        return "VALIDATION_CANDIDATE"
    if protected:
        return "PROTECTED_EVALUATION"
    return "HISTORICAL_OR_SUPERSEDED"


def pair_key(a: str, b: str) -> Tuple[str, str]:
    """Return a stable unordered pair key."""
    return (a, b) if a <= b else (b, a)


def group_by_file(records: Sequence[AuditRecord]) -> Dict[str, List[AuditRecord]]:
    """Group records by source file."""
    grouped: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in records:
        grouped[record.file].append(record)
    return grouped


def build_pair_summary(records: Sequence[AuditRecord], roles: Dict[str, DatasetRole]) -> List[Dict[str, Any]]:
    """Aggregate exact, normalized, and group overlap by file pair."""
    by_file = group_by_file(records)
    exact_pairs = overlap_counts(records, "exact_hash")
    norm_pairs = overlap_counts(records, "normalized_hash")
    group_pairs = group_overlap_counts(records)
    all_pairs = set(exact_pairs) | set(norm_pairs) | set(group_pairs)
    rows: List[Dict[str, Any]] = []
    for file_a, file_b in sorted(all_pairs):
        role_a = roles[file_a]
        role_b = roles[file_b]
        exact_count = exact_pairs.get((file_a, file_b), 0)
        norm_count = norm_pairs.get((file_a, file_b), 0)
        group_count = group_pairs.get((file_a, file_b), 0)
        severity, blocker, reason = classify_pair(role_a, role_b, exact_count, norm_count, group_count)
        unique_a = len({record.normalized_hash for record in by_file[file_a]})
        unique_b = len({record.normalized_hash for record in by_file[file_b]})
        affected_a = affected_unique_count(records, file_a, file_b)
        affected_b = affected_unique_count(records, file_b, file_a)
        rows.append(
            {
                "file_a": file_a,
                "role_a": role_a.role,
                "file_b": file_b,
                "role_b": role_b.role,
                "exact_overlap_count": exact_count,
                "normalized_overlap_count": norm_count,
                "group_overlap_count": group_count,
                "unique_texts_a": unique_a,
                "unique_texts_b": unique_b,
                "percentage_of_a_affected": round((affected_a / unique_a * 100.0) if unique_a else 0.0, 4),
                "percentage_of_b_affected": round((affected_b / unique_b * 100.0) if unique_b else 0.0, 4),
                "severity": severity,
                "blocker": blocker,
                "reason": reason,
            }
        )
    return rows


def overlap_counts(records: Sequence[AuditRecord], attr: str) -> Dict[Tuple[str, str], int]:
    """Count shared unique hashes between files."""
    hash_to_files: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        hash_to_files[getattr(record, attr)][record.file] += 1
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for files in hash_to_files.values():
        names = sorted(files)
        for index, file_a in enumerate(names):
            for file_b in names[index:]:
                if file_a == file_b:
                    if files[file_a] > 1:
                        counts[(file_a, file_b)] += files[file_a] - 1
                else:
                    counts[(file_a, file_b)] += min(files[file_a], files[file_b])
    return dict(counts)


def group_overlap_counts(records: Sequence[AuditRecord]) -> Dict[Tuple[str, str], int]:
    """Count shared group IDs separately from text overlap."""
    group_to_files: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record.group_id:
            group_to_files[record.group_id][record.file] += 1
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for files in group_to_files.values():
        names = sorted(files)
        for index, file_a in enumerate(names):
            for file_b in names[index + 1 :]:
                counts[(file_a, file_b)] += 1
    return dict(counts)


def affected_unique_count(records: Sequence[AuditRecord], file_a: str, file_b: str) -> int:
    """Count normalized texts in file_a that also appear in file_b."""
    hashes_b = {record.normalized_hash for record in records if record.file == file_b}
    return len({record.normalized_hash for record in records if record.file == file_a and record.normalized_hash in hashes_b})


def classify_pair(a: DatasetRole, b: DatasetRole, exact_count: int, norm_count: int, group_count: int) -> Tuple[str, bool, str]:
    """Classify pair severity and whether it blocks clean split construction."""
    if not a.active or not b.active:
        return "HISTORICAL_ONLY", False, root_cause(a, b, exact_count, norm_count, group_count)
    roles = {a.role, b.role}
    has_train = "TRAIN_CANDIDATE" in roles
    has_validation = "VALIDATION_CANDIDATE" in roles
    has_protected = "PROTECTED_EVALUATION" in roles
    has_overlap = any((exact_count, norm_count, group_count))
    if has_train and has_protected:
        protected_file = b.path if b.protected else a.path
        severity = "CRITICAL" if protected_file in FINAL_OR_CRITICAL else "HIGH"
        return severity, has_overlap, root_cause(a, b, exact_count, norm_count, group_count)
    if has_train and has_validation:
        return "MEDIUM", has_overlap, root_cause(a, b, exact_count, norm_count, group_count)
    if has_train and roles == {"TRAIN_CANDIDATE"}:
        return "LOW", False, root_cause(a, b, exact_count, norm_count, group_count)
    if has_protected:
        return "HIGH", False, root_cause(a, b, exact_count, norm_count, group_count)
    return "LOW", False, root_cause(a, b, exact_count, norm_count, group_count)


def root_cause(a: DatasetRole, b: DatasetRole, exact_count: int, norm_count: int, group_count: int) -> str:
    """Return likely root cause from pair roles and overlap type."""
    if not a.active or not b.active:
        return "HISTORICAL_VERSION_OVERLAP"
    if a.path == b.path:
        return "SAME_FILE_DUPLICATES"
    if group_count and not norm_count:
        return "SAME_GROUP_SPLIT"
    if group_count and norm_count:
        return "SAME_TEMPLATE_FAMILY"
    if exact_count:
        return "EXACT_TEXT_COPY"
    if norm_count:
        return "NORMALIZATION_ONLY"
    return "UNKNOWN"


def protected_overlap_records(records: Sequence[AuditRecord]) -> List[Dict[str, Any]]:
    """Return candidate train records that overlap protected sets."""
    protected_by_hash: Dict[str, List[AuditRecord]] = defaultdict(list)
    protected_by_group: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in records:
        if record.file in ACTIVE_PROTECTED or record.file.startswith("data/hebert_targeted_challenge_"):
            protected_by_hash[record.normalized_hash].append(record)
            if record.group_id:
                protected_by_group[record.group_id].append(record)
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for record in records:
        if record.file not in ACTIVE_TRAIN:
            continue
        matches = [(item, "NORMALIZED_TEXT") for item in protected_by_hash.get(record.normalized_hash, [])]
        if record.group_id:
            matches.extend((item, "GROUP_ID") for item in protected_by_group.get(record.group_id, []))
        for protected, overlap_type in matches:
            key = (record.file, protected.file, record.normalized_hash, overlap_type)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "task": record.task,
                    "source_file": record.file,
                    "protected_file": protected.file,
                    "normalized_hash": record.normalized_hash,
                    "text_preview": preview(record.text),
                    "source_label": record.label,
                    "protected_label": protected.label,
                    "group_id": record.group_id,
                    "overlap_type": overlap_type,
                    "severity": "CRITICAL" if protected.file in FINAL_OR_CRITICAL else "HIGH",
                }
            )
    return rows


def group_overlap_summary(records: Sequence[AuditRecord], roles: Dict[str, DatasetRole]) -> List[Dict[str, Any]]:
    """Summarize group overlaps by file pair."""
    rows: List[Dict[str, Any]] = []
    for (file_a, file_b), count in sorted(group_overlap_counts(records).items()):
        if not (roles[file_a].active or roles[file_b].active):
            continue
        rows.append(
            {
                "file_a": file_a,
                "role_a": roles[file_a].role,
                "file_b": file_b,
                "role_b": roles[file_b].role,
                "shared_group_count": count,
                "blocker": roles[file_a].role != roles[file_b].role,
            }
        )
    return rows


def classify_cross_task(records: Sequence[AuditRecord]) -> List[Dict[str, Any]]:
    """Classify token/sequence text overlaps for split planning."""
    by_hash: Dict[str, Dict[str, List[AuditRecord]]] = defaultdict(lambda: {"token": [], "sequence": []})
    for record in records:
        if record.file in ACTIVE_FILES or record.file.startswith("data/hebert_targeted_challenge_"):
            by_hash[record.normalized_hash][record.task].append(record)
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for norm_hash, groups in by_hash.items():
        for token in groups["token"]:
            for sequence in groups["sequence"]:
                key = (norm_hash, token.file, sequence.file)
                if key in seen:
                    continue
                seen.add(key)
                classification = cross_task_classification(token, sequence)
                rows.append(
                    {
                        "normalized_hash": norm_hash,
                        "text_preview": preview(token.text),
                        "token_file": token.file,
                        "token_split": token.split,
                        "sequence_file": sequence.file,
                        "sequence_split": sequence.split,
                        "token_entity_count": token.entity_count,
                        "token_types": json.dumps(list(token.entity_types), ensure_ascii=False),
                        "sequence_label": sequence.label,
                        "classification": classification,
                        "notes": cross_task_notes(classification),
                    }
                )
    return rows


def cross_task_classification(token: AuditRecord, sequence: AuditRecord) -> str:
    """Classify one token/sequence overlap."""
    if token.file in ACTIVE_TRAIN and sequence.file in ACTIVE_PROTECTED:
        return "SPLIT_CONFLICT"
    if sequence.file in ACTIVE_TRAIN and token.file in ACTIVE_PROTECTED:
        return "SPLIT_CONFLICT"
    if token.split == sequence.split and token.label == "LEAK" and sequence.label == "LEAK":
        return "SAFE_BOTH_TASKS"
    if token.split == sequence.split:
        return "SAFE_SEPARATE_TASK_RECORDS"
    if token.file in ACTIVE_VALIDATION and sequence.file in ACTIVE_TRAIN:
        return "SPLIT_CONFLICT"
    if sequence.file in ACTIVE_VALIDATION and token.file in ACTIVE_TRAIN:
        return "SPLIT_CONFLICT"
    return "REVIEW_REQUIRED"


def cross_task_notes(classification: str) -> str:
    """Return a short explanation for cross-task classification."""
    return {
        "SAFE_BOTH_TASKS": "Same split and compatible LEAK labels can become one both-task example.",
        "SAFE_SEPARATE_TASK_RECORDS": "Same split but not directly mergeable; keep task-specific records.",
        "SPLIT_CONFLICT": "Same text appears across incompatible splits or protected data.",
        "REVIEW_REQUIRED": "Overlap needs manual review before split planning.",
    }[classification]


def option_a_projection(records: Sequence[AuditRecord]) -> Dict[str, Any]:
    """Project preserving protected splits and removing overlaps from train."""
    forbidden_hashes = {record.normalized_hash for record in records if record.file in ACTIVE_VALIDATION or record.file in ACTIVE_PROTECTED}
    forbidden_groups = {record.group_id for record in records if record.group_id and (record.file in ACTIVE_VALIDATION or record.file in ACTIVE_PROTECTED)}
    kept = [record for record in records if record.file in ACTIVE_TRAIN and record.normalized_hash not in forbidden_hashes and (not record.group_id or record.group_id not in forbidden_groups)]
    kept_ids = {id(record) for record in kept}
    removed = [record for record in records if record.file in ACTIVE_TRAIN and id(record) not in kept_ids]
    return projection("OPTION_A", kept, removed, records, validation_sources=sorted(ACTIVE_VALIDATION))


def option_b_projection(records: Sequence[AuditRecord]) -> Dict[str, Any]:
    """Project rebuilding only train/validation from allowed train+validation pools."""
    protected_hashes = {record.normalized_hash for record in records if record.file in ACTIVE_PROTECTED}
    protected_groups = {record.group_id for record in records if record.group_id and record.file in ACTIVE_PROTECTED}
    allowed_pool = [
        record
        for record in records
        if record.file in ACTIVE_TRAIN | ACTIVE_VALIDATION
        and record.normalized_hash not in protected_hashes
        and (not record.group_id or record.group_id not in protected_groups)
    ]
    buckets: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in allowed_pool:
        key = record.group_id or record.normalized_hash
        buckets[key].append(record)
    sorted_keys = sorted(buckets)
    validation_keys = set(sorted_keys[::5])
    train = [record for key, items in buckets.items() if key not in validation_keys for record in items]
    validation = [record for key, items in buckets.items() if key in validation_keys for record in items]
    kept_ids = {id(record) for record in train}
    kept_ids.update(id(record) for record in validation)
    removed = [record for record in records if record.file in ACTIVE_TRAIN | ACTIVE_VALIDATION and id(record) not in kept_ids]
    result = projection("OPTION_B", train, removed, records, validation_records=validation)
    result["validation_records"] = len(validation)
    result["validation_groups"] = len({record.group_id or record.normalized_hash for record in validation})
    return result


def projection(
    strategy: str,
    kept: Sequence[AuditRecord],
    removed: Sequence[AuditRecord],
    all_records: Sequence[AuditRecord],
    validation_sources: Sequence[str] | None = None,
    validation_records: Sequence[AuditRecord] | None = None,
) -> Dict[str, Any]:
    """Build projected counts without writing a dataset."""
    token_kept = [record for record in kept if record.task == "token"]
    sequence_kept = [record for record in kept if record.task == "sequence"]
    return {
        "strategy": strategy,
        "train_records": len(kept),
        "token_train_records": len(token_kept),
        "sequence_train_records": len(sequence_kept),
        "removed_train_or_pool_records": len(removed),
        "removed_groups": len({record.group_id for record in removed if record.group_id}),
        "train_groups": len({record.group_id or record.normalized_hash for record in kept}),
        "validation_sources": list(validation_sources or []),
        "validation_records": len(validation_records or []),
        "sequence_label_balance": dict(Counter(record.label for record in sequence_kept)),
        "token_entity_type_distribution": dict(Counter(entity for record in token_kept for entity in record.entity_types)),
        "protected_sets_unchanged": sorted(ACTIVE_PROTECTED),
        "writes_datasets": False,
    }


def active_summary(records: Sequence[AuditRecord], pair_rows: Sequence[Dict[str, Any]], cross_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return focused active leakage numbers."""
    active_pair_rows = [row for row in pair_rows if row["severity"] != "HISTORICAL_ONLY"]
    train_protected = [row for row in active_pair_rows if "TRAIN_CANDIDATE" in {row["role_a"], row["role_b"]} and "PROTECTED_EVALUATION" in {row["role_a"], row["role_b"]}]
    train_validation = [row for row in active_pair_rows if "TRAIN_CANDIDATE" in {row["role_a"], row["role_b"]} and "VALIDATION_CANDIDATE" in {row["role_a"], row["role_b"]}]
    within_train = [row for row in active_pair_rows if row["role_a"] == row["role_b"] == "TRAIN_CANDIDATE"]
    cross_counts = Counter(row["classification"] for row in cross_rows)
    return {
        "active_exact_leakage": sum(int(row["exact_overlap_count"]) for row in active_pair_rows),
        "active_normalized_leakage": sum(int(row["normalized_overlap_count"]) for row in active_pair_rows),
        "active_group_leakage": sum(int(row["group_overlap_count"]) for row in active_pair_rows),
        "active_train_to_protected_leakage": sum(int(row["normalized_overlap_count"]) + int(row["group_overlap_count"]) for row in train_protected),
        "active_train_to_validation_leakage": sum(int(row["normalized_overlap_count"]) + int(row["group_overlap_count"]) for row in train_validation),
        "active_within_train_duplication": sum(int(row["normalized_overlap_count"]) + int(row["group_overlap_count"]) for row in within_train),
        "cross_task_classification_counts": dict(cross_counts),
        "active_pair_count": len(active_pair_rows),
        "train_protected_pair_count": len(train_protected),
        "train_validation_pair_count": len(train_validation),
    }


def recommended_plan(records: Sequence[AuditRecord], option_a: Dict[str, Any], option_b: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Choose the preferred clean split strategy without creating datasets."""
    strategy = "OPTION_A" if option_a["train_records"] >= max(100, int(option_b["train_records"] * 0.5)) else "OPTION_B"
    selected = option_a if strategy == "OPTION_A" else option_b
    remaining = []
    if summary["active_train_to_protected_leakage"]:
        remaining.append("Candidate train overlaps protected sets before applying removal rules.")
    if summary["active_train_to_validation_leakage"]:
        remaining.append("Candidate train overlaps validation before applying removal rules.")
    safe_after_rules = not would_have_overlap_after_rules(records, strategy)
    return {
        "strategy": strategy,
        "protected_sets": sorted(ACTIVE_PROTECTED),
        "token_train_sources": [TOKEN_TRAIN],
        "sequence_train_sources": [SEQUENCE_TRAIN, SEQUENCE_HARD_TRAIN],
        "validation_sources": sorted(ACTIVE_VALIDATION),
        "removal_rules": [
            "Remove any train candidate whose normalized text appears in protected evaluation.",
            "Remove any train candidate whose group_id appears in protected evaluation.",
            "Remove any train candidate whose normalized text or group_id appears in validation for Option A.",
            "Never move calibration, blind, challenge, test, or real-world records into train.",
        ],
        "group_rules": ["Keep each group_id in a single split.", "Use normalized text as the minimum split key when group_id is missing."],
        "deduplication_rules": ["Deduplicate within train by normalized text, task, and label.", "Merge compatible same-split token+sequence records as both-task examples."],
        "projected_counts": selected,
        "remaining_blockers": [] if safe_after_rules else remaining,
        "safe_to_build": safe_after_rules,
    }


def would_have_overlap_after_rules(records: Sequence[AuditRecord], strategy: str) -> bool:
    """Check whether projected removal rules clear protected/validation overlaps."""
    if strategy == "OPTION_A":
        kept = projected_kept_option_a(records)
        protected_validation = [record for record in records if record.file in ACTIVE_PROTECTED | ACTIVE_VALIDATION]
    else:
        kept = projected_kept_option_b(records)[0]
        protected_validation = [record for record in records if record.file in ACTIVE_PROTECTED]
    hashes = {record.normalized_hash for record in protected_validation}
    groups = {record.group_id for record in protected_validation if record.group_id}
    return any(record.normalized_hash in hashes or (record.group_id and record.group_id in groups) for record in kept)


def projected_kept_option_a(records: Sequence[AuditRecord]) -> List[AuditRecord]:
    """Return projected Option A train records."""
    forbidden_hashes = {record.normalized_hash for record in records if record.file in ACTIVE_VALIDATION or record.file in ACTIVE_PROTECTED}
    forbidden_groups = {record.group_id for record in records if record.group_id and (record.file in ACTIVE_VALIDATION or record.file in ACTIVE_PROTECTED)}
    return [record for record in records if record.file in ACTIVE_TRAIN and record.normalized_hash not in forbidden_hashes and (not record.group_id or record.group_id not in forbidden_groups)]


def projected_kept_option_b(records: Sequence[AuditRecord]) -> Tuple[List[AuditRecord], List[AuditRecord]]:
    """Return projected Option B train and validation records."""
    protected_hashes = {record.normalized_hash for record in records if record.file in ACTIVE_PROTECTED}
    protected_groups = {record.group_id for record in records if record.group_id and record.file in ACTIVE_PROTECTED}
    allowed = [record for record in records if record.file in ACTIVE_TRAIN | ACTIVE_VALIDATION and record.normalized_hash not in protected_hashes and (not record.group_id or record.group_id not in protected_groups)]
    buckets: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in allowed:
        buckets[record.group_id or record.normalized_hash].append(record)
    validation_keys = set(sorted(buckets)[::5])
    train = [record for key, items in buckets.items() if key not in validation_keys for record in items]
    validation = [record for key, items in buckets.items() if key in validation_keys for record in items]
    return train, validation


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], headers: Sequence[str]) -> None:
    """Write CSV rows using UTF-8 with BOM for Excel-friendly Hebrew."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    """Write JSON without escaping Hebrew."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_analysis(audit_dir: Path, data_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run focused leakage analysis and write reports."""
    start = time.perf_counter()
    records, roles = load_records(data_dir)
    pair_rows = build_pair_summary(records, roles)
    protected_rows = protected_overlap_records(records)
    group_rows = group_overlap_summary(records, roles)
    cross_rows = classify_cross_task(records)
    option_a = option_a_projection(records)
    option_b = option_b_projection(records)
    summary = active_summary(records, pair_rows, cross_rows)
    historical_rows = [row for row in pair_rows if row["severity"] == "HISTORICAL_ONLY"]
    historical_summary = {
        "historical_pair_count": len(historical_rows),
        "historical_exact_overlap": sum(int(row["exact_overlap_count"]) for row in historical_rows),
        "historical_normalized_overlap": sum(int(row["normalized_overlap_count"]) for row in historical_rows),
        "historical_group_overlap": sum(int(row["group_overlap_count"]) for row in historical_rows),
    }
    plan = recommended_plan(records, option_a, option_b, summary)
    result = {
        "datasets_loaded": len(roles),
        "records_loaded": len(records),
        "historical_only_leakage": historical_summary["historical_normalized_overlap"],
        "active_summary": summary,
        "historical_summary": historical_summary,
        "option_a": option_a,
        "option_b": option_b,
        "recommended_strategy": plan["strategy"],
        "safe_to_build_after_rules": plan["safe_to_build"],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "audit_dir": str(audit_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "leakage_by_file_pair.csv", pair_rows, PAIR_HEADERS)
    write_json(output_dir / "active_leakage_summary.json", summary)
    write_json(output_dir / "historical_overlap_summary.json", historical_summary)
    write_csv(
        output_dir / "protected_overlap_records.csv",
        protected_rows,
        ["task", "source_file", "protected_file", "normalized_hash", "text_preview", "source_label", "protected_label", "group_id", "overlap_type", "severity"],
    )
    write_csv(output_dir / "group_overlap_summary.csv", group_rows, ["file_a", "role_a", "file_b", "role_b", "shared_group_count", "blocker"])
    write_csv(
        output_dir / "cross_task_overlap_classification.csv",
        cross_rows,
        ["normalized_hash", "text_preview", "token_file", "token_split", "sequence_file", "sequence_split", "token_entity_count", "token_types", "sequence_label", "classification", "notes"],
    )
    write_json(output_dir / "option_a_projection.json", option_a)
    write_json(output_dir / "option_b_projection.json", option_b)
    write_json(output_dir / "recommended_clean_split_plan.json", plan)
    (output_dir / "leakage_analysis_summary.txt").write_text(build_text_summary(result, plan), encoding="utf-8")
    return result


def build_text_summary(result: Dict[str, Any], plan: Dict[str, Any]) -> str:
    """Build a concise human-readable leakage report."""
    active = result["active_summary"]
    return "\n".join(
        [
            "Multi-Task Leakage Focused Analysis",
            "====================================",
            f"Datasets loaded: {result['datasets_loaded']}",
            f"Records loaded: {result['records_loaded']}",
            f"Historical-only normalized overlap: {result['historical_only_leakage']}",
            f"Active normalized leakage: {active['active_normalized_leakage']}",
            f"Active group leakage: {active['active_group_leakage']}",
            f"Active train-to-protected leakage: {active['active_train_to_protected_leakage']}",
            f"Active train-to-validation leakage: {active['active_train_to_validation_leakage']}",
            f"Active within-train duplication: {active['active_within_train_duplication']}",
            f"Cross-task classifications: {json.dumps(active['cross_task_classification_counts'], ensure_ascii=False)}",
            f"Option A projected train records: {result['option_a']['train_records']}",
            f"Option B projected train records: {result['option_b']['train_records']}",
            f"Recommended strategy: {plan['strategy']}",
            f"Safe to build after rules: {plan['safe_to_build']}",
            "Removal rules:",
            *[f"- {rule}" for rule in plan["removal_rules"]],
            "Remaining blockers:",
            *(f"- {item}" for item in (plan["remaining_blockers"] or ["None after applying projected rules."])),
        ]
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Analyze active Multi-Task leakage sources without writing datasets.")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(args.audit_dir, args.data_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
