from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TOKEN_TRAIN = "hebert_token_classification_train_v3_targeted.jsonl"
TOKEN_VALIDATION = "hebert_token_classification_validation_v3_targeted.jsonl"
SEQUENCE_TRAIN = ["hebert_sequence_train.jsonl", "hebert_sequence_hard_train_v3.jsonl"]
SEQUENCE_VALIDATION = "hebert_sequence_validation.jsonl"
PROTECTED = [
    "hebert_token_classification_test_v3_targeted.jsonl",
    "hebert_targeted_challenge_v2.jsonl",
    "hebert_targeted_challenge_v3.jsonl",
    "hebert_sequence_test.jsonl",
    "hebert_sequence_development_v3.jsonl",
    "hebert_sequence_calibration_v3.jsonl",
    "hebert_sequence_real_world_test_v3.jsonl",
    "svm_blind_test_v6.csv",
    "real_world_definition_challenge.csv",
]


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def valid_group(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    value = str(value).strip()
    return value or None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError(f"Invalid record in {path} at line {line_number}")
            records.append(item)
    return records


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        text = next((row[k] for k in ("text", "sentence", "segment") if row.get(k)), None)
        if text:
            records.append({**row, "text": text, "_row": i})
    return records


def read_any(path: Path) -> list[dict[str, Any]]:
    return read_csv_records(path) if path.suffix.lower() == ".csv" else read_jsonl(path)


def sequence_label(record: dict[str, Any]) -> int | None:
    value = record.get("sequence_label", record.get("document_label", record.get("label", record.get("label_name"))))
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str):
        cleaned = value.strip().upper()
        if cleaned in {"1", "LEAK"}:
            return 1
        if cleaned in {"0", "NON_LEAK", "NON-LEAK"}:
            return 0
    return None


def protected_keys(data_dir: Path) -> tuple[set[str], set[str], dict[str, int]]:
    hashes: set[str] = set()
    groups: set[str] = set()
    counts: dict[str, int] = {}
    for name in PROTECTED:
        path = data_dir / name
        if not path.exists():
            continue
        rows = read_any(path)
        counts[name] = len(rows)
        for row in rows:
            hashes.add(text_hash(row["text"]))
            group = valid_group(row.get("group_id") or row.get("pair_id") or row.get("case_id"))
            if group:
                groups.add(group)
    return hashes, groups, counts


def is_blocked(record: dict[str, Any], blocked_hashes: set[str], blocked_groups: set[str]) -> bool:
    if text_hash(record["text"]) in blocked_hashes:
        return True
    group = valid_group(record.get("group_id") or record.get("pair_id") or record.get("case_id"))
    return bool(group and group in blocked_groups)


def token_payload(record: dict[str, Any]) -> dict[str, Any]:
    tokens = record.get("tokens")
    ner_tags = record.get("ner_tags")
    tag_names = record.get("tag_names")
    if not isinstance(tokens, list) or not isinstance(ner_tags, list) or len(tokens) != len(ner_tags):
        raise ValueError("Invalid token record: tokens/ner_tags mismatch")
    if tag_names is not None and (not isinstance(tag_names, list) or len(tag_names) != len(tokens)):
        raise ValueError("Invalid token record: tag_names mismatch")
    return {
        "tokens": tokens,
        "ner_tags": ner_tags,
        "tag_names": tag_names,
        "typed_spans": record.get("typed_spans", []),
    }


def canonical_record(record: dict[str, Any], task: str, split: str, source: str, index: int) -> dict[str, Any]:
    label = sequence_label(record)
    result: dict[str, Any] = {
        "id": f"{split}_{task}_{index:07d}",
        "text": record["text"],
        "normalized_hash": text_hash(record["text"]),
        "task": task,
        "split": split,
        "sequence_label": label,
        "group_id": valid_group(record.get("group_id") or record.get("pair_id") or record.get("case_id")),
        "category": record.get("category"),
        "source_dataset": source,
        "source_id": record.get("id", record.get("case_id")),
    }
    if task in {"token", "both"}:
        result.update(token_payload(record))
    else:
        result.update({"tokens": None, "ner_tags": None, "tag_names": None, "typed_spans": None})
    return result


def dedup_key(record: dict[str, Any]) -> tuple[Any, ...]:
    token_signature = tuple(record.get("ner_tags") or []) if record["task"] in {"token", "both"} else None
    return (record["normalized_hash"], record["task"], record.get("sequence_label"), token_signature)


def merge_same_text(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge compatible token/both and sequence-only records within one split."""
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_hash.setdefault(record["normalized_hash"], []).append(record)
    merged: list[dict[str, Any]] = []
    merge_count = 0
    for group in by_hash.values():
        token_records = [r for r in group if r["task"] in {"token", "both"}]
        sequence_records = [r for r in group if r["task"] == "sequence"]
        consumed: set[int] = set()
        for token_rec in token_records:
            match_index = next((i for i, seq in enumerate(sequence_records) if i not in consumed and seq["sequence_label"] == token_rec["sequence_label"]), None)
            if match_index is not None:
                seq = sequence_records[match_index]
                consumed.add(match_index)
                token_rec = dict(token_rec)
                token_rec["task"] = "both"
                token_rec["source_dataset"] = [token_rec["source_dataset"], seq["source_dataset"]]
                merge_count += 1
            merged.append(token_rec)
        merged.extend(seq for i, seq in enumerate(sequence_records) if i not in consumed)
        merged.extend(r for r in group if r["task"] not in {"token", "both", "sequence"})
    return merged, merge_count


def clean_split(
    token_rows: list[dict[str, Any]], sequence_rows: list[tuple[str, dict[str, Any]]], split: str,
    blocked_hashes: set[str], blocked_groups: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int], set[str], set[str]]:
    output: list[dict[str, Any]] = []
    stats = Counter()
    for i, row in enumerate(token_rows):
        if is_blocked(row, blocked_hashes, blocked_groups):
            stats["removed_token"] += 1
            continue
        # Token data contains document_label, so it provides both heads.
        output.append(canonical_record(row, "both", split, TOKEN_TRAIN if split == "train" else TOKEN_VALIDATION, i))
    offset = len(output)
    for i, (source, row) in enumerate(sequence_rows):
        if is_blocked(row, blocked_hashes, blocked_groups):
            stats["removed_sequence"] += 1
            continue
        output.append(canonical_record(row, "sequence", split, source, offset + i))

    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in output:
        key = dedup_key(row)
        if key in seen:
            stats["deduplicated"] += 1
            continue
        seen.add(key)
        unique.append(row)
    unique, merge_count = merge_same_text(unique)
    stats["merged_cross_task"] = merge_count
    hashes = {r["normalized_hash"] for r in unique}
    groups = {r["group_id"] for r in unique if r.get("group_id")}
    return unique, dict(stats), hashes, groups


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter(r["task"] for r in rows)
    labels = Counter("LEAK" if r.get("sequence_label") == 1 else "NON_LEAK" if r.get("sequence_label") == 0 else "MISSING" for r in rows)
    entity_types = Counter()
    for r in rows:
        for name in r.get("tag_names") or []:
            if isinstance(name, str) and name != "O":
                entity_types[name[2:] if name.startswith(("B-", "I-")) else name] += 1
    lengths = [len(r["text"]) for r in rows]
    return {
        "records": len(rows), "tasks": dict(tasks), "sequence_labels": dict(labels),
        "entity_types": dict(entity_types),
        "text_length": {"min": min(lengths, default=0), "max": max(lengths, default=0), "mean": statistics.mean(lengths) if lengths else 0},
    }


def verify_disjoint(train: list[dict[str, Any]], validation: list[dict[str, Any]], protected_hashes: set[str], protected_groups: set[str]) -> dict[str, int]:
    train_hashes = {r["normalized_hash"] for r in train}; val_hashes = {r["normalized_hash"] for r in validation}
    train_groups = {r["group_id"] for r in train if r.get("group_id")}; val_groups = {r["group_id"] for r in validation if r.get("group_id")}
    checks = {
        "train_validation_text_overlap": len(train_hashes & val_hashes),
        "train_validation_group_overlap": len(train_groups & val_groups),
        "train_protected_text_overlap": len(train_hashes & protected_hashes),
        "validation_protected_text_overlap": len(val_hashes & protected_hashes),
        "train_protected_group_overlap": len(train_groups & protected_groups),
        "validation_protected_group_overlap": len(val_groups & protected_groups),
    }
    if any(checks.values()):
        raise RuntimeError(f"Split isolation failed: {checks}")
    return checks


def build(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    protected_hashes, protected_groups, protected_counts = protected_keys(data_dir)
    token_val = read_jsonl(data_dir / TOKEN_VALIDATION)
    seq_val = [(SEQUENCE_VALIDATION, r) for r in read_jsonl(data_dir / SEQUENCE_VALIDATION)]
    validation, val_stats, val_hashes, val_groups = clean_split(token_val, seq_val, "validation", protected_hashes, protected_groups)

    train_block_hashes = protected_hashes | val_hashes
    train_block_groups = protected_groups | val_groups
    token_train = read_jsonl(data_dir / TOKEN_TRAIN)
    seq_train = [(name, row) for name in SEQUENCE_TRAIN for row in read_jsonl(data_dir / name)]
    train, train_stats, _, _ = clean_split(token_train, seq_train, "train", train_block_hashes, train_block_groups)

    checks = verify_disjoint(train, validation, protected_hashes, protected_groups)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation)

    label_schema_src = data_dir / "hebert_label_schema.json"
    if label_schema_src.exists():
        (output_dir / "label_schema.json").write_text(label_schema_src.read_text(encoding="utf-8-sig"), encoding="utf-8")

    manifest = {
        "schema_version": "1.0",
        "strategy": "OPTION_A_CLEANED",
        "train_sources": {"token": [TOKEN_TRAIN], "sequence": SEQUENCE_TRAIN},
        "validation_sources": {"token": [TOKEN_VALIDATION], "sequence": [SEQUENCE_VALIDATION]},
        "protected_sources": PROTECTED,
        "protected_counts": protected_counts,
        "rules": [
            "Protected sets remain unchanged.",
            "Validation removes normalized-text and group overlaps with protected sets.",
            "Train removes overlaps with protected sets and cleaned validation.",
            "Token records use document_label and therefore supervise both heads.",
            "Sequence-only records supervise only the sequence head.",
            "Deduplication occurs within split by normalized text, task, label, and token labels.",
        ],
        "train": distribution(train), "validation": distribution(validation),
        "train_cleaning": train_stats, "validation_cleaning": val_stats,
        "isolation_checks": checks,
        "safe_to_train": True,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = build(Path(args.data_dir), Path(args.output_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
