from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import sys

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.hebert_multitask_model import HeBERTMultiTaskModel


SEQUENCE_LABEL_NAMES = {0: "NON_LEAK", 1: "LEAK"}
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(value: str) -> str:
    value = SAFE_NAME_RE.sub("_", value.strip())
    return value.strip("_") or "unnamed"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def normalize_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid binary label")
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    text = str(value).strip().upper()
    mapping = {
        "0": 0,
        "NON_LEAK": 0,
        "NON-LEAK": 0,
        "SAFE": 0,
        "1": 1,
        "LEAK": 1,
        "SENSITIVE": 1,
    }
    if text not in mapping:
        raise ValueError(f"Unsupported sequence label: {value!r}")
    return mapping[text]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            value["_line_number"] = line_number
            rows.append(value)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv_rows(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def sequence_rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(read_rows(path), 1):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing text in {path}, record {index}")
        raw_label = row.get("label", row.get("sequence_label", row.get("document_label")))
        label = normalize_binary_label(raw_label)
        output.append(
            {
                "row_id": str(row.get("case_id", row.get("id", index))),
                "pair_id": str(row.get("pair_id", "")),
                "text": text,
                "label": label,
                "label_name": SEQUENCE_LABEL_NAMES[label],
                "category": str(row.get("category", "UNKNOWN")),
                "difficulty": str(row.get("difficulty", "unknown")),
                "group_id": str(row.get("group_id", "")),
                "expected_reason": str(row.get("expected_reason", "")),
            }
        )
    return output


def token_rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path), 1):
        tokens = row.get("tokens")
        labels = row.get("ner_tags")
        if not isinstance(tokens, list) or not all(isinstance(x, str) for x in tokens):
            raise ValueError(f"Invalid tokens in {path}, record {index}")
        if not isinstance(labels, list) or len(tokens) != len(labels):
            raise ValueError(f"Invalid ner_tags in {path}, record {index}")
        output.append(
            {
                "row_id": str(row.get("case_id", row.get("id", index))),
                "text": str(row.get("text", " ".join(tokens))),
                "tokens": tokens,
                "ner_tags": [int(x) for x in labels],
                "tag_names": list(row.get("tag_names", [])),
                "category": str(row.get("category", "UNKNOWN")),
                "group_id": str(row.get("group_id", "")),
                "document_label": (
                    normalize_binary_label(row["document_label"])
                    if row.get("document_label") is not None
                    else None
                ),
            }
        )
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(predictions, dtype=int)
    if len(y) != len(pred):
        raise ValueError("labels and predictions have different lengths")
    if len(y) == 0:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "fpr": 0.0,
            "fnr": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    result: dict[str, Any] = {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y, pred)) if len(set(y.tolist())) > 1 else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if scores is not None and len(set(y.tolist())) == 2:
        probability = np.asarray(scores, dtype=float)
        result["roc_auc"] = float(roc_auc_score(y, probability))
        result["average_precision"] = float(average_precision_score(y, probability))
    return result


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Expected sequence logits with shape [N, 2]")
    if len(logits) != len(labels) or len(labels) == 0:
        raise ValueError("Calibration logits and labels must be non-empty and aligned")
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=0.05, max=20.0)
        loss = criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).detach().item())
    return min(20.0, max(0.05, temperature))


def probabilities_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scaled = torch.tensor(logits / temperature, dtype=torch.float32)
    return torch.softmax(scaled, dim=-1)[:, 1].numpy()


def choose_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_recall: float,
    max_fpr: float,
) -> tuple[float, list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for threshold in np.arange(0.01, 0.991, 0.01):
        prediction = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                **binary_metrics(labels.tolist(), prediction.tolist(), probabilities.tolist()),
            }
        )
    strict = [
        row for row in rows
        if row["recall"] >= min_recall and row["fpr"] <= max_fpr
    ]
    if strict:
        best = max(strict, key=lambda r: (r["f1"], r["precision"], r["specificity"]))
        rule = "STRICT_RECALL_AND_FPR"
    else:
        recall_ok = [row for row in rows if row["recall"] >= min_recall]
        if recall_ok:
            best = max(recall_ok, key=lambda r: (r["f1"], -r["fpr"], r["precision"]))
            rule = "RECALL_ONLY_FALLBACK"
        else:
            best = max(rows, key=lambda r: (r["f1"], r["recall"], -r["fpr"]))
            rule = "BEST_F1_FALLBACK"
    return float(best["threshold"]), rows, rule


@dataclass
class ModelSpec:
    name: str
    path: str
    kind: str  # multitask | token_baseline | sequence_baseline


class SequenceTextDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.rows[index]["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


class TokenTextDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        aligned: list[int] = []
        first_subtoken_mask: list[int] = []
        previous = None
        for word_id in word_ids:
            is_first = word_id is not None and word_id != previous
            first_subtoken_mask.append(1 if is_first else 0)
            if is_first and word_id < len(row["ner_tags"]):
                aligned.append(int(row["ner_tags"][word_id]))
            else:
                aligned.append(-100)
            previous = word_id
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(aligned, dtype=torch.long)
        item["first_subtoken_mask"] = torch.tensor(first_subtoken_mask, dtype=torch.bool)
        item["row_index"] = torch.tensor(index, dtype=torch.long)
        return item


class LoadedModel:
    def __init__(self, spec: ModelSpec, device: torch.device) -> None:
        self.spec = spec
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(spec.path, use_fast=True)
        if spec.kind == "multitask":
            self.model = HeBERTMultiTaskModel.from_pretrained(spec.path)
        elif spec.kind == "token_baseline":
            self.model = AutoModelForTokenClassification.from_pretrained(spec.path)
        elif spec.kind == "sequence_baseline":
            self.model = AutoModelForSequenceClassification.from_pretrained(spec.path)
        else:
            raise ValueError(f"Unsupported model kind: {spec.kind}")
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def sequence_logits(
        self,
        rows: list[dict[str, Any]],
        batch_size: int,
        max_length: int,
    ) -> tuple[np.ndarray, float]:
        if self.spec.kind == "token_baseline":
            raise ValueError("Token-only model has no sequence head")
        loader = DataLoader(
            SequenceTextDataset(rows, self.tokenizer, max_length),
            batch_size=batch_size,
            shuffle=False,
        )
        output: list[np.ndarray] = []
        started = time.perf_counter()
        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            result = self.model(**batch)
            logits = (
                result.sequence_logits
                if self.spec.kind == "multitask"
                else result.logits
            )
            output.append(logits.detach().cpu().numpy())
        elapsed = time.perf_counter() - started
        if not output:
            return np.empty((0, 2), dtype=float), elapsed
        return np.concatenate(output, axis=0).astype(float), elapsed

    @torch.inference_mode()
    def token_predictions(
        self,
        rows: list[dict[str, Any]],
        batch_size: int,
        max_length: int,
    ) -> tuple[list[int], list[int], list[int], float]:
        if self.spec.kind == "sequence_baseline":
            raise ValueError("Sequence-only model has no token head")
        loader = DataLoader(
            TokenTextDataset(rows, self.tokenizer, max_length),
            batch_size=batch_size,
            shuffle=False,
        )
        gold: list[int] = []
        predicted: list[int] = []
        row_indices: list[int] = []
        started = time.perf_counter()
        for batch in loader:
            labels = batch.pop("labels")
            first_mask = batch.pop("first_subtoken_mask")
            batch_rows = batch.pop("row_index")
            batch = {key: value.to(self.device) for key, value in batch.items()}
            result = self.model(**batch)
            logits = (
                result.token_logits
                if self.spec.kind == "multitask"
                else result.logits
            )
            prediction = logits.argmax(-1).cpu()
            valid = labels != -100
            for i in range(labels.shape[0]):
                mask = valid[i] & first_mask[i]
                count = int(mask.sum().item())
                gold.extend(labels[i][mask].tolist())
                predicted.extend(prediction[i][mask].tolist())
                row_indices.extend([int(batch_rows[i].item())] * count)
        return gold, predicted, row_indices, time.perf_counter() - started


def token_micro_metrics(gold: Sequence[int], predicted: Sequence[int]) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("Token labels and predictions have different lengths")
    tp = sum(y == p and y != 0 for y, p in zip(gold, predicted))
    fp = sum(p != 0 and p != y for y, p in zip(gold, predicted))
    fn = sum(y != 0 and p != y for y, p in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(y == p for y, p in zip(gold, predicted)) / len(gold) if gold else 0.0
    return {
        "tokens": len(gold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def entity_type(label_name: str) -> str:
    if label_name == "O":
        return "O"
    if "-" in label_name:
        return label_name.split("-", 1)[1]
    return label_name


def token_per_type_metrics(
    gold: Sequence[int],
    predicted: Sequence[int],
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    types = sorted(
        {
            entity_type(name)
            for name in id2label.values()
            if entity_type(name) != "O"
        }
    )
    rows: list[dict[str, Any]] = []
    for current_type in types:
        gold_binary = [
            1 if entity_type(id2label.get(int(label), str(label))) == current_type else 0
            for label in gold
        ]
        pred_binary = [
            1 if entity_type(id2label.get(int(label), str(label))) == current_type else 0
            for label in predicted
        ]
        current = binary_metrics(gold_binary, pred_binary)
        rows.append({"entity_type": current_type, **current})
    return rows


def calibration_config(
    model_name: str,
    calibration_path: Path,
    logits: np.ndarray,
    labels: np.ndarray,
    min_recall: float,
    max_fpr: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    temperature = fit_temperature(logits, labels)
    probabilities = probabilities_from_logits(logits, temperature)
    threshold, sweep, selection_rule = choose_threshold(
        labels, probabilities, min_recall, max_fpr
    )
    prediction = (probabilities >= threshold).astype(int)
    config = {
        "model": model_name,
        "temperature": temperature,
        "threshold": threshold,
        "selection_set": str(calibration_path),
        "minimum_recall": min_recall,
        "maximum_fpr_target": max_fpr,
        "selection_rule": selection_rule,
        "calibration_metrics": {
            "threshold": threshold,
            **binary_metrics(labels.tolist(), prediction.tolist(), probabilities.tolist()),
        },
    }
    return config, sweep, probabilities


def evaluate_sequence_model(
    loaded: LoadedModel,
    dataset_name: str,
    rows: list[dict[str, Any]],
    temperature: float,
    threshold: float,
    batch_size: int,
    max_length: int,
    output_dir: Path,
) -> dict[str, Any]:
    logits, runtime = loaded.sequence_logits(rows, batch_size, max_length)
    probabilities = probabilities_from_logits(logits, temperature)
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    predictions = (probabilities >= threshold).astype(int)
    current_metrics = binary_metrics(
        labels.tolist(), predictions.tolist(), probabilities.tolist()
    )
    current_metrics.update(
        {
            "temperature": temperature,
            "threshold": threshold,
            "runtime_seconds": runtime,
            "runtime_ms_per_sample": runtime * 1000 / max(1, len(rows)),
        }
    )

    details: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row, probability, prediction in zip(rows, probabilities, predictions):
        item = {
            **row,
            "model": loaded.spec.name,
            "probability_leak": round(float(probability), 8),
            "prediction": int(prediction),
            "prediction_name": SEQUENCE_LABEL_NAMES[int(prediction)],
            "correct": bool(int(prediction) == int(row["label"])),
        }
        details.append(item)
        if not item["correct"]:
            errors.append(item)

    prefix = f"sequence_{safe_name(dataset_name)}_{safe_name(loaded.spec.name)}"
    write_csv(output_dir / f"{prefix}_predictions.csv", details)
    write_csv(output_dir / f"{prefix}_errors.csv", errors)

    by_category: list[dict[str, Any]] = []
    categories: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        categories[row["category"]].append(index)
    for category, indexes in sorted(categories.items()):
        y = labels[indexes]
        p = predictions[indexes]
        s = probabilities[indexes]
        by_category.append(
            {"category": category, **binary_metrics(y.tolist(), p.tolist(), s.tolist())}
        )
    write_csv(output_dir / f"{prefix}_by_category.csv", by_category)

    return {
        "task": "sequence",
        "model": loaded.spec.name,
        "dataset": dataset_name,
        **current_metrics,
        "errors": len(errors),
    }


def evaluate_token_model(
    loaded: LoadedModel,
    dataset_name: str,
    rows: list[dict[str, Any]],
    id2label: dict[int, str],
    batch_size: int,
    max_length: int,
    output_dir: Path,
) -> dict[str, Any]:
    gold, predicted, row_indices, runtime = loaded.token_predictions(
        rows, batch_size, max_length
    )
    metrics = token_micro_metrics(gold, predicted)
    metrics.update(
        {
            "runtime_seconds": runtime,
            "runtime_ms_per_record": runtime * 1000 / max(1, len(rows)),
        }
    )
    per_type = token_per_type_metrics(gold, predicted, id2label)
    prefix = f"token_{safe_name(dataset_name)}_{safe_name(loaded.spec.name)}"
    write_csv(output_dir / f"{prefix}_per_type.csv", per_type)

    row_errors: dict[int, dict[str, Any]] = {}
    for y, p, row_index in zip(gold, predicted, row_indices):
        if y == p:
            continue
        entry = row_errors.setdefault(
            row_index,
            {
                "row_id": rows[row_index]["row_id"],
                "text": rows[row_index]["text"],
                "category": rows[row_index]["category"],
                "group_id": rows[row_index]["group_id"],
                "model": loaded.spec.name,
                "mismatch_count": 0,
                "gold_entity_tokens": 0,
                "predicted_entity_tokens": 0,
            },
        )
        entry["mismatch_count"] += 1
        entry["gold_entity_tokens"] += int(y != 0)
        entry["predicted_entity_tokens"] += int(p != 0)
    write_csv(output_dir / f"{prefix}_errors.csv", row_errors.values())

    return {
        "task": "token",
        "model": loaded.spec.name,
        "dataset": dataset_name,
        **metrics,
        "error_records": len(row_errors),
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def existing_defaults(candidates: list[tuple[str, str]]) -> list[tuple[str, Path]]:
    return [(name, Path(path)) for name, path in candidates if Path(path).exists()]


def build_report(
    summary_rows: list[dict[str, Any]],
    calibration_values: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    lines = [
        "HeBERT Multi-Task Evaluation and Calibration",
        "=" * 76,
        "",
        "Calibration:",
    ]
    for model, config in calibration_values.items():
        metrics = config["calibration_metrics"]
        lines.extend(
            [
                f"- {model}: temperature={config['temperature']:.6f}, "
                f"threshold={config['threshold']:.2f}, rule={config['selection_rule']}",
                f"  recall={metrics['recall']:.4f}, specificity={metrics['specificity']:.4f}, "
                f"f1={metrics['f1']:.4f}, fpr={metrics['fpr']:.4f}",
            ]
        )
    lines.extend(["", "Evaluation summary:"])
    for row in summary_rows:
        if row["task"] == "sequence":
            lines.append(
                f"- [Sequence] {row['model']} on {row['dataset']}: "
                f"F1={row['f1']:.4f}, recall={row['recall']:.4f}, "
                f"specificity={row['specificity']:.4f}, FP/FN={row['fp']}/{row['fn']}"
            )
        else:
            lines.append(
                f"- [Token] {row['model']} on {row['dataset']}: "
                f"F1={row['f1']:.4f}, recall={row['recall']:.4f}, "
                f"precision={row['precision']:.4f}, FP/FN={row['fp']}/{row['fn']}"
            )
    lines.extend(
        [
            "",
            "Important:",
            "- Calibration was selected only on the calibration dataset.",
            "- Test, challenge, real-world and blind datasets were evaluation-only.",
            "- A perfect synthetic validation score is not sufficient evidence of real-world generalization.",
            "- Compare the Multi-Task best checkpoint against the original Token V3 and Sequence V3 baselines.",
        ]
    )
    (output_dir / "evaluation_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and calibrate HeBERT Multi-Task checkpoints and V3 baselines."
    )
    parser.add_argument(
        "--multitask-model",
        action="append",
        help="Repeatable NAME=PATH, e.g. best=models/hebert_multitask_v1/best",
    )
    parser.add_argument("--token-baseline")
    parser.add_argument("--sequence-baseline")
    parser.add_argument("--calibration", default="data/hebert_sequence_calibration_v3.jsonl")
    parser.add_argument(
        "--token-dataset",
        action="append",
        help="Repeatable NAME=PATH",
    )
    parser.add_argument(
        "--sequence-dataset",
        action="append",
        help="Repeatable NAME=PATH",
    )
    parser.add_argument("--label-schema", default="data/hebert_label_schema.json")
    parser.add_argument("--output-dir", default="output/hebert_multitask_evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--min-recall", type=float, default=0.86)
    parser.add_argument("--max-fpr", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    multitask_inputs = args.multitask_model or [
        "best=models/hebert_multitask_v1/best",
        "final=models/hebert_multitask_v1/final",
    ]
    multitask_specs = [
        ModelSpec(name, str(path), "multitask")
        for name, path in map(parse_named_path, multitask_inputs)
    ]
    for spec in multitask_specs:
        if not Path(spec.path).exists():
            raise FileNotFoundError(f"Missing Multi-Task model: {spec.path}")

    token_baseline_path = args.token_baseline or "models/hebert_typed_leak_detector_v3"
    sequence_baseline_path = args.sequence_baseline or "models/hebert_sequence_classifier_v3"
    token_spec = (
        ModelSpec("token_v3", token_baseline_path, "token_baseline")
        if Path(token_baseline_path).exists()
        else None
    )
    sequence_spec = (
        ModelSpec("sequence_v3", sequence_baseline_path, "sequence_baseline")
        if Path(sequence_baseline_path).exists()
        else None
    )

    calibration_path = Path(args.calibration)
    calibration_rows = sequence_rows(calibration_path)
    calibration_labels = np.asarray([row["label"] for row in calibration_rows], dtype=int)

    if args.token_dataset:
        token_datasets = list(map(parse_named_path, args.token_dataset))
    else:
        token_datasets = existing_defaults(
            [
                ("token_test_v3", "data/hebert_token_classification_test_v3_targeted.jsonl"),
                ("token_challenge_v3", "data/hebert_targeted_challenge_v3.jsonl"),
                ("token_challenge_v2", "data/hebert_targeted_challenge_v2.jsonl"),
            ]
        )

    if args.sequence_dataset:
        sequence_datasets = list(map(parse_named_path, args.sequence_dataset))
    else:
        sequence_datasets = existing_defaults(
            [
                ("sequence_test", "data/hebert_sequence_test.jsonl"),
                ("development_v3", "data/hebert_sequence_development_v3.jsonl"),
                ("real_world_v3", "data/hebert_sequence_real_world_test_v3.jsonl"),
                ("targeted_challenge_v3", "data/hebert_targeted_challenge_v3.jsonl"),
                ("blind_v6", "data/svm_blind_test_v6.csv"),
                ("definition_challenge", "data/real_world_definition_challenge.csv"),
            ]
        )

    schema = read_json(Path(args.label_schema))
    raw_id2label = schema.get("id2label", {})
    id2label = {int(key): str(value) for key, value in raw_id2label.items()}

    summary_rows: list[dict[str, Any]] = []
    calibration_values: dict[str, dict[str, Any]] = {}
    loaded_models: list[LoadedModel] = []

    try:
        for spec in multitask_specs:
            loaded_models.append(LoadedModel(spec, device))
        if token_spec:
            loaded_models.append(LoadedModel(token_spec, device))
        if sequence_spec:
            loaded_models.append(LoadedModel(sequence_spec, device))

        sequence_capable = [
            model for model in loaded_models if model.spec.kind != "token_baseline"
        ]
        for loaded in sequence_capable:
            logits, calibration_runtime = loaded.sequence_logits(
                calibration_rows, args.batch_size, args.max_length
            )
            config, sweep, _ = calibration_config(
                loaded.spec.name,
                calibration_path,
                logits,
                calibration_labels,
                args.min_recall,
                args.max_fpr,
            )
            config["calibration_runtime_seconds"] = calibration_runtime
            calibration_values[loaded.spec.name] = config
            write_json(
                output_dir / f"calibration_{safe_name(loaded.spec.name)}.json",
                config,
            )
            write_csv(
                output_dir / f"threshold_sweep_{safe_name(loaded.spec.name)}.csv",
                sweep,
            )

        for dataset_name, dataset_path in token_datasets:
            rows = token_rows(dataset_path)
            for loaded in loaded_models:
                if loaded.spec.kind == "sequence_baseline":
                    continue
                summary_rows.append(
                    evaluate_token_model(
                        loaded,
                        dataset_name,
                        rows,
                        id2label,
                        args.batch_size,
                        args.max_length,
                        output_dir,
                    )
                )

        for dataset_name, dataset_path in sequence_datasets:
            rows = sequence_rows(dataset_path)
            for loaded in sequence_capable:
                config = calibration_values[loaded.spec.name]
                summary_rows.append(
                    evaluate_sequence_model(
                        loaded,
                        dataset_name,
                        rows,
                        config["temperature"],
                        config["threshold"],
                        args.batch_size,
                        args.max_length,
                        output_dir,
                    )
                )

        write_json(
            output_dir / "evaluation_summary.json",
            {
                "device": str(device),
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "calibration": calibration_values,
                "results": summary_rows,
            },
        )
        write_csv(output_dir / "evaluation_summary.csv", summary_rows)

        rankings: dict[str, Any] = {}
        for task in ("token", "sequence"):
            task_rows = [row for row in summary_rows if row["task"] == task]
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in task_rows:
                grouped[row["model"]].append(float(row["f1"]))
            rankings[task] = sorted(
                (
                    {
                        "model": model,
                        "mean_f1": float(np.mean(scores)),
                        "datasets": len(scores),
                    }
                    for model, scores in grouped.items()
                ),
                key=lambda row: row["mean_f1"],
                reverse=True,
            )
        write_json(output_dir / "model_rankings.json", rankings)
        build_report(summary_rows, calibration_values, output_dir)
        print(
            json.dumps(
                {
                    "status": "success",
                    "device": str(device),
                    "models": [model.spec.name for model in loaded_models],
                    "token_datasets": [name for name, _ in token_datasets],
                    "sequence_datasets": [name for name, _ in sequence_datasets],
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
