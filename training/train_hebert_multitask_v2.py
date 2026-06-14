from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hebert_multitask_model import HeBERTMultiTaskModel


HARD_SPLIT_NAME = "hard_augmentation_v2"
HARD_SOURCE_STYLE = "new_targeted_hard_augmentation"
PROJECT_ROOT = SCRIPT_DIR.parent
PRODUCTION_MODEL_DIR = PROJECT_ROOT / "models" / "hebert_multitask_v2" / "best"


class MultiTaskDataset(Dataset):
    """JSONL-backed dataset with deterministic optional sampling."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_length: int,
        limit: int | None = None,
        sample_seed: int = 42,
        only_hard: bool = False,
        sample_hard_ratio: float | None = None,
    ) -> None:
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL row in {path} is not an object")
                    if only_hard and not is_hard_record(value):
                        continue
                    records.append(value)

        self.total_available_records = len(records)
        self.records = deterministic_mixed_sample(
            records,
            limit=limit,
            seed=sample_seed,
            hard_ratio=sample_hard_ratio,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hard_flags = [is_hard_record(record) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        has_tokens = (
            record.get("task") in {"token", "both"}
            and isinstance(record.get("tokens"), list)
            and isinstance(record.get("ner_tags"), list)
        )

        if has_tokens:
            encoded = self.tokenizer(
                record["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            word_ids = encoded.word_ids(batch_index=0)
            original_labels = record["ner_tags"]
            aligned: list[int] = []
            previous = None
            for word_id in word_ids:
                if word_id is None or word_id == previous:
                    aligned.append(-100)
                elif word_id < len(original_labels):
                    aligned.append(int(original_labels[word_id]))
                else:
                    aligned.append(-100)
                previous = word_id
        else:
            encoded = self.tokenizer(
                record["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            aligned = [-100] * self.max_length

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["token_labels"] = torch.tensor(aligned, dtype=torch.long)

        sequence_label = record.get("sequence_label")
        if sequence_label not in (0, 1):
            sequence_label = record.get("label")
        item["sequence_labels"] = torch.tensor(
            sequence_label if sequence_label in (0, 1) else -100,
            dtype=torch.long,
        )
        return item


def is_hard_record(record: dict[str, Any]) -> bool:
    return (
        record.get("split") == HARD_SPLIT_NAME
        or record.get("source_style") == HARD_SOURCE_STYLE
        or record.get("copied_from_blind_set") is False
    )


def deterministic_mixed_sample(
    records: list[dict[str, Any]],
    limit: int | None,
    seed: int,
    hard_ratio: float | None = None,
) -> list[dict[str, Any]]:
    """Create a deterministic subset while preserving hard examples in smoke runs."""
    if limit is None or limit >= len(records):
        return list(records)
    if limit <= 0:
        return []

    rng = random.Random(seed)

    if hard_ratio is None:
        selected_indices = rng.sample(range(len(records)), limit)
        return [records[index] for index in selected_indices]

    if not 0.0 <= hard_ratio <= 1.0:
        raise ValueError("hard_ratio must be between 0 and 1")

    hard_records = [record for record in records if is_hard_record(record)]
    base_records = [record for record in records if not is_hard_record(record)]

    requested_hard = round(limit * hard_ratio)
    hard_count = min(requested_hard, len(hard_records))
    base_count = min(limit - hard_count, len(base_records))

    remaining = limit - hard_count - base_count
    if remaining > 0:
        extra_hard = min(remaining, len(hard_records) - hard_count)
        hard_count += extra_hard
        remaining -= extra_hard
    if remaining > 0:
        extra_base = min(remaining, len(base_records) - base_count)
        base_count += extra_base
        remaining -= extra_base
    if remaining > 0:
        raise ValueError(
            f"Could not sample {limit} records from dataset of {len(records)}"
        )

    selected = (
        rng.sample(hard_records, hard_count)
        + rng.sample(base_records, base_count)
    )
    rng.shuffle(selected)
    return selected


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def token_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    tp = sum(y == p and y != 0 for y, p in zip(labels, predictions))
    fp = sum(p != 0 and p != y for y, p in zip(labels, predictions))
    fn = sum(y != 0 and p != y for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


@torch.inference_mode()
def evaluate(
    model: HeBERTMultiTaskModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    seq_true: list[int] = []
    seq_pred: list[int] = []
    tok_true: list[int] = []
    tok_pred: list[int] = []
    losses: list[float] = []
    token_losses: list[float] = []
    sequence_losses: list[float] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)

        if output.loss is not None:
            losses.append(float(output.loss.detach().cpu()))
        if output.token_loss is not None:
            token_losses.append(float(output.token_loss.detach().cpu()))
        if output.sequence_loss is not None:
            sequence_losses.append(float(output.sequence_loss.detach().cpu()))

        valid_seq = batch["sequence_labels"] != -100
        seq_true.extend(batch["sequence_labels"][valid_seq].cpu().tolist())
        seq_pred.extend(output.sequence_logits.argmax(-1)[valid_seq].cpu().tolist())

        predicted_tokens = output.token_logits.argmax(-1)
        valid_tok = batch["token_labels"] != -100
        tok_true.extend(batch["token_labels"][valid_tok].cpu().tolist())
        tok_pred.extend(predicted_tokens[valid_tok].cpu().tolist())

    return {
        "records": len(loader.dataset),
        "loss": float(np.mean(losses)) if losses else None,
        "token_loss": float(np.mean(token_losses)) if token_losses else None,
        "sequence_loss": float(np.mean(sequence_losses)) if sequence_losses else None,
        "sequence": binary_metrics(seq_true, seq_pred),
        "token": token_metrics(tok_true, tok_pred),
    }


def combined_selection_score(
    overall: dict[str, Any],
    hard: dict[str, Any],
    *,
    minimum_overall_token_f1: float,
    minimum_hard_sequence_recall: float,
    minimum_hard_sequence_specificity: float,
) -> tuple[float, list[str]]:
    score = (
        0.30 * float(overall["token"]["f1"])
        + 0.30 * float(hard["token"]["f1"])
        + 0.20 * float(overall["sequence"]["f1"])
        + 0.20 * float(hard["sequence"]["f1"])
    )

    failed_constraints: list[str] = []
    if float(overall["token"]["f1"]) < minimum_overall_token_f1:
        failed_constraints.append("overall_token_f1")
    if float(hard["sequence"]["recall"]) < minimum_hard_sequence_recall:
        failed_constraints.append("hard_sequence_recall")
    if float(hard["sequence"]["specificity"]) < minimum_hard_sequence_specificity:
        failed_constraints.append("hard_sequence_specificity")

    if failed_constraints:
        score -= 0.10 * len(failed_constraints)

    return score, failed_constraints


def freeze_bottom_encoder_layers(
    model: HeBERTMultiTaskModel,
    trainable_top_layers: int,
) -> dict[str, int]:
    layers = list(model.bert.encoder.layer)
    total_layers = len(layers)

    if trainable_top_layers < 0 or trainable_top_layers > total_layers:
        raise ValueError(
            f"trainable_top_layers must be between 0 and {total_layers}"
        )

    frozen_layers = total_layers - trainable_top_layers

    # Keep embeddings frozen to preserve the pretrained/token knowledge.
    for parameter in model.bert.embeddings.parameters():
        parameter.requires_grad = False

    for index, layer in enumerate(layers):
        trainable = index >= frozen_layers
        for parameter in layer.parameters():
            parameter.requires_grad = trainable

    for parameter in model.token_classifier.parameters():
        parameter.requires_grad = True
    for parameter in model.sequence_classifier.parameters():
        parameter.requires_grad = True

    return {
        "total_encoder_layers": total_layers,
        "frozen_encoder_layers": frozen_layers,
        "trainable_encoder_layers": trainable_top_layers,
    }


def build_optimizer(
    model: HeBERTMultiTaskModel,
    *,
    encoder_lr: float,
    token_head_lr: float,
    sequence_head_lr: float,
    weight_decay: float,
) -> AdamW:
    parameter_groups: list[dict[str, Any]] = []

    encoder_parameters = [
        parameter
        for parameter in model.bert.parameters()
        if parameter.requires_grad
    ]
    token_parameters = [
        parameter
        for parameter in model.token_classifier.parameters()
        if parameter.requires_grad
    ]
    sequence_parameters = [
        parameter
        for parameter in model.sequence_classifier.parameters()
        if parameter.requires_grad
    ]

    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "group_name": "encoder",
            }
        )
    if token_parameters:
        parameter_groups.append(
            {
                "params": token_parameters,
                "lr": token_head_lr,
                "weight_decay": weight_decay,
                "group_name": "token_head",
            }
        )
    if sequence_parameters:
        parameter_groups.append(
            {
                "params": sequence_parameters,
                "lr": sequence_head_lr,
                "weight_decay": weight_decay,
                "group_name": "sequence_head",
            }
        )

    if not parameter_groups:
        raise RuntimeError("No trainable parameters were found")

    return AdamW(parameter_groups)


def make_balanced_train_loader(
    dataset: MultiTaskDataset,
    batch_size: int,
    seed: int,
    epoch_index: int,
    hard_ratio: float,
) -> DataLoader:
    hard_count = sum(dataset.hard_flags)
    base_count = len(dataset) - hard_count

    if hard_count == 0 or base_count == 0:
        raise ValueError(
            f"Balanced sampling requires both base and hard records. "
            f"base={base_count}, hard={hard_count}"
        )

    if not 0.0 < hard_ratio < 1.0:
        raise ValueError("hard_ratio must be between 0 and 1")

    hard_weight = hard_ratio / hard_count
    base_weight = (1.0 - hard_ratio) / base_count
    sample_weights = [
        hard_weight if is_hard else base_weight
        for is_hard in dataset.hard_flags
    ]

    generator = torch.Generator()
    generator.manual_seed(seed + epoch_index)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    model: HeBERTMultiTaskModel,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    name: str,
    metadata: dict[str, Any],
    trainer_state: dict[str, Any],
) -> None:
    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    torch.save(optimizer.state_dict(), target / "optimizer.pt")
    torch.save(scheduler.state_dict(), target / "scheduler.pt")
    torch.save(capture_rng_state(), target / "rng_state.pt")
    (target / "trainer_state.json").write_text(
        json.dumps(trainer_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target / "training_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_resume_state(
    checkpoint: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    required = [
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        checkpoint / "rng_state.pt",
        checkpoint / "trainer_state.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint does not contain full resume state: " + ", ".join(missing)
        )

    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location="cpu")
    )
    scheduler.load_state_dict(
        torch.load(checkpoint / "scheduler.pt", map_location="cpu")
    )
    trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    rng_state = torch.load(
        checkpoint / "rng_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    restore_rng_state(rng_state)
    return trainer_state


def _resolve_existing_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")
    return target.resolve()


def validate_output_directory_safety(output_dir: str | Path) -> Path:
    """Reject output paths that could overwrite the production model."""

    output_path = Path(output_dir).resolve()
    production_path = PRODUCTION_MODEL_DIR.resolve()
    if output_path == production_path or production_path in output_path.parents:
        raise ValueError(
            "Output directory must not be the production model directory "
            "or a child of it."
        )
    return output_path


def positive_int(value: str) -> int:
    """Argparse type for positive integer CLI values."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def optimizer_update_count(
    micro_batch_count: int,
    gradient_accumulation_steps: int,
) -> int:
    """Number of optimizer updates for a micro-batch count."""

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    return math.ceil(micro_batch_count / gradient_accumulation_steps)


def should_apply_optimizer_update(
    micro_batch_index: int,
    micro_batch_count: int,
    gradient_accumulation_steps: int,
) -> bool:
    """Return true at complete or final partial accumulation windows."""

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    is_window_complete = (micro_batch_index + 1) % gradient_accumulation_steps == 0
    is_final_micro_batch = micro_batch_index + 1 == micro_batch_count
    return is_window_complete or is_final_micro_batch


def _load_checkpoint_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    safetensors_path = checkpoint / "model.safetensors"
    pytorch_path = checkpoint / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu")
    if pytorch_path.exists():
        value = torch.load(pytorch_path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            return value
    raise FileNotFoundError(
        f"No supported model weights file found in {checkpoint}"
    )


def _validate_label_schema_compatibility(
    checkpoint_config: Any,
    label_schema_path: str | Path | None,
) -> dict[str, Any]:
    if label_schema_path is None:
        return {
            "label_schema_path": None,
            "label_schema_hash": None,
            "compatible": True,
            "issues": [],
        }

    schema_path = _resolve_existing_path(label_schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    issues: list[str] = []
    schema_label2id = schema.get("label2id", {})
    config_label2id = getattr(checkpoint_config, "tokenlabel2id", None) or getattr(
        checkpoint_config,
        "label2id",
        {},
    )
    if schema_label2id and dict(config_label2id) != dict(schema_label2id):
        issues.append("token label2id mapping differs from label schema")
    return {
        "label_schema_path": str(schema_path),
        "label_schema_hash": hashlib_sha256(schema_path),
        "compatible": not issues,
        "issues": issues,
    }


def hashlib_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_weights_only_checkpoint(
    checkpoint: str | Path,
    label_schema_path: str | Path | None = None,
    *,
    load_weights: bool = True,
) -> dict[str, Any]:
    """Validate a multitask checkpoint for weights-only initialization."""

    checkpoint_path = _resolve_existing_path(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint_path)
    issues: list[str] = []
    architecture = list(getattr(config, "architectures", []) or [])
    if "HeBERTMultiTaskModel" not in architecture:
        issues.append("checkpoint architecture is not HeBERTMultiTaskModel")
    if int(getattr(config, "num_sequence_labels", -1)) != 2:
        issues.append("num_sequence_labels must be 2")
    token_labels = getattr(config, "tokenlabel2id", None) or getattr(
        config,
        "label2id",
        {},
    )
    if int(getattr(config, "num_token_labels", -1)) != len(token_labels):
        issues.append("num_token_labels does not match token label mapping")
    if not (checkpoint_path / "tokenizer.json").exists() and not (
        checkpoint_path / "vocab.txt"
    ).exists():
        issues.append("tokenizer files are missing")

    label_schema = _validate_label_schema_compatibility(
        config,
        label_schema_path,
    )
    issues.extend(label_schema["issues"])

    missing_parameters: list[str] = []
    unexpected_parameters: list[str] = []
    shape_mismatches: list[str] = []
    parameter_count = 0
    if load_weights:
        model = HeBERTMultiTaskModel.from_pretrained(checkpoint_path)
        model_state = model.state_dict()
        checkpoint_state = _load_checkpoint_state_dict(checkpoint_path)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model_keys = set(model_state)
        checkpoint_keys = set(checkpoint_state)
        missing_parameters = sorted(model_keys - checkpoint_keys)
        unexpected_parameters = sorted(checkpoint_keys - model_keys)
        for key in sorted(model_keys & checkpoint_keys):
            if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape):
                shape_mismatches.append(
                    f"{key}: model {tuple(model_state[key].shape)} "
                    f"checkpoint {tuple(checkpoint_state[key].shape)}"
                )
        if missing_parameters:
            issues.append("missing model parameters")
        if shape_mismatches:
            issues.append("shape mismatches")

    return {
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "num_sequence_labels": int(getattr(config, "num_sequence_labels", -1)),
        "num_token_labels": int(getattr(config, "num_token_labels", -1)),
        "token_label_count": len(token_labels),
        "tokenizer_present": not (
            "tokenizer files are missing" in issues
        ),
        "parameter_count": parameter_count,
        "missing_parameters": missing_parameters,
        "unexpected_parameters": unexpected_parameters,
        "shape_mismatches": shape_mismatches,
        "label_schema": label_schema,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "trainer_state_loaded": False,
        "rng_state_loaded": False,
        "compatible": not issues,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--token-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=1,
    )
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--token-loss-weight", type=float, default=1.5)
    parser.add_argument("--sequence-loss-weight", type=float, default=1.0)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-6)
    parser.add_argument("--token-head-learning-rate", type=float, default=3e-6)
    parser.add_argument("--sequence-head-learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--trainable-top-layers", type=int, default=4)
    parser.add_argument("--hard-sampling-ratio", type=float, default=0.30)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--minimum-overall-token-f1", type=float, default=0.90)
    parser.add_argument("--minimum-hard-sequence-recall", type=float, default=0.85)
    parser.add_argument(
        "--minimum-hard-sequence-specificity",
        type=float,
        default=0.70,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--hard-validation-limit", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--initialize-from-model")
    parser.add_argument("--label-schema", default="data/hebert_label_schema.json")
    parser.add_argument(
        "--dry-run-initialize-only",
        action="store_true",
        help=(
            "Validate datasets/checkpoint and initialize fresh optimizer/scheduler "
            "without entering the training loop or saving weights."
        ),
    )
    args = parser.parse_args()

    if args.resume_from_checkpoint and args.initialize_from_model:
        raise ValueError(
            "--resume-from-checkpoint and --initialize-from-model are mutually exclusive"
        )
    if not args.resume_from_checkpoint and not args.initialize_from_model and not args.token_checkpoint:
        raise ValueError(
            "--token-checkpoint is required unless --resume-from-checkpoint "
            "or --initialize-from-model is provided"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = validate_output_directory_safety(args.output_dir)
    if not args.dry_run_initialize_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = (
        args.resume_from_checkpoint
        or args.initialize_from_model
        or args.token_checkpoint
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    if args.resume_from_checkpoint:
        model = HeBERTMultiTaskModel.from_pretrained(args.resume_from_checkpoint)
    elif args.initialize_from_model:
        compatibility = validate_model_weights_only_checkpoint(
            args.initialize_from_model,
            args.label_schema,
            load_weights=True,
        )
        if not compatibility["compatible"]:
            raise ValueError(
                "Weights-only checkpoint is incompatible: "
                + json.dumps(compatibility["issues"], ensure_ascii=False)
            )
        model = HeBERTMultiTaskModel.from_pretrained(args.initialize_from_model)
    else:
        model = HeBERTMultiTaskModel.from_token_checkpoint(
            args.token_checkpoint,
            token_loss_weight=args.token_loss_weight,
            sequence_loss_weight=args.sequence_loss_weight,
        )

    freeze_summary = freeze_bottom_encoder_layers(
        model,
        trainable_top_layers=args.trainable_top_layers,
    )
    model.to(device)

    train_ds = MultiTaskDataset(
        args.train,
        tokenizer,
        args.max_length,
        args.train_limit,
        sample_seed=args.seed,
        sample_hard_ratio=args.hard_sampling_ratio,
    )
    overall_val_ds = MultiTaskDataset(
        args.validation,
        tokenizer,
        args.max_length,
        args.validation_limit,
        sample_seed=args.seed + 1,
    )
    hard_val_ds = MultiTaskDataset(
        args.validation,
        tokenizer,
        args.max_length,
        args.hard_validation_limit,
        sample_seed=args.seed + 2,
        only_hard=True,
    )

    if len(hard_val_ds) == 0:
        raise RuntimeError(
            "No hard validation records were found. "
            "Expected records with split=hard_augmentation_v2 or "
            "source_style=new_targeted_hard_augmentation."
        )

    overall_val_loader = DataLoader(
        overall_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )
    hard_val_loader = DataLoader(
        hard_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    micro_batches_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    optimizer_updates_per_epoch = optimizer_update_count(
        micro_batches_per_epoch,
        args.gradient_accumulation_steps,
    )
    total_steps = max(1, optimizer_updates_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    optimizer = build_optimizer(
        model,
        encoder_lr=args.encoder_learning_rate,
        token_head_lr=args.token_head_learning_rate,
        sequence_head_lr=args.sequence_head_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
    )

    best_score = -math.inf
    global_step = 0
    history: list[dict[str, Any]] = []
    start_epoch = 0
    start_batch_index = 0
    evaluations_without_improvement = 0

    if args.dry_run_initialize_only:
        report = {
            "mode": (
                "resume"
                if args.resume_from_checkpoint
                else "weights_only"
                if args.initialize_from_model
                else "token_checkpoint"
            ),
            "device": str(device),
            "output_dir": str(output_dir),
            "train_records": len(train_ds),
            "validation_records": len(overall_val_ds),
            "hard_validation_records": len(hard_val_ds),
            "micro_batches_per_epoch": micro_batches_per_epoch,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "scheduler_total_steps": total_steps,
            "effective_batch_size": effective_batch_size,
            "global_step": global_step,
            "start_epoch": start_epoch,
            "start_batch_index": start_batch_index,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "trainer_state_restored": False,
            "rng_state_restored": False,
            "parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
            "production_output_protected": True,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.resume_from_checkpoint:
        resume_state = load_resume_state(
            Path(args.resume_from_checkpoint),
            optimizer,
            scheduler,
        )
        expected = {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "train_records": len(train_ds),
            "validation_records": len(overall_val_ds),
            "hard_validation_records": len(hard_val_ds),
            "total_steps": total_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "hard_sampling_ratio": args.hard_sampling_ratio,
            "trainable_top_layers": args.trainable_top_layers,
        }
        mismatches = {
            key: (resume_state.get(key), value)
            for key, value in expected.items()
            if resume_state.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Resume arguments do not match checkpoint: {mismatches}"
            )

        best_score = float(resume_state.get("best_score", -math.inf))
        global_step = int(resume_state.get("global_step", 0))
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state.get("next_epoch_index", 0))
        start_batch_index = int(resume_state.get("next_batch_index", 0))
        evaluations_without_improvement = int(
            resume_state.get("evaluations_without_improvement", 0)
        )

    started = time.time()
    stop_training = False
    micro_batch_step = 0

    for epoch_index in range(start_epoch, args.epochs):
        train_loader = make_balanced_train_loader(
            train_ds,
            batch_size=args.batch_size,
            seed=args.seed,
            epoch_index=epoch_index,
            hard_ratio=args.hard_sampling_ratio,
        )
        model.train()
        micro_batch_count = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        accumulation_loss_sum = 0.0
        accumulation_token_loss_sum = 0.0
        accumulation_sequence_loss_sum = 0.0
        accumulation_micro_batches = 0

        for batch_index, batch in enumerate(train_loader):
            if epoch_index == start_epoch and batch_index < start_batch_index:
                continue

            micro_batch_step += 1
            batch = {key: value.to(device) for key, value in batch.items()}

            output = model(**batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError(
                    f"Invalid loss at micro-batch {micro_batch_step}: {output.loss}"
                )

            original_loss = output.loss
            backward_loss = original_loss / args.gradient_accumulation_steps
            accumulation_loss_sum += float(original_loss.detach().cpu())
            if output.token_loss is not None:
                accumulation_token_loss_sum += float(output.token_loss.detach().cpu())
            if output.sequence_loss is not None:
                accumulation_sequence_loss_sum += float(
                    output.sequence_loss.detach().cpu()
                )
            accumulation_micro_batches += 1

            backward_loss.backward()
            if not should_apply_optimizer_update(
                batch_index,
                micro_batch_count,
                args.gradient_accumulation_steps,
            ):
                continue

            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            update_loss = accumulation_loss_sum / accumulation_micro_batches
            update_token_loss = (
                accumulation_token_loss_sum / accumulation_micro_batches
                if output.token_loss is not None
                else None
            )
            update_sequence_loss = (
                accumulation_sequence_loss_sum / accumulation_micro_batches
                if output.sequence_loss is not None
                else None
            )
            accumulation_loss_sum = 0.0
            accumulation_token_loss_sum = 0.0
            accumulation_sequence_loss_sum = 0.0
            accumulation_micro_batches = 0

            should_eval = (
                global_step % args.eval_every == 0
                or global_step == total_steps
            )
            if not should_eval:
                continue

            overall_metrics = evaluate(
                model,
                overall_val_loader,
                device,
            )
            hard_metrics = evaluate(
                model,
                hard_val_loader,
                device,
            )
            selection_score, failed_constraints = combined_selection_score(
                overall_metrics,
                hard_metrics,
                minimum_overall_token_f1=args.minimum_overall_token_f1,
                minimum_hard_sequence_recall=args.minimum_hard_sequence_recall,
                minimum_hard_sequence_specificity=(
                    args.minimum_hard_sequence_specificity
                ),
            )

            entry = {
                "epoch": epoch_index + 1,
                "step": global_step,
                "micro_batch_step": micro_batch_step,
                "optimizer_update_step": global_step,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "train_loss": update_loss,
                "scaled_backward_loss": float(backward_loss.detach().cpu()),
                "train_token_loss": update_token_loss,
                "train_sequence_loss": update_sequence_loss,
                "overall_validation": overall_metrics,
                "hard_validation": hard_metrics,
                "selection_score": selection_score,
                "failed_constraints": failed_constraints,
            }
            history.append(entry)
            print(json.dumps(entry, ensure_ascii=False))

            improved = selection_score > best_score + args.minimum_improvement
            if improved:
                best_score = selection_score
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1

            next_epoch_index = epoch_index
            next_batch_index = batch_index + 1
            if next_batch_index >= len(train_loader):
                next_epoch_index = epoch_index + 1
                next_batch_index = 0

            trainer_state = {
                "global_step": global_step,
                "best_score": best_score,
                "history": history,
                "next_epoch_index": next_epoch_index,
                "next_batch_index": next_batch_index,
                "evaluations_without_improvement": evaluations_without_improvement,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "max_length": args.max_length,
                "train_records": len(train_ds),
                "validation_records": len(overall_val_ds),
                "hard_validation_records": len(hard_val_ds),
                "micro_batches_per_epoch": micro_batches_per_epoch,
                "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
                "total_steps": total_steps,
                "hard_sampling_ratio": args.hard_sampling_ratio,
                "trainable_top_layers": args.trainable_top_layers,
            }

            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                output_dir,
                f"checkpoint-{global_step}",
                entry,
                trainer_state,
            )

            if improved:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    output_dir,
                    "best",
                    entry,
                    trainer_state,
                )

            if (
                args.early_stopping_patience > 0
                and evaluations_without_improvement
                >= args.early_stopping_patience
            ):
                stop_training = True
                break

            model.train()

        start_batch_index = 0
        if stop_training:
            break

    summary = {
        "device": str(device),
        "train_records": len(train_ds),
        "hard_train_records": sum(train_ds.hard_flags),
        "base_train_records": len(train_ds) - sum(train_ds.hard_flags),
        "validation_records": len(overall_val_ds),
        "hard_validation_records": len(hard_val_ds),
        "steps": global_step,
        "micro_batch_steps": micro_batch_step,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
        "total_optimizer_updates": total_steps,
        "warmup_steps": warmup_steps,
        "best_selection_score": best_score,
        "stopped_early": stop_training,
        "runtime_seconds": time.time() - started,
        "freeze_summary": freeze_summary,
        "optimizer_learning_rates": {
            "encoder": args.encoder_learning_rate,
            "token_head": args.token_head_learning_rate,
            "sequence_head": args.sequence_head_learning_rate,
        },
        "hard_sampling_ratio": args.hard_sampling_ratio,
        "history": history,
    }

    final_state = {
        "global_step": global_step,
        "best_score": best_score,
        "history": history,
        "next_epoch_index": args.epochs if not stop_training else epoch_index,
        "next_batch_index": 0,
        "evaluations_without_improvement": evaluations_without_improvement,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "max_length": args.max_length,
        "train_records": len(train_ds),
        "validation_records": len(overall_val_ds),
        "hard_validation_records": len(hard_val_ds),
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
        "total_steps": total_steps,
        "hard_sampling_ratio": args.hard_sampling_ratio,
        "trainable_top_layers": args.trainable_top_layers,
    }

    save_checkpoint(
        model,
        tokenizer,
        optimizer,
        scheduler,
        output_dir,
        "final",
        summary,
        final_state,
    )

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
