from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from hebert_multitask_model import HeBERTMultiTaskModel


class MultiTaskDataset(Dataset):
    """JSONL-backed dataset with deterministic optional sampling."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_length: int,
        limit: int | None = None,
        sample_seed: int = 42,
    ) -> None:
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))

        self.total_available_records = len(records)
        self.records = deterministic_sample(records, limit=limit, seed=sample_seed)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        has_tokens = (
            record.get("task") in {"token", "both"}
            and isinstance(record.get("tokens"), list)
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
        seq_label = record.get("sequence_label")
        item["sequence_labels"] = torch.tensor(
            seq_label if seq_label in (0, 1) else -100,
            dtype=torch.long,
        )
        return item


def deterministic_sample(
    records: list[dict[str, Any]],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic random subset instead of the first N records."""
    if limit is None or limit >= len(records):
        return list(records)
    if limit <= 0:
        return []

    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(records)), limit)
    return [records[index] for index in selected_indices]


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
    # Micro metrics for entity tokens only; O is label 0.
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

    sequence = binary_metrics(seq_true, seq_pred)
    token = token_metrics(tok_true, tok_pred)
    score = 0.5 * float(sequence["f1"]) + 0.5 * float(token["f1"])
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "token_loss": float(np.mean(token_losses)) if token_losses else None,
        "sequence_loss": (
            float(np.mean(sequence_losses)) if sequence_losses else None
        ),
        "sequence": sequence,
        "token": token,
        "selection_score": score,
    }


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
    """Save model plus all state required for an exact training resume."""
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

    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu"))
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu"))
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


def make_train_loader(
    dataset: MultiTaskDataset,
    batch_size: int,
    seed: int,
    epoch_index: int,
) -> DataLoader:
    """Create reproducible per-epoch shuffling for exact mid-epoch resume."""
    generator = torch.Generator()
    generator.manual_seed(seed + epoch_index)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--token-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--token-loss-weight", type=float, default=1.0)
    parser.add_argument("--sequence-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = args.resume_from_checkpoint or args.token_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if args.resume_from_checkpoint:
        model = HeBERTMultiTaskModel.from_pretrained(args.resume_from_checkpoint)
    else:
        model = HeBERTMultiTaskModel.from_token_checkpoint(
            args.token_checkpoint,
            token_loss_weight=args.token_loss_weight,
            sequence_loss_weight=args.sequence_loss_weight,
        )
    model.to(device)

    train_ds = MultiTaskDataset(
        args.train,
        tokenizer,
        args.max_length,
        args.train_limit,
        sample_seed=args.seed,
    )
    val_ds = MultiTaskDataset(
        args.validation,
        tokenizer,
        args.max_length,
        args.validation_limit,
        sample_seed=args.seed + 1,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * args.warmup_ratio),
        total_steps,
    )

    best_score = -math.inf
    global_step = 0
    history: list[dict[str, Any]] = []
    start_epoch = 0
    start_batch_index = 0

    if args.resume_from_checkpoint:
        resume_state = load_resume_state(
            Path(args.resume_from_checkpoint), optimizer, scheduler
        )
        expected = {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "train_records": len(train_ds),
            "validation_records": len(val_ds),
            "total_steps": total_steps,
        }
        mismatches = {
            key: (resume_state.get(key), value)
            for key, value in expected.items()
            if resume_state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Resume arguments do not match checkpoint: {mismatches}")

        best_score = float(resume_state.get("best_score", -math.inf))
        global_step = int(resume_state.get("global_step", 0))
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state.get("next_epoch_index", 0))
        start_batch_index = int(resume_state.get("next_batch_index", 0))

    started = time.time()
    for epoch_index in range(start_epoch, args.epochs):
        train_loader = make_train_loader(
            train_ds,
            batch_size=args.batch_size,
            seed=args.seed,
            epoch_index=epoch_index,
        )
        model.train()

        for batch_index, batch in enumerate(train_loader):
            if epoch_index == start_epoch and batch_index < start_batch_index:
                continue

            global_step += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError(f"Invalid loss at step {global_step}: {output.loss}")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            should_eval = global_step % args.eval_every == 0 or global_step == total_steps
            if should_eval:
                metrics = evaluate(model, val_loader, device)
                entry = {
                    "epoch": epoch_index + 1,
                    "step": global_step,
                    "train_loss": float(output.loss.detach().cpu()),
                    "train_token_loss": (
                        float(output.token_loss.detach().cpu())
                        if output.token_loss is not None
                        else None
                    ),
                    "train_sequence_loss": (
                        float(output.sequence_loss.detach().cpu())
                        if output.sequence_loss is not None
                        else None
                    ),
                    **metrics,
                }
                history.append(entry)
                print(json.dumps(entry, ensure_ascii=False))

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
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "train_records": len(train_ds),
                    "validation_records": len(val_ds),
                    "total_steps": total_steps,
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

                if metrics["selection_score"] > best_score:
                    best_score = metrics["selection_score"]
                    trainer_state["best_score"] = best_score
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
                model.train()

        start_batch_index = 0

    summary = {
        "device": str(device),
        "train_records": len(train_ds),
        "validation_records": len(val_ds),
        "steps": global_step,
        "best_selection_score": best_score,
        "runtime_seconds": time.time() - started,
        "history": history,
    }
    final_state = {
        "global_step": global_step,
        "best_score": best_score,
        "history": history,
        "next_epoch_index": args.epochs,
        "next_batch_index": 0,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_records": len(train_ds),
        "validation_records": len(val_ds),
        "total_steps": total_steps,
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
