from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from train_hebert_multitask import (  # noqa: E402
    deterministic_sample,
    load_resume_state,
    save_checkpoint,
)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def save_pretrained(self, target: Path) -> None:
        torch.save(self.state_dict(), target / "fake_model.pt")


class FakeTokenizer:
    def save_pretrained(self, target: Path) -> None:
        (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_deterministic_sample_is_random_not_prefix() -> None:
    records = [{"id": index} for index in range(100)]
    sample = deterministic_sample(records, limit=10, seed=42)
    assert sample == deterministic_sample(records, limit=10, seed=42)
    assert sample != records[:10]
    assert len({row["id"] for row in sample}) == 10


def test_deterministic_sample_changes_with_seed() -> None:
    records = [{"id": index} for index in range(100)]
    assert deterministic_sample(records, 10, 1) != deterministic_sample(records, 10, 2)


def test_limit_none_keeps_all_records() -> None:
    records = [{"id": index} for index in range(5)]
    sampled = deterministic_sample(records, limit=None, seed=42)
    assert sampled == records
    assert sampled is not records


def test_checkpoint_contains_full_resume_state(tmp_path: Path) -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer_state = {
        "global_step": 12,
        "best_score": 0.8,
        "history": [{"step": 12}],
        "next_epoch_index": 0,
        "next_batch_index": 12,
        "seed": 42,
        "batch_size": 4,
        "max_length": 192,
        "train_records": 100,
        "validation_records": 20,
        "total_steps": 25,
    }

    save_checkpoint(
        model,
        tokenizer,
        optimizer,
        scheduler,
        tmp_path,
        "checkpoint-12",
        {"selection_score": 0.8},
        trainer_state,
    )

    checkpoint = tmp_path / "checkpoint-12"
    assert (checkpoint / "optimizer.pt").exists()
    assert (checkpoint / "scheduler.pt").exists()
    assert (checkpoint / "rng_state.pt").exists()
    assert (checkpoint / "trainer_state.json").exists()
    assert json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] == 12


def test_load_resume_restores_optimizer_scheduler_and_state(tmp_path: Path) -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer_state = {
        "global_step": 3,
        "best_score": 0.7,
        "history": [],
        "next_epoch_index": 0,
        "next_batch_index": 3,
        "seed": 42,
        "batch_size": 4,
        "max_length": 192,
        "train_records": 20,
        "validation_records": 10,
        "total_steps": 5,
    }
    save_checkpoint(
        model,
        tokenizer,
        optimizer,
        scheduler,
        tmp_path,
        "checkpoint-3",
        {},
        trainer_state,
    )

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    new_optimizer = torch.optim.AdamW(model.parameters(), lr=9e-3)
    new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lambda _: 1.0)
    loaded = load_resume_state(tmp_path / "checkpoint-3", new_optimizer, new_scheduler)

    assert loaded["global_step"] == 3
    assert new_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-bad"
    checkpoint.mkdir()
    optimizer = torch.optim.AdamW(FakeModel().parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    try:
        load_resume_state(checkpoint, optimizer, scheduler)
    except FileNotFoundError as exc:
        assert "full resume state" in str(exc)
    else:
        raise AssertionError("Incomplete checkpoint should be rejected")
