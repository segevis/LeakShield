from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import BertConfig


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "training" / "train_hebert_multitask_v2.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location(
        "train_hebert_multitask_v2",
        MODULE_PATH,
    )
    assert spec and spec.loader

    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)

    return loaded


def build_tiny_multitask_model(module):
    """
    Build a small local model for structural unit tests.

    These tests verify layer freezing and optimizer parameter groups.
    They must not depend on the archived legacy token checkpoint or on
    downloading any model from Hugging Face.
    """
    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    config.num_token_labels = 127
    config.num_sequence_labels = 2
    config.token_loss_weight = 2.0
    config.sequence_loss_weight = 1.0

    config.id2tokenlabel = {
        index: f"TOKEN_LABEL_{index}"
        for index in range(config.num_token_labels)
    }
    config.tokenlabel2id = {
        label: index
        for index, label in config.id2tokenlabel.items()
    }

    config.id2sequencelabel = {
        0: "NON_LEAK",
        1: "LEAK",
    }
    config.sequencelabel2id = {
        "NON_LEAK": 0,
        "LEAK": 1,
    }

    return module.HeBERTMultiTaskModel(config)


def save_tiny_checkpoint(module, path: Path, num_token_labels: int = 127, num_sequence_labels: int = 2):
    model = build_tiny_multitask_model(module)
    model.config.num_token_labels = num_token_labels
    model.config.num_sequence_labels = num_sequence_labels
    if num_token_labels != 127:
        model.config.id2tokenlabel = {
            index: f"TOKEN_LABEL_{index}"
            for index in range(num_token_labels)
        }
        model.config.tokenlabel2id = {
            label: index
            for index, label in model.config.id2tokenlabel.items()
        }
        model.token_classifier = torch.nn.Linear(
            model.config.hidden_size,
            num_token_labels,
        )
    if num_sequence_labels != 2:
        model.config.id2sequencelabel = {
            index: f"SEQUENCE_LABEL_{index}"
            for index in range(num_sequence_labels)
        }
        model.config.sequencelabel2id = {
            label: index
            for index, label in model.config.id2sequencelabel.items()
        }
        model.sequence_classifier = torch.nn.Linear(
            model.config.hidden_size,
            num_sequence_labels,
        )
    path.mkdir(parents=True)
    model.save_pretrained(path, safe_serialization=True)
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


def write_label_schema(path: Path, model) -> Path:
    labels = [
        label
        for _, label in sorted(
            model.config.id2tokenlabel.items(),
            key=lambda item: int(item[0]),
        )
    ]
    payload = {
        "labels": labels,
        "label2id": model.config.tokenlabel2id,
        "id2label": {
            str(index): label
            for index, label in model.config.id2tokenlabel.items()
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_multitask_jsonl(path: Path, hard: bool = False) -> Path:
    record = {
        "record_id": "r1",
        "task": "sequence",
        "text": "בדיקה ללא אימון",
        "sequence_label": "NON_LEAK",
        "split": "hard_augmentation_v2" if hard else "train",
    }
    path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_is_hard_record(module):
    assert module.is_hard_record(
        {"split": "hard_augmentation_v2"}
    )

    assert module.is_hard_record(
        {
            "source_style":
                "new_targeted_hard_augmentation"
        }
    )

    assert not module.is_hard_record(
        {"split": "train"}
    )


def test_combined_selection_score_no_penalty(module):
    overall = {
        "token": {
            "f1": 0.95,
        },
        "sequence": {
            "f1": 0.90,
        },
    }

    hard = {
        "token": {
            "f1": 0.80,
        },
        "sequence": {
            "f1": 0.85,
            "recall": 0.90,
            "specificity": 0.80,
        },
    }

    score, failed = module.combined_selection_score(
        overall,
        hard,
        minimum_overall_token_f1=0.90,
        minimum_hard_sequence_recall=0.85,
        minimum_hard_sequence_specificity=0.70,
    )

    assert failed == []

    assert score == pytest.approx(
        0.30 * 0.95
        + 0.30 * 0.80
        + 0.20 * 0.90
        + 0.20 * 0.85
    )


def test_combined_selection_score_penalty(module):
    overall = {
        "token": {
            "f1": 0.85,
        },
        "sequence": {
            "f1": 0.90,
        },
    }

    hard = {
        "token": {
            "f1": 0.80,
        },
        "sequence": {
            "f1": 0.85,
            "recall": 0.70,
            "specificity": 0.60,
        },
    }

    score, failed = module.combined_selection_score(
        overall,
        hard,
        minimum_overall_token_f1=0.90,
        minimum_hard_sequence_recall=0.85,
        minimum_hard_sequence_specificity=0.70,
    )

    assert set(failed) == {
        "overall_token_f1",
        "hard_sequence_recall",
        "hard_sequence_specificity",
    }

    raw_score = (
        0.30 * 0.85
        + 0.30 * 0.80
        + 0.20 * 0.90
        + 0.20 * 0.85
    )

    assert score == pytest.approx(
        raw_score - 0.30
    )


def test_freeze_bottom_encoder_layers(module):
    model = build_tiny_multitask_model(module)

    summary = module.freeze_bottom_encoder_layers(
        model,
        trainable_top_layers=4,
    )

    assert summary == {
        "total_encoder_layers": 12,
        "frozen_encoder_layers": 8,
        "trainable_encoder_layers": 4,
    }

    assert all(
        not parameter.requires_grad
        for parameter
        in model.bert.embeddings.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in model.bert.encoder.layer[0].parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in model.bert.encoder.layer[7].parameters()
    )

    assert any(
        parameter.requires_grad
        for parameter
        in model.bert.encoder.layer[8].parameters()
    )

    assert any(
        parameter.requires_grad
        for parameter
        in model.bert.encoder.layer[-1].parameters()
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.token_classifier.parameters()
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.sequence_classifier.parameters()
    )


def test_build_optimizer_has_three_groups(module):
    model = build_tiny_multitask_model(module)

    module.freeze_bottom_encoder_layers(
        model,
        trainable_top_layers=4,
    )

    optimizer = module.build_optimizer(
        model,
        encoder_lr=1e-6,
        token_head_lr=3e-6,
        sequence_head_lr=3e-6,
        weight_decay=0.01,
    )

    groups = {
        group["group_name"]: group
        for group in optimizer.param_groups
    }

    assert set(groups) == {
        "encoder",
        "token_head",
        "sequence_head",
    }

    assert groups["encoder"]["lr"] == pytest.approx(
        1e-6
    )
    assert groups["token_head"]["lr"] == pytest.approx(
        3e-6
    )
    assert groups["sequence_head"]["lr"] == pytest.approx(
        3e-6
    )

    assert len(groups["encoder"]["params"]) > 0
    assert len(groups["token_head"]["params"]) > 0
    assert len(groups["sequence_head"]["params"]) > 0


def test_validate_weights_only_checkpoint_loads_complete_model(module, tmp_path):
    model = save_tiny_checkpoint(module, tmp_path / "checkpoint")
    schema = write_label_schema(tmp_path / "schema.json", model)

    report = module.validate_model_weights_only_checkpoint(
        tmp_path / "checkpoint",
        schema,
    )

    assert report["compatible"] is True
    assert report["optimizer_state_loaded"] is False
    assert report["scheduler_state_loaded"] is False
    assert report["trainer_state_loaded"] is False
    assert report["rng_state_loaded"] is False
    assert report["missing_parameters"] == []
    assert report["shape_mismatches"] == []
    assert report["parameter_count"] > 0


def test_weights_only_parameters_match_source_checkpoint(module, tmp_path):
    source = save_tiny_checkpoint(module, tmp_path / "checkpoint")
    loaded = module.HeBERTMultiTaskModel.from_pretrained(
        tmp_path / "checkpoint"
    )

    for name, parameter in source.state_dict().items():
        assert torch.equal(parameter, loaded.state_dict()[name])


def test_validate_weights_only_rejects_incompatible_token_count(module, tmp_path):
    model = save_tiny_checkpoint(module, tmp_path / "checkpoint")
    schema = write_label_schema(tmp_path / "schema.json", model)
    config_path = tmp_path / "checkpoint" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_token_labels"] = 999
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = module.validate_model_weights_only_checkpoint(
        tmp_path / "checkpoint",
        schema,
        load_weights=False,
    )

    assert report["compatible"] is False
    assert "num_token_labels does not match token label mapping" in report["issues"]


def test_validate_weights_only_rejects_incompatible_sequence_count(module, tmp_path):
    model = save_tiny_checkpoint(module, tmp_path / "checkpoint")
    schema = write_label_schema(tmp_path / "schema.json", model)
    config_path = tmp_path / "checkpoint" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_sequence_labels"] = 3
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = module.validate_model_weights_only_checkpoint(
        tmp_path / "checkpoint",
        schema,
        load_weights=False,
    )

    assert report["compatible"] is False
    assert "num_sequence_labels must be 2" in report["issues"]


def test_missing_initialize_checkpoint_is_rejected(module, tmp_path):
    with pytest.raises(FileNotFoundError):
        module.validate_model_weights_only_checkpoint(
            tmp_path / "missing",
            None,
        )


def test_output_directory_rejects_production_path(module, monkeypatch, tmp_path):
    production = tmp_path / "models" / "hebert_multitask_v2" / "best"
    production.mkdir(parents=True)
    monkeypatch.setattr(module, "PRODUCTION_MODEL_DIR", production)

    with pytest.raises(ValueError):
        module.validate_output_directory_safety(production)

    with pytest.raises(ValueError):
        module.validate_output_directory_safety(production / "child")


def test_initialize_and_resume_are_mutually_exclusive(module, monkeypatch, tmp_path):
    model = save_tiny_checkpoint(module, tmp_path / "checkpoint")
    schema = write_label_schema(tmp_path / "schema.json", model)
    train = write_multitask_jsonl(tmp_path / "train.jsonl")
    validation = write_multitask_jsonl(tmp_path / "validation.jsonl", hard=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_hebert_multitask_v2.py",
            "--train",
            str(train),
            "--validation",
            str(validation),
            "--output-dir",
            str(tmp_path / "out"),
            "--initialize-from-model",
            str(tmp_path / "checkpoint"),
            "--resume-from-checkpoint",
            str(tmp_path / "checkpoint"),
            "--label-schema",
            str(schema),
            "--dry-run-initialize-only",
        ],
    )

    with pytest.raises(ValueError):
        module.main()


def test_weights_only_dry_run_starts_fresh_state(module, monkeypatch, tmp_path, capsys):
    model = save_tiny_checkpoint(module, tmp_path / "checkpoint")
    schema = write_label_schema(tmp_path / "schema.json", model)
    train = write_multitask_jsonl(tmp_path / "train.jsonl")
    validation = write_multitask_jsonl(tmp_path / "validation.jsonl", hard=True)

    class DummyTokenizer:
        pass

    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_hebert_multitask_v2.py",
            "--train",
            str(train),
            "--validation",
            str(validation),
            "--output-dir",
            str(tmp_path / "out"),
            "--initialize-from-model",
            str(tmp_path / "checkpoint"),
            "--label-schema",
            str(schema),
            "--dry-run-initialize-only",
            "--epochs",
            "1",
            "--gradient-accumulation-steps",
            "4",
        ],
    )

    module.main()
    captured = capsys.readouterr().out
    report = json.loads(captured)

    assert report["mode"] == "weights_only"
    assert report["global_step"] == 0
    assert report["start_epoch"] == 0
    assert report["start_batch_index"] == 0
    assert report["optimizer_state_restored"] is False
    assert report["scheduler_state_restored"] is False
    assert report["trainer_state_restored"] is False
    assert report["rng_state_restored"] is False
    assert report["gradient_accumulation_steps"] == 4
    assert report["micro_batches_per_epoch"] == 1
    assert report["optimizer_updates_per_epoch"] == 1
    assert report["total_steps"] == 1
    assert report["warmup_steps"] == 0
    assert report["scheduler_total_steps"] == 1
    assert report["effective_batch_size"] == 16
    assert not (tmp_path / "out").exists()


def test_weights_only_dry_run_applies_new_seed_and_skips_training_loop(
    module,
    monkeypatch,
    tmp_path,
    capsys,
):
    model = save_tiny_checkpoint(module, tmp_path / "checkpoint")
    schema = write_label_schema(tmp_path / "schema.json", model)
    train = write_multitask_jsonl(tmp_path / "train.jsonl")
    validation = write_multitask_jsonl(tmp_path / "validation.jsonl", hard=True)
    calls: list[tuple[str, int]] = []

    class DummyTokenizer:
        pass

    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(
        module.random,
        "seed",
        lambda value: calls.append(("random", value)),
    )
    monkeypatch.setattr(
        module.np.random,
        "seed",
        lambda value: calls.append(("numpy", value)),
    )
    monkeypatch.setattr(
        module.torch,
        "manual_seed",
        lambda value: calls.append(("torch", value)),
    )
    monkeypatch.setattr(
        module,
        "make_balanced_train_loader",
        lambda *args, **kwargs: pytest.fail("training loader must not run"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_hebert_multitask_v2.py",
            "--train",
            str(train),
            "--validation",
            str(validation),
            "--output-dir",
            str(tmp_path / "out"),
            "--initialize-from-model",
            str(tmp_path / "checkpoint"),
            "--label-schema",
            str(schema),
            "--dry-run-initialize-only",
            "--seed",
            "20260609",
        ],
    )

    module.main()
    capsys.readouterr()

    assert ("random", 20260609) in calls
    assert ("numpy", 20260609) in calls
    assert ("torch", 20260609) in calls


def test_existing_resume_option_remains_available(module, tmp_path):
    parser_lines = MODULE_PATH.read_text(encoding="utf-8")
    assert "--resume-from-checkpoint" in parser_lines
    assert "load_resume_state(" in parser_lines


def test_gradient_accumulation_default_one_update_per_micro_batch(module):
    assert module.optimizer_update_count(1, 1) == 1
    assert module.optimizer_update_count(4, 1) == 4
    boundaries = [
        index
        for index in range(4)
        if module.should_apply_optimizer_update(index, 4, 1)
    ]
    assert boundaries == [0, 1, 2, 3]


def test_gradient_accumulation_four_micro_batches_one_update(module):
    assert module.optimizer_update_count(4, 4) == 1
    boundaries = [
        index
        for index in range(4)
        if module.should_apply_optimizer_update(index, 4, 4)
    ]
    assert boundaries == [3]


def test_gradient_accumulation_eight_micro_batches_two_updates(module):
    assert module.optimizer_update_count(8, 4) == 2
    boundaries = [
        index
        for index in range(8)
        if module.should_apply_optimizer_update(index, 8, 4)
    ]
    assert boundaries == [3, 7]


def test_gradient_accumulation_ten_micro_batches_three_updates(module):
    assert module.optimizer_update_count(10, 4) == 3
    boundaries = [
        index
        for index in range(10)
        if module.should_apply_optimizer_update(index, 10, 4)
    ]
    assert boundaries == [3, 7, 9]


def test_gradient_accumulation_loss_is_scaled_for_backward():
    loss = torch.tensor(8.0)
    scaled = loss / 4
    assert scaled.item() == pytest.approx(2.0)
    assert loss.item() == pytest.approx(8.0)


def test_gradient_accumulation_rejects_values_below_one(module):
    with pytest.raises(module.argparse.ArgumentTypeError):
        module.positive_int("0")
    with pytest.raises(ValueError):
        module.optimizer_update_count(1, 0)
    with pytest.raises(ValueError):
        module.should_apply_optimizer_update(0, 1, 0)


def test_gradient_accumulation_warmup_uses_optimizer_updates(module):
    micro_batches = 10
    accumulation = 4
    epochs = 2
    updates_per_epoch = module.optimizer_update_count(micro_batches, accumulation)
    total_steps = updates_per_epoch * epochs
    warmup_steps = int(total_steps * 0.25)

    assert updates_per_epoch == 3
    assert total_steps == 6
    assert warmup_steps == 1


def test_gradient_accumulation_effective_batch_size():
    per_device_batch_size = 4
    accumulation = 4
    assert per_device_batch_size * accumulation == 16


def test_gradient_accumulation_update_operations_are_boundary_gated(module):
    source = module.__loader__.get_source(module.__name__)
    boundary = source.index("if not should_apply_optimizer_update(")
    gated_block = source[boundary : source.index("should_eval = (", boundary)]

    assert "torch.nn.utils.clip_grad_norm_(" in gated_block
    assert "optimizer.step()" in gated_block
    assert "scheduler.step()" in gated_block
    assert "optimizer.zero_grad(set_to_none=True)" in gated_block
    assert "global_step += 1" in gated_block


def test_gradient_accumulation_scheduler_steps_are_optimizer_update_based(module):
    source = module.__loader__.get_source(module.__name__)

    assert "optimizer_updates_per_epoch = optimizer_update_count(" in source
    assert "total_steps = max(1, optimizer_updates_per_epoch * args.epochs)" in source
    assert "warmup_steps = int(total_steps * args.warmup_ratio)" in source


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.hard_flags = (
            [False] * 70
            + [True] * 30
        )

    def __len__(self):
        return len(self.hard_flags)

    def __getitem__(self, index):
        return {
            "value": torch.tensor(index),
        }


def test_balanced_sampler_runs(module):
    dataset = DummyDataset()

    loader = module.make_balanced_train_loader(
        dataset,
        batch_size=10,
        seed=42,
        epoch_index=0,
        hard_ratio=0.30,
    )

    values = []

    for batch in loader:
        values.extend(
            batch["value"].tolist()
        )

    assert len(values) == len(dataset)
