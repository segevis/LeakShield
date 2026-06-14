from __future__ import annotations
import sys
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "training" / "evaluate_and_calibrate_hebert_multitask.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("evaluate_and_calibrate_hebert_multitask", MODULE_PATH)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_normalize_binary_label(module):
    assert module.normalize_binary_label("LEAK") == 1
    assert module.normalize_binary_label("NON_LEAK") == 0
    assert module.normalize_binary_label("1") == 1
    assert module.normalize_binary_label(0) == 0


def test_bool_is_not_binary_label(module):
    with pytest.raises(ValueError):
        module.normalize_binary_label(True)


def test_sequence_rows_jsonl(module, tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"text": "סוד", "label": 1, "category": "X"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = module.sequence_rows(path)
    assert rows[0]["label"] == 1
    assert rows[0]["label_name"] == "LEAK"
    assert rows[0]["text"] == "סוד"


def test_sequence_rows_csv(module, tmp_path):
    path = tmp_path / "rows.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "category"])
        writer.writeheader()
        writer.writerow({"text": "תקין", "label": "0", "category": "SAFE"})
    rows = module.sequence_rows(path)
    assert rows[0]["label"] == 0


def test_token_rows(module, tmp_path):
    path = tmp_path / "token.jsonl"
    path.write_text(
        json.dumps(
            {"text": "יעל", "tokens": ["יעל"], "ner_tags": [1], "document_label": 1},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    rows = module.token_rows(path)
    assert rows[0]["tokens"] == ["יעל"]
    assert rows[0]["ner_tags"] == [1]
    assert rows[0]["document_label"] == 1


def test_binary_metrics(module):
    result = module.binary_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.7, 0.8, 0.9])
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["tn"] == 1
    assert result["recall"] == 1.0


def test_binary_metrics_empty(module):
    result = module.binary_metrics([], [])
    assert result["samples"] == 0
    assert result["f1"] == 0.0


def test_temperature_is_positive(module):
    logits = np.asarray([[4.0, -4.0], [-3.0, 3.0], [2.0, -2.0], [-2.0, 2.0]])
    labels = np.asarray([0, 1, 0, 1])
    temperature = module.fit_temperature(logits, labels)
    assert 0.05 <= temperature <= 20.0


def test_probabilities_from_logits(module):
    logits = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    probabilities = module.probabilities_from_logits(logits, 1.0)
    assert probabilities[0] < 0.5
    assert probabilities[1] > 0.5


def test_invalid_temperature_rejected(module):
    with pytest.raises(ValueError):
        module.probabilities_from_logits(np.asarray([[1.0, 0.0]]), 0.0)


def test_threshold_respects_constraints_when_possible(module):
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold, rows, rule = module.choose_threshold(labels, probabilities, 0.8, 0.25)
    selected = next(row for row in rows if row["threshold"] == threshold)
    assert selected["recall"] >= 0.8
    assert selected["fpr"] <= 0.25
    assert rule == "STRICT_RECALL_AND_FPR"


def test_token_micro_metrics(module):
    result = module.token_micro_metrics([0, 1, 2, 0], [0, 1, 0, 3])
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1


def test_entity_type(module):
    assert module.entity_type("B-PERSON_LEAK") == "PERSON_LEAK"
    assert module.entity_type("I-PERSON_LEAK") == "PERSON_LEAK"
    assert module.entity_type("O") == "O"


def test_token_per_type_metrics(module):
    rows = module.token_per_type_metrics(
        [0, 1, 2, 3],
        [0, 1, 0, 3],
        {0: "O", 1: "B-PERSON", 2: "I-PERSON", 3: "B-CONTACT"},
    )
    by_name = {row["entity_type"]: row for row in rows}
    assert by_name["PERSON"]["tp"] == 1
    assert by_name["PERSON"]["fn"] == 1
    assert by_name["CONTACT"]["tp"] == 1


def test_calibration_config(module, tmp_path):
    logits = np.asarray([[3.0, -3.0], [-3.0, 3.0], [2.0, -2.0], [-2.0, 2.0]])
    labels = np.asarray([0, 1, 0, 1])
    config, sweep, probabilities = module.calibration_config(
        "best", tmp_path / "cal.jsonl", logits, labels, 0.8, 0.25
    )
    assert config["model"] == "best"
    assert 0.01 <= config["threshold"] <= 0.99
    assert config["temperature"] > 0
    assert len(sweep) == 99
    assert len(probabilities) == 4


def test_parse_named_path(module):
    name, path = module.parse_named_path("best=models/x")
    assert name == "best"
    assert str(path).replace("\\", "/") == "models/x"


def test_write_json_preserves_hebrew(module, tmp_path):
    path = tmp_path / "value.json"
    module.write_json(path, {"text": "עברית"})
    raw = path.read_text(encoding="utf-8")
    assert "עברית" in raw
    assert "\\u" not in raw


def test_write_csv_creates_headers(module, tmp_path):
    path = tmp_path / "rows.csv"
    module.write_csv(path, [{"a": 1, "b": "עברית"}])
    raw = path.read_text(encoding="utf-8-sig")
    assert "a,b" in raw
    assert "עברית" in raw


def test_safe_name(module):
    assert module.safe_name("best model/1") == "best_model_1"
