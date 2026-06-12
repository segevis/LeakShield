from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from training.hebert_multitask_model import HeBERTMultiTaskModel


DEFAULT_PRODUCTION_CONFIG_PATH = "config/hebert_multitask_v2_production.json"
DEFAULT_MODEL_PATH = "models/hebert_multitask_v2/best"
DEFAULT_TOKEN_THRESHOLD = 0.55
SOURCE = "hebert_multitask_v2"

_PREDICTOR = None
_PREDICTOR_LOCK = threading.Lock()


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _split_bio(label: str) -> tuple[str, str]:
    label = str(label)
    if label.startswith(("B-", "I-")):
        return label[0], label[2:]
    return "", label


class HeBERTMultiTaskPredictor:
    """One shared HeBERT encoder, one forward pass, two outputs."""

    def __init__(
        self,
        model_path: str,
        production_config_path: str,
        token_threshold: float = DEFAULT_TOKEN_THRESHOLD,
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.production_config_path = production_config_path
        self.production_config = _load_json(production_config_path)

        calibration = self.production_config.get("sequence_calibration")
        if not isinstance(calibration, dict):
            raise ValueError("Missing sequence_calibration in production config")

        self.temperature = _finite_float(
            calibration.get("temperature"), "temperature"
        )
        self.sequence_threshold = _finite_float(
            calibration.get("threshold"), "sequence threshold"
        )
        self.token_threshold = _finite_float(
            token_threshold, "token threshold"
        )

        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if not 0.0 < self.sequence_threshold < 1.0:
            raise ValueError("Sequence threshold must be between 0 and 1")
        if not 0.0 <= self.token_threshold <= 1.0:
            raise ValueError("Token threshold must be between 0 and 1")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"HeBERT Multi-Task model was not found at '{model_path}'"
            )

        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required")

        self.model = HeBERTMultiTaskModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        raw_labels = getattr(self.model.config, "id2tokenlabel", None)
        if not isinstance(raw_labels, dict):
            raw_labels = getattr(self.model.config, "id2label", {})
        self.id2tokenlabel = {
            int(key): str(value) for key, value in raw_labels.items()
        }
        self.max_length = int(
            self.production_config.get("architecture", {}).get(
                "max_length", 192
            )
        )

    def _merge_tokens(
        self,
        text: str,
        offsets: list[list[int]],
        label_ids: list[int],
        scores: list[float],
        special_mask: list[int],
    ) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        def flush() -> None:
            nonlocal current
            if current is None:
                return
            start, end = int(current["start"]), int(current["end"])
            value = text[start:end].strip()
            if value:
                spans.append(
                    {
                        "type": current["type"],
                        "value": value,
                        "score": round(
                            sum(current["scores"]) / len(current["scores"]), 4
                        ),
                        "start": start,
                        "end": end,
                        "source": SOURCE,
                    }
                )
            current = None

        for offset, label_id, score, special in zip(
            offsets, label_ids, scores, special_mask
        ):
            start, end = int(offset[0]), int(offset[1])
            raw_label = self.id2tokenlabel.get(int(label_id), "O")
            prefix, entity_type = _split_bio(raw_label)
            valid = (
                not special
                and end > start
                and raw_label != "O"
                and entity_type != "O"
                and float(score) >= self.token_threshold
            )
            if not valid:
                flush()
                continue

            gap_text = (
                text[int(current["end"]):start]
                 if current is not None
                else ""
            )

            merge = (
              current is not None
              and current["type"] == entity_type
              and start <= int(current["end"]) + 1
              and not gap_text.strip()
            )

            if merge:
                current["end"] = max(int(current["end"]), end)
                current["scores"].append(float(score))
            else:
                flush()
                current = {
                    "type": entity_type,
                    "start": start,
                    "end": end,
                    "scores": [float(score)],
                }

        flush()
        return spans

    @torch.inference_mode()
    def predict(self, text: str) -> dict[str, Any]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        special_mask = encoded.pop("special_tokens_mask")[0].tolist()
        inputs = {key: value.to(self.device) for key, value in encoded.items()}

        output = self.model(**inputs)

        token_probs = F.softmax(output.token_logits[0], dim=-1)
        token_scores, token_ids = token_probs.max(dim=-1)
        token_results = self._merge_tokens(
            text,
            offsets,
            token_ids.cpu().tolist(),
            token_scores.cpu().tolist(),
            special_mask,
        )

        sequence_probs = F.softmax(
            output.sequence_logits[0] / self.temperature, dim=-1
        )
        probability_leak = float(sequence_probs[1].cpu())
        prediction = int(probability_leak >= self.sequence_threshold)
        confidence = (
            probability_leak if prediction else 1.0 - probability_leak
        )

        return {
            "token_results": token_results,
            "sequence_result": {
                "label": "LEAK" if prediction else "NON_LEAK",
                "prediction": prediction,
                "confidence": round(confidence, 6),
                "probability_leak": round(probability_leak, 6),
                "temperature": self.temperature,
                "threshold": self.sequence_threshold,
                "source": SOURCE,
            },
            "runtime": {
                "source": SOURCE,
                "model_path": self.model_path,
                "device": str(self.device),
                "single_forward_pass": True,
            },
        }


def get_multitask_predictor(
    model_path: str | None = None,
    production_config_path: str | None = None,
    token_threshold: float | None = None,
) -> HeBERTMultiTaskPredictor:
    global _PREDICTOR
    with _PREDICTOR_LOCK:
        if _PREDICTOR is None:
            _PREDICTOR = HeBERTMultiTaskPredictor(
                model_path=model_path or os.getenv(
                    "HEBERT_MULTITASK_MODEL_PATH", DEFAULT_MODEL_PATH
                ),
                production_config_path=production_config_path or os.getenv(
                    "HEBERT_MULTITASK_CONFIG_PATH",
                    DEFAULT_PRODUCTION_CONFIG_PATH,
                ),
                token_threshold=(
                    token_threshold
                    if token_threshold is not None
                    else float(
                        os.getenv(
                            "HEBERT_MULTITASK_TOKEN_THRESHOLD",
                            str(DEFAULT_TOKEN_THRESHOLD),
                        )
                    )
                ),
            )
    return _PREDICTOR


def reset_multitask_predictor_cache() -> None:
    global _PREDICTOR
    with _PREDICTOR_LOCK:
        _PREDICTOR = None
