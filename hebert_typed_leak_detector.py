"""
Inference wrapper for the fine-tuned HeBERT typed leakage detector.

This file loads the fine-tuned HeBERT token-classification model and returns
clean typed sensitive spans from the original text.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from transformers import pipeline


try:
    from config import HEBERT_TYPED_MODEL_PATH
except ImportError:
    HEBERT_TYPED_MODEL_PATH = "models/hebert_typed_leak_detector"


DEFAULT_MODEL_PATH = os.getenv("HEBERT_TYPED_MODEL_PATH", HEBERT_TYPED_MODEL_PATH)

_typed_ner = None


def _get_pipeline(model_path: str = DEFAULT_MODEL_PATH):
    global _typed_ner

    if _typed_ner is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fine-tuned HeBERT model was not found at '{model_path}'. "
                "Train it first or set HEBERT_TYPED_MODEL_PATH."
            )

        _typed_ner = pipeline(
            "token-classification",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="none",
        )

    return _typed_ner


def _normalize_label(label: str) -> str:
    label = str(label)

    if label.startswith("LABEL_"):
        return label

    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]

    return label


def _strip_bio_prefix(label: str) -> str:
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]

    return label


def _get_raw_label(entity: Dict[str, Any]) -> str:
    return str(entity.get("entity", entity.get("entity_group", "")))


def _is_entity_label(raw_label: str) -> bool:
    clean = _strip_bio_prefix(raw_label)
    return clean != "O" and not clean.endswith("O")


def _merge_entities(text: str, entities: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """
    Merge nearby token pieces of the same predicted label into clean spans
    using original start/end offsets.

    This fixes outputs like:
    ap + i + _key -> api_key
    con + ne + ction string -> connection string
    """
    spans: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for ent in entities:
        raw_label = _get_raw_label(ent)
        score = float(ent.get("score", 0.0))

        start = ent.get("start")
        end = ent.get("end")

        if start is None or end is None:
            continue

        start = int(start)
        end = int(end)

        if score < threshold or not _is_entity_label(raw_label):
            if current is not None:
                spans.append(current)
                current = None
            continue

        clean_type = _strip_bio_prefix(raw_label)

        should_merge = (
            current is not None
            and current["type"] == clean_type
            and start <= current["end"] + 2
        )

        if should_merge:
            current["end"] = max(current["end"], end)
            current["scores"].append(score)
        else:
            if current is not None:
                spans.append(current)

            current = {
                "type": clean_type,
                "start": start,
                "end": end,
                "scores": [score],
            }

    if current is not None:
        spans.append(current)

    clean_spans: List[Dict[str, Any]] = []

    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        value = text[start:end].strip()

        if not value:
            continue

        clean_spans.append(
            {
                "type": span["type"],
                "value": value,
                "score": round(sum(span["scores"]) / len(span["scores"]), 4),
                "start": start,
                "end": end,
                "source": "fine_tuned_hebert",
            }
        )

    return clean_spans


def detect_hebert_typed_leaks(
    text: str,
    threshold: float = 0.55,
    model_path: str = DEFAULT_MODEL_PATH,
) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []

    ner = _get_pipeline(model_path)
    entities = ner(text)

    return _merge_entities(text=text, entities=entities, threshold=threshold)