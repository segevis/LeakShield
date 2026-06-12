from __future__ import annotations

import ast
from pathlib import Path

import analysis_pipeline
import config
from decision_engine import make_final_decision_v2

ROOT = Path(__file__).parents[1]
RUNTIME_FILES = [
    "main.py", "gui.py", "analysis_pipeline.py", "decision_engine.py",
    "detectors.py", "sensitive_text_analyzer.py", "hebert_multitask_detector.py",
    "pdf_processor.py", "pdf_colored_report.py", "config.py", "utils.py",
]
FORBIDDEN_RUNTIME_TOKENS = (
    "ml_classifier_svm_final", "predict_svm_final", "SVM_MODEL_PATH",
    "SVM_VECTORIZER_PATH", "ACTIVE_SVM_MODEL_NAME", "models/svm_final",
    "hebert_typed_leak_detector", "hebert_sequence_classifier",
)


def test_runtime_files_do_not_reference_svm_or_old_hebert_wrappers():
    violations = []
    for relative in RUNTIME_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8-sig")
        for token in FORBIDDEN_RUNTIME_TOKENS:
            if token in text:
                violations.append((relative, token))
    assert violations == []


def test_runtime_python_imports_do_not_import_legacy_detectors():
    forbidden_modules = {"ml_classifier_svm_final", "hebert_typed_leak_detector", "hebert_sequence_classifier"}
    violations = []
    for relative in RUNTIME_FILES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            overlap = names & forbidden_modules
            if overlap:
                violations.append((relative, sorted(overlap)))
    assert violations == []


def test_config_exposes_only_production_hebert_detection_model():
    assert config.HEBERT_MULTITASK_MODEL_PATH == "models/hebert_multitask_v2/best"
    assert config.ACTIVE_DETECTION_LAYERS == ("regex", "hebert_multitask_v2")
    assert not hasattr(config, "SVM_MODEL_PATH")
    assert not hasattr(config, "SVM_VECTORIZER_PATH")
    assert not hasattr(config, "ACTIVE_SVM_MODEL_NAME")


def test_analysis_pipeline_has_no_svm_callable():
    assert not hasattr(analysis_pipeline, "predict_svm_final")
    assert not hasattr(analysis_pipeline, "classify_text")


def test_token_only_is_localization_not_final_leak():
    result = make_final_decision_v2(
        regex_results=[],
        token_analysis={"status": "OK", "results": [{"type": "PERSON_LEAK", "value": "דנה", "score": 0.99}], "source": "hebert_multitask_v2", "error": None},
        sequence_analysis={"status": "OK", "result": {"label": "NON_LEAK", "prediction": 0, "confidence": 0.9, "probability_leak": 0.1, "temperature": 2.8, "threshold": 0.13}, "source": "hebert_multitask_v2", "error": None},
        analysis_status="COMPLETE",
    )
    assert result["final_label"] == "NON_LEAK"
    assert result["token_support"]["used_for_final_label"] is False
    assert result["token_support"]["used_for_localization"] is True


def test_regex_or_sequence_can_establish_leak():
    regex_decision = make_final_decision_v2(
        [{"type": "EMAIL", "value": "a@example.com"}],
        {"status": "OK", "results": [], "source": "hebert_multitask_v2", "error": None},
        {"status": "OK", "result": {"label": "NON_LEAK", "prediction": 0, "confidence": 0.99, "probability_leak": 0.01, "temperature": 2.8, "threshold": 0.13}, "source": "hebert_multitask_v2", "error": None},
        "COMPLETE",
    )
    sequence_decision = make_final_decision_v2(
        [],
        {"status": "OK", "results": [], "source": "hebert_multitask_v2", "error": None},
        {"status": "OK", "result": {"label": "LEAK", "prediction": 1, "confidence": 0.95, "probability_leak": 0.95, "temperature": 2.8, "threshold": 0.13}, "source": "hebert_multitask_v2", "error": None},
        "COMPLETE",
    )
    assert regex_decision["final_label"] == "LEAK"
    assert sequence_decision["final_label"] == "LEAK"
