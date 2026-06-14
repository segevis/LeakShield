from __future__ import annotations


def test_analyzer_runs_one_shared_prediction(monkeypatch):
    import sensitive_text_analyzer as analyzer

    calls = {"count": 0}

    class FakePredictor:
        def predict(self, text):
            calls["count"] += 1
            return {
                "token_results": [],
                "sequence_result": {
                    "label": "LEAK",
                    "prediction": 1,
                    "confidence": 0.95,
                    "probability_leak": 0.95,
                    "temperature": 2.8146,
                    "threshold": 0.13,
                    "source": "hebert_multitask_v2",
                },
                "runtime": {"single_forward_pass": True},
            }

    monkeypatch.setattr(analyzer, "get_multitask_predictor", lambda: FakePredictor())
    result = analyzer.analyze_sensitive_text("בדיקה")
    assert calls["count"] == 1
    assert result["sequence_analysis"]["result"]["prediction"] == 1
    assert result["runtime"]["single_forward_pass"] is True


def test_sequence_leak_without_span_is_leak():
    from decision_engine import make_final_decision_v2

    result = make_final_decision_v2(
        [],
        {"status": "OK", "results": [], "source": "model", "error": None},
        {
            "status": "OK",
            "result": {
                "label": "LEAK",
                "prediction": 1,
                "confidence": 0.9,
                "probability_leak": 0.9,
                "threshold": 0.13,
            },
            "source": "model",
            "error": None,
        },
        "COMPLETE",
    )
    assert result["final_label"] == "LEAK"


def test_token_alone_does_not_establish_leak():
    from decision_engine import make_final_decision_v2

    result = make_final_decision_v2(
        [],
        {
            "status": "OK",
            "results": [{"type": "PERSON_LEAK", "value": "דנה", "score": 0.99}],
            "source": "model",
            "error": None,
        },
        {
            "status": "OK",
            "result": {
                "label": "NON_LEAK",
                "prediction": 0,
                "confidence": 0.9,
                "probability_leak": 0.1,
                "threshold": 0.13,
            },
            "source": "model",
            "error": None,
        },
        "COMPLETE",
    )
    assert result["final_label"] == "NON_LEAK"
    assert result["token_support"]["used_for_final_label"] is False


def test_regex_can_establish_leak():
    from decision_engine import make_final_decision_v2

    result = make_final_decision_v2(
        [{"type": "EMAIL", "value": "a@example.com"}],
        {"status": "OK", "results": [], "source": "model", "error": None},
        {
            "status": "OK",
            "result": {
                "label": "NON_LEAK",
                "prediction": 0,
                "confidence": 0.99,
                "probability_leak": 0.01,
                "threshold": 0.13,
            },
            "source": "model",
            "error": None,
        },
        "COMPLETE",
    )
    assert result["final_label"] == "LEAK"
    assert result["confidence_source"] == "regex_deterministic"
