from __future__ import annotations

import sensitive_text_analyzer as analyzer


def test_empty_text_avoids_model_load(monkeypatch):
    monkeypatch.setattr(analyzer, "get_multitask_predictor", lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    result=analyzer.analyze_sensitive_text("   ")
    assert result["analysis_status"]=="COMPLETE"
    assert result["token_analysis"]["results"]==[]
    assert result["sequence_analysis"]["result"]["prediction"]==0


def test_public_api_uses_one_multitask_prediction(monkeypatch):
    calls={"count":0}
    class Predictor:
        def predict(self,text):
            calls["count"]+=1
            return {"token_results":[{"type":"PERSON_LEAK","value":"דנה","score":0.9}], "sequence_result":{"label":"LEAK","prediction":1,"confidence":0.95,"probability_leak":0.95,"temperature":2.8,"threshold":0.13,"source":"hebert_multitask_v2"}, "runtime":{"single_forward_pass":True,"model_path":"models/hebert_multitask_v2/best"}}
    predictor=Predictor()
    monkeypatch.setattr(analyzer,"get_multitask_predictor",lambda:predictor)
    result=analyzer.analyze_sensitive_text("דנה")
    assert calls["count"]==1
    assert result["analysis_status"]=="COMPLETE"
    assert result["runtime"]["single_forward_pass"] is True
    assert result["token_analysis"]["source"]=="hebert_multitask_v2"
    assert result["sequence_analysis"]["source"]=="hebert_multitask_v2"


def test_shared_predictor_failure_fails_both_heads(monkeypatch):
    class Predictor:
        def predict(self,text): raise RuntimeError("model unavailable")
    monkeypatch.setattr(analyzer,"get_multitask_predictor",lambda:Predictor())
    result=analyzer.analyze_sensitive_text("בדיקה")
    assert result["analysis_status"]=="FAILED"
    assert result["token_analysis"]["status"]=="ERROR"
    assert result["sequence_analysis"]["status"]=="ERROR"
    assert len(result["model_errors"])==1
