from __future__ import annotations

from decision_engine import make_final_decision_v2


def token(spans=False, status="OK", error=None):
    return {"status": status, "results": ([{"type":"PASSWORD_CREDENTIAL_LEAK","value":"secret","score":0.9}] if spans else []), "source":"hebert_multitask_v2", "error":error}


def sequence(leak=False, status="OK", error=None):
    probability = 0.9 if leak else 0.05
    return {"status":status, "result": (None if status != "OK" else {"label":"LEAK" if leak else "NON_LEAK", "prediction":1 if leak else 0, "confidence":0.9 if leak else 0.95, "probability_leak":probability, "temperature":2.8146, "threshold":0.13}), "source":"hebert_multitask_v2", "error":error}


def decide(regex=None, tok=None, seq=None, status="COMPLETE"):
    return make_final_decision_v2(regex or [], tok or token(), seq or sequence(), status)


def test_regex_only_establishes_leak():
    result=decide(regex=[{"type":"EMAIL","value":"a@example.com"}])
    assert result["final_label"]=="LEAK"
    assert result["confidence_source"]=="regex_deterministic"


def test_sequence_only_establishes_leak():
    result=decide(seq=sequence(True))
    assert result["final_label"]=="LEAK"
    assert result["confidence_source"]=="hebert_sequence"


def test_regex_and_sequence_are_high_risk():
    result=decide(regex=[{"type":"EMAIL","value":"a@example.com"}], seq=sequence(True))
    assert result["final_label"]=="LEAK"
    assert result["risk_level"]=="HIGH"


def test_token_only_does_not_establish_leak():
    result=decide(tok=token(True), seq=sequence(False))
    assert result["final_label"]=="NON_LEAK"
    assert result["token_support"]["used_for_final_label"] is False
    assert result["token_support"]["used_for_localization"] is True


def test_no_detection_is_non_leak():
    result=decide()
    assert result["final_label"]=="NON_LEAK"
    assert result["risk_level"]=="LOW"


def test_sequence_failure_without_regex_is_undetermined():
    result=decide(seq=sequence(status="ERROR", error="failed"), status="FAILED")
    assert result["final_label"]=="UNDETERMINED"
    assert result["analysis_status"]=="FAILED"


def test_regex_survives_model_failure():
    result=decide(regex=[{"type":"ID","value":"123456789"}], tok=token(status="ERROR",error="failed"), seq=sequence(status="ERROR",error="failed"), status="PARTIAL")
    assert result["final_label"]=="LEAK"
    assert result["analysis_status"]=="PARTIAL"


def test_public_decision_has_no_svm_fields():
    result=decide(seq=sequence(True), tok=token(True))
    serialized=str(result).lower()
    assert "svm" not in serialized
    assert "ml_result" not in serialized
