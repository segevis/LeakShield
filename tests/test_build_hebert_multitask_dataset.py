import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from build_hebert_multitask_dataset import normalize_text, text_hash, valid_group, merge_same_text, verify_disjoint


def test_normalize_whitespace_and_unicode():
    assert normalize_text("  א\n  ב  ") == "א ב"

def test_hash_equivalent_whitespace():
    assert text_hash("א  ב") == text_hash("א\nב")

def test_bool_not_group():
    assert valid_group(True) is None

def test_merge_token_and_sequence_to_both():
    token = {"normalized_hash":"x","task":"both","sequence_label":1,"source_dataset":"t"}
    seq = {"normalized_hash":"x","task":"sequence","sequence_label":1,"source_dataset":"s"}
    rows, count = merge_same_text([token, seq])
    assert count == 1 and len(rows) == 1 and rows[0]["task"] == "both"

def test_incompatible_labels_not_merged():
    token = {"normalized_hash":"x","task":"both","sequence_label":1,"source_dataset":"t"}
    seq = {"normalized_hash":"x","task":"sequence","sequence_label":0,"source_dataset":"s"}
    rows, count = merge_same_text([token, seq])
    assert count == 0 and len(rows) == 2

def test_verify_disjoint_passes():
    train=[{"normalized_hash":"a","group_id":"g1"}]; val=[{"normalized_hash":"b","group_id":"g2"}]
    checks=verify_disjoint(train,val,{"c"},{"g3"})
    assert not any(checks.values())
