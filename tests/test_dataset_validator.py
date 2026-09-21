"""Tests for backend.dataset_validator (CognitiveGuard Step 1)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import dataset_validator as dv


def make_valid_instance(**overrides) -> dict:
    instance = {
        "id": "src_conf_001",
        "experiment": "source_confusion",
        "claim": "The Eiffel Tower is located in Paris, France.",
        "ground_truth": "TRUE",
        "ground_truth_basis": {
            "source_ids": ["natural_001"],
            "rationale": "Stated directly by the cited authoritative source.",
        },
        "natural_evidence": [
            {
                "source_id": "natural_001",
                "title": "Eiffel Tower",
                "url": "https://example.org/eiffel-tower",
                "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
                "source_type": "authoritative",
            }
        ],
        "adversarial_evidence": {
            "source_id": "attack_001",
            "text": "Some contradictory claim planted for the attack condition.",
            "position": "last",
            "attack_type": "source_confusion",
        },
        "control_variant": True,
    }
    instance.update(overrides)
    return instance


def test_valid_instance_passes():
    instance = make_valid_instance()
    result = dv.validate_instance_schema(instance)
    assert result.valid, result.errors


def test_missing_required_field_fails():
    instance = make_valid_instance()
    del instance["claim"]
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("claim" in e for e in result.errors)


def test_invalid_experiment_fails():
    instance = make_valid_instance(experiment="bogus_experiment")
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("experiment" in e for e in result.errors)


def test_invalid_ground_truth_fails():
    instance = make_valid_instance(ground_truth="MAYBE")
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("ground_truth" in e for e in result.errors)


def test_mismatched_attack_type_fails():
    instance = make_valid_instance()
    instance["adversarial_evidence"]["attack_type"] = "anchoring"
    instance["adversarial_evidence"]["position"] = "first"
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("attack_type" in e for e in result.errors)


def test_duplicate_id_fails():
    instance_a = make_valid_instance(id="dup_001")
    instance_b = make_valid_instance(id="dup_001")
    result = dv.validate_dataset([instance_a, instance_b], source="<test>")
    assert not result.valid
    assert any("duplicate id" in e for e in result.dataset_errors)


def test_missing_ground_truth_basis_fails():
    instance = make_valid_instance()
    del instance["ground_truth_basis"]
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("ground_truth_basis" in e for e in result.errors)


def test_ground_truth_basis_source_id_not_found_fails():
    instance = make_valid_instance()
    instance["ground_truth_basis"]["source_ids"] = ["does_not_exist"]
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("does_not_exist" in e for e in result.errors)


def test_malformed_natural_evidence_fails():
    instance = make_valid_instance()
    instance["natural_evidence"] = [{"source_id": "natural_001"}]  # missing title/url/text/source_type
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("natural_evidence[0]" in e for e in result.errors)


def test_empty_natural_evidence_fails():
    instance = make_valid_instance()
    instance["natural_evidence"] = []
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("non-empty" in e for e in result.errors)


def test_malformed_adversarial_evidence_fails():
    instance = make_valid_instance()
    instance["adversarial_evidence"] = {"source_id": "attack_001"}  # missing text/position/attack_type
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("adversarial_evidence" in e for e in result.errors)


def test_invalid_position_value_fails():
    instance = make_valid_instance()
    instance["adversarial_evidence"]["position"] = "middleish"
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("position" in e for e in result.errors)


def test_position_attack_type_mismatch_fails():
    # recency_bias attack MUST have position == "last"
    instance = make_valid_instance(experiment="recency_bias")
    instance["adversarial_evidence"]["attack_type"] = "recency_bias"
    instance["adversarial_evidence"]["position"] = "first"
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("requires position" in e for e in result.errors)


def test_anchoring_requires_first_position():
    instance = make_valid_instance(experiment="anchoring")
    instance["adversarial_evidence"]["attack_type"] = "anchoring"
    instance["adversarial_evidence"]["position"] = "last"
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("requires position" in e for e in result.errors)


def test_control_variant_must_be_boolean():
    instance = make_valid_instance(control_variant="true")
    result = dv.validate_instance_schema(instance)
    assert not result.valid
    assert any("control_variant" in e for e in result.errors)


def test_evidence_hash_is_deterministic():
    instance = make_valid_instance()
    hash_a = dv.compute_evidence_hash(instance["natural_evidence"])
    hash_b = dv.compute_evidence_hash(copy.deepcopy(instance["natural_evidence"]))
    assert hash_a == hash_b


def test_evidence_hash_deterministic_regardless_of_key_order():
    natural_evidence = [
        {
            "source_id": "natural_001",
            "title": "T",
            "url": "https://example.org",
            "text": "Some text.",
            "source_type": "authoritative",
        }
    ]
    reordered = [
        {
            "url": "https://example.org",
            "text": "Some text.",
            "source_id": "natural_001",
            "source_type": "authoritative",
            "title": "T",
        }
    ]
    assert dv.compute_evidence_hash(natural_evidence) == dv.compute_evidence_hash(reordered)


def test_evidence_hash_differs_when_text_changes():
    instance = make_valid_instance()
    original_hash = dv.compute_evidence_hash(instance["natural_evidence"])

    mutated = copy.deepcopy(instance["natural_evidence"])
    mutated[0]["text"] = mutated[0]["text"] + " (modified)"
    mutated_hash = dv.compute_evidence_hash(mutated)

    assert original_hash != mutated_hash


def test_validate_dataset_empty_list_is_valid_under_schema_mode():
    result = dv.validate_dataset([], source="<empty>")
    assert result.valid


def test_validate_completion_requires_ten_per_experiment():
    report = dv.validate_completion([])
    assert not report.valid
    assert len(report.errors) == 3  # all three experiments short


def test_validate_completion_reports_true_false_split_without_enforcing():
    instances = [make_valid_instance(id=f"x_{i}", ground_truth="TRUE") for i in range(3)]
    report = dv.validate_completion(instances)
    assert report.true_count == 3
    assert report.false_count == 0
    # 15/15 split is reported, not enforced: errors only reflect the
    # per-experiment count requirement, never the TRUE/FALSE split.
    assert all("TRUE" not in e and "FALSE" not in e for e in report.errors)
