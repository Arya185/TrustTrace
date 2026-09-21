"""Dataset schema validation for CognitiveGuard.

Validates claim instances against the CognitiveGuard dataset contract
(see README.md, "Dataset Schema" and "Ground Truth Rules"). Two modes:

    --schema     structural validation only (safe on empty datasets)
    --complete   structural validation + exact-count/distribution checks
                 (requires 10 instances per experiment)

This module also defines the frozen-evidence hashing contract used to
guarantee byte-identical natural_evidence across control / attack /
attack+intervention runs (see README.md, "Evidence Freezing").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXPERIMENTS = {"source_confusion", "anchoring", "recency_bias"}
GROUND_TRUTH_VALUES = {"TRUE", "FALSE"}
POSITIONS = {"first", "middle", "last"}

# attack_type -> the only position it is allowed to appear at.
# Attack types absent from this map (source_confusion) may use any
# position in POSITIONS.
REQUIRED_POSITION_BY_ATTACK_TYPE = {
    "recency_bias": "last",
    "anchoring": "first",
}

REQUIRED_TOP_LEVEL_FIELDS = (
    "id",
    "experiment",
    "claim",
    "ground_truth",
    "ground_truth_basis",
    "natural_evidence",
    "adversarial_evidence",
    "control_variant",
)

REQUIRED_GROUND_TRUTH_BASIS_FIELDS = ("source_ids", "rationale")

REQUIRED_NATURAL_EVIDENCE_FIELDS = ("source_id", "title", "url", "text", "source_type")

REQUIRED_ADVERSARIAL_EVIDENCE_FIELDS = ("source_id", "text", "position", "attack_type")

REQUIRED_INSTANCES_PER_EXPERIMENT = 10
TARGET_TRUE_COUNT = 15
TARGET_FALSE_COUNT = 15


class DatasetValidationError(Exception):
    """Raised when a dataset fails validation and the caller wants a hard stop."""


@dataclass
class InstanceResult:
    """Validation outcome for a single claim instance."""

    instance_id: Any
    errors: list[str] = field(default_factory=list)
    natural_evidence_hash: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass
class DatasetResult:
    """Validation outcome for an entire dataset file (list of instances)."""

    source: str
    instances: list[InstanceResult] = field(default_factory=list)
    dataset_errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.dataset_errors and all(r.valid for r in self.instances)

    @property
    def errors(self) -> list[str]:
        out = list(self.dataset_errors)
        for r in self.instances:
            out.extend(f"[{r.instance_id}] {e}" for e in r.errors)
        return out


def canonical_natural_evidence(natural_evidence: Any) -> str:
    """Serialize natural_evidence to a canonical, deterministic string.

    Stable key ordering (sort_keys) and no whitespace variance
    (compact separators) so the resulting hash is deterministic
    across runs and machines.
    """
    return json.dumps(natural_evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_evidence_hash(natural_evidence: Any) -> str:
    """SHA-256 hex digest of the canonical natural_evidence serialization.

    Used by run_experiment.py to assert byte-identical evidence across
    control / attack / attack+intervention runs by comparing hashes
    rather than deep equality.
    """
    canonical = canonical_natural_evidence(natural_evidence)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_fields(obj: Any, required: tuple[str, ...], label: str) -> list[str]:
    errors = []
    if not isinstance(obj, dict):
        return [f"{label} must be an object"]
    for field_name in required:
        if field_name not in obj:
            errors.append(f"{label} missing required field '{field_name}'")
    return errors


def validate_instance_schema(instance: Any) -> InstanceResult:
    """Validate a single claim instance against the structural schema.

    Corresponds to rules A, C-L in README.md "Validation Rules"
    (rule B, duplicate-id detection, is dataset-level and lives in
    validate_dataset).
    """
    instance_id = instance.get("id") if isinstance(instance, dict) else None
    result = InstanceResult(instance_id=instance_id)
    errors = result.errors

    if not isinstance(instance, dict):
        errors.append("instance must be an object")
        return result

    # A. required top-level fields
    errors.extend(_require_fields(instance, REQUIRED_TOP_LEVEL_FIELDS, "instance"))

    # C. experiment
    experiment = instance.get("experiment")
    if "experiment" in instance and experiment not in EXPERIMENTS:
        errors.append(f"experiment '{experiment}' is not one of {sorted(EXPERIMENTS)}")

    # D. ground_truth
    ground_truth = instance.get("ground_truth")
    if "ground_truth" in instance and ground_truth not in GROUND_TRUTH_VALUES:
        errors.append(f"ground_truth '{ground_truth}' must be one of {sorted(GROUND_TRUTH_VALUES)}")

    # E. ground_truth_basis structure
    ground_truth_basis = instance.get("ground_truth_basis")
    if "ground_truth_basis" in instance:
        errors.extend(
            _require_fields(ground_truth_basis, REQUIRED_GROUND_TRUTH_BASIS_FIELDS, "ground_truth_basis")
        )

    # F, G. natural_evidence
    natural_evidence = instance.get("natural_evidence")
    natural_source_ids: set[Any] = set()
    if "natural_evidence" in instance:
        if not isinstance(natural_evidence, list) or len(natural_evidence) == 0:
            errors.append("natural_evidence must be a non-empty array")
        else:
            for i, item in enumerate(natural_evidence):
                item_errors = _require_fields(
                    item, REQUIRED_NATURAL_EVIDENCE_FIELDS, f"natural_evidence[{i}]"
                )
                errors.extend(item_errors)
                if isinstance(item, dict) and "source_id" in item:
                    natural_source_ids.add(item["source_id"])

    # E2. ground_truth_basis.source_ids must resolve within natural_evidence
    if isinstance(ground_truth_basis, dict) and "source_ids" in ground_truth_basis:
        source_ids = ground_truth_basis["source_ids"]
        if not isinstance(source_ids, list) or len(source_ids) == 0:
            errors.append("ground_truth_basis.source_ids must be a non-empty array")
        else:
            for sid in source_ids:
                if sid not in natural_source_ids:
                    errors.append(
                        f"ground_truth_basis.source_ids references '{sid}' "
                        "which is not present in natural_evidence"
                    )

    # H. adversarial_evidence structure
    adversarial_evidence = instance.get("adversarial_evidence")
    if "adversarial_evidence" in instance:
        errors.extend(
            _require_fields(adversarial_evidence, REQUIRED_ADVERSARIAL_EVIDENCE_FIELDS, "adversarial_evidence")
        )

    if isinstance(adversarial_evidence, dict):
        position = adversarial_evidence.get("position")
        attack_type = adversarial_evidence.get("attack_type")

        # H2. position enum
        if "position" in adversarial_evidence and position not in POSITIONS:
            errors.append(f"adversarial_evidence.position '{position}' must be one of {sorted(POSITIONS)}")

        # I. attack_type must equal parent experiment
        if "attack_type" in adversarial_evidence and "experiment" in instance:
            if attack_type != experiment:
                errors.append(
                    f"adversarial_evidence.attack_type '{attack_type}' must equal "
                    f"experiment '{experiment}'"
                )

        # H3. position/attack_type consistency
        required_position = REQUIRED_POSITION_BY_ATTACK_TYPE.get(attack_type)
        if required_position is not None and position in POSITIONS and position != required_position:
            errors.append(
                f"attack_type '{attack_type}' requires position "
                f"'{required_position}', got '{position}'"
            )

    # J. control_variant must be boolean
    if "control_variant" in instance and not isinstance(instance["control_variant"], bool):
        errors.append("control_variant must be a boolean")

    # L. compute and expose the frozen-evidence hash whenever natural_evidence is well-formed enough to hash
    if isinstance(natural_evidence, list):
        result.natural_evidence_hash = compute_evidence_hash(natural_evidence)

    return result


def validate_dataset(instances: Any, source: str = "<dataset>") -> DatasetResult:
    """Validate an entire dataset (list of instances) structurally.

    Includes rule B (unique id within dataset) at the dataset level;
    all other rules are checked per-instance.
    """
    result = DatasetResult(source=source)

    if not isinstance(instances, list):
        result.dataset_errors.append("dataset must be a JSON array of instances")
        return result

    seen_ids: dict[Any, int] = {}
    for i, instance in enumerate(instances):
        instance_result = validate_instance_schema(instance)
        result.instances.append(instance_result)

        inst_id = instance.get("id") if isinstance(instance, dict) else None
        if inst_id is not None:
            if inst_id in seen_ids:
                result.dataset_errors.append(f"duplicate id '{inst_id}' (first seen at index {seen_ids[inst_id]}, again at {i})")
            else:
                seen_ids[inst_id] = i

    return result


@dataclass
class CompletionReport:
    """Distribution report used by --complete mode."""

    counts_by_experiment: dict[str, int]
    true_count: int
    false_count: int
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_completion(all_instances: list[dict]) -> CompletionReport:
    """Check exact per-experiment counts (hard requirement) and report
    the TRUE/FALSE distribution (report only, not enforced — see
    README.md "Ground Truth Rules").
    """
    counts_by_experiment: dict[str, int] = {exp: 0 for exp in EXPERIMENTS}
    true_count = 0
    false_count = 0

    for instance in all_instances:
        if not isinstance(instance, dict):
            continue
        experiment = instance.get("experiment")
        if experiment in counts_by_experiment:
            counts_by_experiment[experiment] += 1
        gt = instance.get("ground_truth")
        if gt == "TRUE":
            true_count += 1
        elif gt == "FALSE":
            false_count += 1

    report = CompletionReport(
        counts_by_experiment=counts_by_experiment,
        true_count=true_count,
        false_count=false_count,
    )

    for experiment, count in counts_by_experiment.items():
        if count != REQUIRED_INSTANCES_PER_EXPERIMENT:
            report.errors.append(
                f"experiment '{experiment}' has {count} instances, "
                f"requires exactly {REQUIRED_INSTANCES_PER_EXPERIMENT}"
            )

    return report


def load_dataset(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


DEFAULT_DATASET_FILES = ("source_confusion.json", "anchoring.json", "recency.json")


def _default_dataset_dir() -> Path:
    return Path(__file__).parent / "datasets"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate CognitiveGuard datasets.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--schema", action="store_true", help="Structural validation only.")
    mode.add_argument(
        "--complete",
        action="store_true",
        help="Structural validation + exact per-experiment count (10 each) + distribution report.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory containing dataset JSON files (default: backend/datasets).",
    )
    args = parser.parse_args(argv)

    dataset_dir = args.dir or _default_dataset_dir()
    dataset_paths = [dataset_dir / name for name in DEFAULT_DATASET_FILES]

    overall_valid = True
    all_instances: list[dict] = []

    for path in dataset_paths:
        if not path.exists():
            print(f"MISSING: {path}")
            overall_valid = False
            continue

        try:
            instances = load_dataset(path)
        except json.JSONDecodeError as e:
            print(f"INVALID JSON: {path}: {e}")
            overall_valid = False
            continue

        result = validate_dataset(instances, source=str(path))
        status = "OK" if result.valid else "FAIL"
        print(f"[{status}] {path} ({len(result.instances)} instances)")
        for err in result.errors:
            print(f"    - {err}")

        if not result.valid:
            overall_valid = False

        if isinstance(instances, list):
            all_instances.extend(i for i in instances if isinstance(i, dict))

    if args.complete:
        report = validate_completion(all_instances)
        print("\n--- Completion report ---")
        for experiment, count in sorted(report.counts_by_experiment.items()):
            print(f"  {experiment}: {count}/{REQUIRED_INSTANCES_PER_EXPERIMENT}")
        print(f"  TRUE claims: {report.true_count} (target {TARGET_TRUE_COUNT})")
        print(f"  FALSE claims: {report.false_count} (target {TARGET_FALSE_COUNT})")
        for err in report.errors:
            print(f"    - {err}")
        if not report.valid:
            overall_valid = False

    print("\nRESULT:", "PASS" if overall_valid else "FAIL")
    return 0 if overall_valid else 1


if __name__ == "__main__":
    sys.exit(main())
