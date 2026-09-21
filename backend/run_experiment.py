"""Experiment runner (deferred) — Step 1 only defines the frozen-evidence
data contract; no execution logic yet.

The experimental loop this module will eventually drive:

    Task -> Evidence Bundle -> Target Agent -> Ground-Truth Evaluation
         -> Autopsy (if incorrect) -> Intervention
         -> Re-run on IDENTICAL evidence -> Before/After metrics

Frozen-evidence invariant (see README.md, "Evidence Freezing"):
for a given claim instance, the natural_evidence retrieved once must
be byte-identical (verified by SHA-256 hash, not deep equality) across
all three conditions:

    1. clean / control        (natural_evidence only)
    2. adversarial attack     (natural_evidence + adversarial_evidence)
    3. attack + intervention  (same evidence, agent behavior modified)

dataset_validator.compute_evidence_hash(natural_evidence) is the single
source of truth for that hash. This module will call it before each
run and assert equality against the hash recorded at dataset-load
time; it must never re-retrieve evidence (no fresh Tavily calls)
during replay.

Nothing below is implemented yet; run_experiment.main() intentionally
does not exist until model/API integration lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Condition = Literal["control", "attack", "attack_intervention"]


@dataclass(frozen=True)
class FrozenEvidenceRef:
    """Pointer to a claim instance's evidence plus the hash it must match.

    Any code that (re)constructs an evidence bundle for a given
    instance_id + condition must compute
    dataset_validator.compute_evidence_hash(natural_evidence) on the
    result and assert it equals expected_hash before proceeding.
    """

    instance_id: str
    expected_hash: str


@dataclass(frozen=True)
class RunRecord:
    """One (instance, condition) execution result — schema only, not populated by Step 1."""

    instance_id: str
    condition: Condition
    evidence_hash: str
    agent_decision: Any = None
    correct: bool | None = None
    autopsy: dict | None = None
