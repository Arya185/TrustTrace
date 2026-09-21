"""Intervention Engine (deferred).

Step 2+ will implement the intervention applied before the re-run on
IDENTICAL evidence (verified via dataset_validator.compute_evidence_hash).
The intervention is the ONLY intended change between the baseline
attack run and the intervention run (see README.md, "Evidence
Freezing"). Not implemented in Step 1.
"""

from __future__ import annotations


def apply_intervention(*args, **kwargs):
    raise NotImplementedError("intervention_engine is not implemented until a later step")
