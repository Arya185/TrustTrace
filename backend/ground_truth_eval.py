"""Ground-Truth Evaluation (deferred).

Step 2+ will implement scoring of the target agent's decision against
the instance's independently-established ground_truth (see
README.md, "Ground Truth Rules"). Ground truth itself is authored
data (dataset_validator.py enforces its schema); this module will
compare an agent decision to that already-fixed ground truth. Not
implemented in Step 1.
"""

from __future__ import annotations


def evaluate(*args, **kwargs):
    raise NotImplementedError("ground_truth_eval is not implemented until a later step")
