"""Target Agent invocation (deferred).

Step 2+ will implement the model call (Nebius) that presents a task +
evidence bundle to the target agent and records its decision. The
target agent is NEVER used to determine ground truth (see README.md,
"Ground Truth Rules"). Not implemented in Step 1.
"""

from __future__ import annotations


def run_target_agent(*args, **kwargs):
    raise NotImplementedError("target_agent is not implemented until a later step")
