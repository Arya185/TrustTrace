"""Autopsy Engine (deferred).

Step 2+ will implement diagnosis of *why* the target agent produced
an incorrect decision on an attack run (e.g. which cognitive failure
mode markers were present), run only when the agent's decision was
wrong. Not implemented in Step 1.
"""

from __future__ import annotations


def run_autopsy(*args, **kwargs):
    raise NotImplementedError("autopsy_engine is not implemented until a later step")
