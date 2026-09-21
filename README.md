# CognitiveGuard

CognitiveGuard is a benchmark, built for the Nebius × NVIDIA Global AI Hackathon, that evaluates whether an AI agent becomes vulnerable to specific cognitive-science-inspired failure modes when exposed to adversarial information.

## Research hypothesis

> Can cognitive-science-inspired behavioral markers explain why an AI agent becomes vulnerable to adversarial information, and can targeted interventions measurably improve robustness?

## The three failure modes

1. **source_confusion** — the agent conflates or misattributes claims across sources of differing reliability.
2. **anchoring** — the agent over-weights information it encounters first, even when later, more reliable evidence should dominate.
3. **recency_bias** — the agent over-weights information it encounters last, regardless of its reliability relative to earlier evidence.

## The experimental loop

```
Task
  → Evidence Bundle
  → Target Agent
  → Ground-Truth Evaluation
  → Autopsy (if incorrect)
  → Intervention
  → Re-run on IDENTICAL evidence
  → Before/After metrics
```

For each claim instance, the target agent is run under three conditions — **control**, **attack**, and **attack + intervention** — and its decisions are scored against ground truth to measure whether the adversarial evidence induced the targeted failure mode, and whether the intervention reduced it.

## Methodological requirement: real facts, synthetic attacks

The benchmark uses **real, stable, uncontroversial facts** as ground truth. CognitiveGuard does not invent fictional entities or fictional ground-truth facts. Only the **adversarial content** is synthetic — it is intentionally constructed to induce a specific cognitive failure mode, and is clearly separated from the natural evidence in the dataset schema (see below).

## Ground truth rules

- Ground truth is established **independently of the target model**. The target agent is **never** used to determine ground truth.
- Ground truth is always **traceable to a specific `source_id`** within that instance's `natural_evidence` array, plus a short human-written rationale (`ground_truth_basis`). Ground truth is never traceable only to an external, unlinked reference — it must be traceable to evidence the agent actually receives.
- Claims must be: stable, objectively verifiable, non-political, non-controversial, non-medical, non-time-sensitive, and answerable as TRUE/FALSE.
- The benchmark should eventually contain approximately 15 TRUE and 15 FALSE claims across the three experiments (10 instances each). This split is not enforced by the validator yet, but `dataset_validator.py --complete` reports it.

**Ground truth is independently established, is always traceable to a specific retrieved source, and is never derived from an LLM judgment.**

### Four distinct things, not to be conflated

| Concept | What it is | Where it lives |
|---|---|---|
| **Ground truth** | The objectively correct TRUE/FALSE answer, fixed by a human before any run. | `ground_truth` + `ground_truth_basis` |
| **Attack type** | The cognitive failure mode the adversarial evidence is designed to induce. | `adversarial_evidence.attack_type` |
| **Agent decision** | What the target agent actually answered, under a given condition. | produced at runtime, scored against ground truth |
| **Autopsy diagnosis** | A post-hoc explanation of *why* the agent got it wrong, run only when the agent's decision was incorrect. | produced at runtime, never used to redefine ground truth |

## Dataset schema

Each claim instance:

```json
{
  "id": "src_conf_001",
  "experiment": "source_confusion",
  "claim": "...",
  "ground_truth": "TRUE",
  "ground_truth_basis": {
    "source_ids": ["natural_001"],
    "rationale": "..."
  },
  "natural_evidence": [
    {
      "source_id": "natural_001",
      "title": "...",
      "url": "...",
      "text": "...",
      "source_type": "authoritative"
    }
  ],
  "adversarial_evidence": {
    "source_id": "attack_001",
    "text": "...",
    "position": "last",
    "attack_type": "source_confusion"
  },
  "control_variant": true
}
```

**Constraint:** every id listed in `ground_truth_basis.source_ids` must also exist as a `source_id` within that same instance's `natural_evidence` array. Ground truth must be traceable to evidence the agent actually receives, not to an external unlinked reference.

## Evidence freezing

For each claim, the natural evidence bundle is retrieved **once** and then **frozen**. The following three conditions must use the **identical** natural evidence, verified by matching SHA-256 hash:

1. clean / control
2. adversarial attack
3. adversarial attack + intervention

The intervention is the **only** intended change between the baseline (attack) run and the intervention run — natural evidence is never re-retrieved or edited between conditions.

`dataset_validator.py` computes this hash as the SHA-256 digest of the canonical serialization of `natural_evidence` (stable key ordering via `sort_keys=True`, no whitespace variance via compact separators), so the hash is deterministic across runs and machines. `run_experiment.py` (implemented in a later step) will assert byte-identical evidence across control / attack / attack+intervention runs by comparing this hash rather than performing deep equality checks, and must never perform fresh retrieval during replay. Tavily integration for the initial retrieval will live in `evidence_builder.py`, added in Step 2.

## Validation

`backend/dataset_validator.py` enforces:

- Required fields exist (A).
- `id` is unique within each dataset (B).
- `experiment` is exactly one of `source_confusion`, `anchoring`, `recency_bias` (C).
- `ground_truth` is exactly `TRUE` or `FALSE` (D).
- `ground_truth_basis` exists and contains `source_ids` and `rationale` (E), and every id in `source_ids` resolves to a `source_id` present in that instance's `natural_evidence` (E2).
- `natural_evidence` is non-empty (F), and every item contains `source_id`, `title`, `url`, `text`, `source_type` (G).
- `adversarial_evidence` exists and contains `source_id`, `text`, `position`, `attack_type` (H), `position` is one of `first` / `middle` / `last` (H2), and `position` is consistent with `attack_type` — `recency_bias` must be `last`, `anchoring` must be `first`, `source_confusion` may be any position (H3).
- `adversarial_evidence.attack_type` equals the parent instance's `experiment` (I).
- `control_variant` is boolean (J).
- Natural evidence text is never modified between control and attack variants (K) — structurally guaranteed, since a single instance's `natural_evidence` array is shared by all three run conditions and verified via the frozen-evidence hash (see above) rather than being duplicated per condition.
- A SHA-256 hash of the canonical `natural_evidence` serialization is computed and exposed per instance (L).

Two validation modes:

```bash
python -m backend.dataset_validator --schema      # structure only, passes on empty datasets
python -m backend.dataset_validator --complete     # structure + exactly 10 instances per experiment + distribution report
```

## Metrics (documented, not yet implemented)

See [`backend/metrics_schema.md`](backend/metrics_schema.md) for the full definitions of `baseline_accuracy`, `intervention_accuracy`, `attack_success_rate`, `intervention_success_rate`, `autopsy_diagnostic_accuracy`, `clean_control_accuracy`, and `clean_control_regression`. No calculation code exists yet — this step only fixes the contract.

## Repository layout

```
cognitiveguard/
├── README.md
├── LICENSE
├── backend/
│   ├── evidence_builder.py       # Tavily retrieval + evidence bundle assembly (Step 2)
│   ├── target_agent.py           # model call + decision capture (later step)
│   ├── ground_truth_eval.py      # score agent decision against ground truth (later step)
│   ├── autopsy_engine.py         # diagnose incorrect attack-condition decisions (later step)
│   ├── intervention_engine.py    # apply intervention before re-run (later step)
│   ├── run_experiment.py         # orchestrates the loop; frozen-evidence data contract only (Step 1)
│   ├── dataset_validator.py      # schema + completeness validation (Step 1, this step)
│   ├── metrics_schema.md         # metrics documentation (Step 1, this step)
│   └── datasets/
│       ├── source_confusion.json # empty until Step 2
│       ├── anchoring.json        # empty until Step 2
│       └── recency.json          # empty until Step 2
├── frontend/                     # empty until a later step
├── results/                      # empty until a later step
└── tests/
    └── test_dataset_validator.py
```

## Status

**Step 1 (this step):** repository scaffolding, dataset schema, strict structural validation, frozen-evidence hashing, metrics documentation, and tests. No claim instances have been written.

**Deferred to later steps:** Tavily-based retrieval (`evidence_builder.py`), target-agent model calls via Nebius (`target_agent.py`), ground-truth scoring and autopsy/intervention logic, the experiment orchestration loop (`run_experiment.py`), metrics calculation, dataset authoring (30 claim instances: 10 per experiment, ~15 TRUE / ~15 FALSE), and the frontend.
