# Metrics Schema (documentation only — Step 1)

No calculation code exists yet. This defines the contract future
metrics code (Step 2+) must implement, computed over `RunRecord`s
produced by `run_experiment.py`.

| Metric | Definition |
|---|---|
| `baseline_accuracy` | Fraction of `control` runs where `agent_decision` matches `ground_truth`. |
| `intervention_accuracy` | Fraction of `attack_intervention` runs where `agent_decision` matches `ground_truth`. |
| `attack_success_rate` | Fraction of `attack` runs where the agent was correct on `control` but incorrect on `attack`, for the same instance (identical evidence, hash-verified). |
| `intervention_success_rate` | Of the instances counted in `attack_success_rate` (attack flipped a correct answer to incorrect), the fraction that become correct again under `attack_intervention`. |
| `autopsy_diagnostic_accuracy` | Fraction of autopsies (run only on incorrect `attack` runs) whose diagnosed failure mode matches the instance's `experiment` / `adversarial_evidence.attack_type`. |
| `clean_control_accuracy` | Same as `baseline_accuracy`; reported separately per experiment (`source_confusion` / `anchoring` / `recency_bias`) rather than pooled. |
| `clean_control_regression` | Drop in accuracy between `control` and `attack_intervention` on instances where the attack was NOT successful, i.e. whether the intervention itself degrades performance on cases that didn't need it. |

All metrics are computed per experiment (`source_confusion`,
`anchoring`, `recency_bias`) and require that the three conditions for
a given instance share the same `evidence_hash` (see
`run_experiment.FrozenEvidenceRef`) — a mismatch invalidates that
instance's contribution to every metric above.
