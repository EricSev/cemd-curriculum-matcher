# Experiment Comparison Template

Use this table in every experiment review note or checkpoint once a task is measured.

## Comparison Table

| Experiment Name | Variable Changed | Baseline Artifact | Candidate Artifact | Top-1 Delta | Hit@10 Delta | MRR Delta | Key Slice Wins | Key Slice Regressions | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| example-publisher-aliases | publisher alias normalization only | `baseline-summary.json` | `candidate-summary.json` | `+0.0000` | `+0.0000` | `+0.0000` | `Assessment`, `catalog_unspecified` | `state_specific_risk` | `accepted / rejected / deferred` |

## Review Notes Template

- Task ID:
- Status:
- Single variable changed:
- Baseline held constant:
- Overall deltas:
  - top-1:
  - hit@10:
  - MRR:
- Required hard-slice deltas:
  - worst overall family:
  - worst assessment slice:
  - worst placeholder slice:
  - worst state-risk slice:
- Interpretation:
- Decision:
- Does this become the new baseline?
