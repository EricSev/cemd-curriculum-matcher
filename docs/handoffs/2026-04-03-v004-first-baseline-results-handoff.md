# First Baseline Results Handoff

- Date: 2026-04-03
- Version: v004
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Session focus: run the first real benchmark baselines after setting up the native `.venv`

## What changed this session

- Standardized the repo docs toward a Mac-first `.venv/` workflow.
- Fixed a package import issue so headless evaluation does not require `tkinter` at import time.
- Cached the MiniLM sentence-transformer model in the new `.venv`.
- Successfully ran the first baseline evaluations in offline mode.

## Important code change

The matcher package previously imported `tkinter` at module import time through `src/curriculum_matcher/app.py`, which broke:

- tests
- evaluation CLI
- headless usage on Python builds without Tk support

This was fixed by making the GUI imports lazy inside `MatcherApp`.

## Environment notes

The new `.venv` is now the intended local environment convention.

For baseline runs in the current environment, use offline flags because the model is cached locally:

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/evaluate_matcher.py ...
```

Without those flags, the current library stack still attempts Hugging Face metadata requests.

## Baseline outputs created

### Gold benchmark

- Summary:
  `benchmarks/outputs/starter-gold-fast-summary.json`
- Per-record results:
  `benchmarks/outputs/starter-gold-fast-records.csv`

Metrics:

- record count: 100
- top-1 accuracy: 0.41
- top-3 recall: 0.52
- prediction rate: 0.74
- mean top-1 score: 0.4662
- mean correct top-1 score: 0.6713

Confidence bands:

- high: 13 rows, top-1 accuracy 0.7692
- medium: 49 rows, top-1 accuracy 0.5510
- low: 10 rows, top-1 accuracy 0.4000
- no prediction: 26 rows
- very low: 2 rows

Notable interpretation:

- high-confidence predictions look promising
- the matcher is still failing to produce any prediction on 26% of the gold set
- top-3 recall at 52% confirms meaningful headroom even on a small curated subset

### Weak-label benchmark

- Summary:
  `benchmarks/outputs/starter-weak-fast-summary.json`
- Per-record results:
  `benchmarks/outputs/starter-weak-fast-records.csv`

Metrics:

- record count: 250
- top-1 accuracy: 0.488
- top-3 recall: 0.66
- prediction rate: 0.824
- mean top-1 score: 0.4045
- mean correct top-1 score: 0.5193

Confidence bands:

- high: 4 rows, top-1 accuracy 1.0
- medium: 76 rows, top-1 accuracy 0.6711
- low: 115 rows, top-1 accuracy 0.5826
- no prediction: 44 rows
- very low: 11 rows

Notable interpretation:

- weak-label results are somewhat better than the gold set, which is expected
- assessment records are a major weak area
- supplemental products perform much better than assessment records

## Tests

Verified:

- `source .venv/bin/activate && python -m unittest discover -s tests`

Result:

- 10 tests passed

## Most important current findings

1. The matcher now has real baseline numbers, not just intuition.
2. The current fast matcher is far from production-ready if the goal is high-confidence auto-matching.
3. Confidence bands appear directionally useful, but calibration is not trustworthy yet.
4. Non-prediction rate is a major issue and deserves direct analysis.
5. Product-type slices, especially Assessment, look like a likely early optimization target.

## Recommended next step

Use the saved per-record outputs to perform the first structured error analysis:

1. review no-prediction rows from the gold set
2. review top-1 wrong / top-3 right cases
3. review top-3 misses
4. cluster failure patterns by:
   - publisher variants
   - missing publisher
   - assessments
   - grade mismatch
   - title normalization issues

That analysis should drive the first targeted matcher improvements rather than guessing at weights.
