# Next-Phase Review Task List

- Date: 2026-04-05
- Status: `in progress`
- Purpose: live review checklist for the next work cycle
- Review rule: only one experiment-changing task may be `in progress` at a time

## Status Key

- `not started`
- `in progress`
- `blocked`
- `measured`
- `accepted`
- `rejected`
- `deferred`

## Current Locked Baseline

Unless a later task clearly beats it, hold this constant:

- stage-1 recall: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist size: `10`
- model: `gpt-5.4-mini`
- reasoning effort: `medium`

Locked baseline artifacts:

- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-scored-summary.json`
- `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54-vs-gpt54mini-medium-comparison.json`

Latest checkpoint:

- `2026-04-14 04:45 PM MST`: `G3` measured and rejected; removing derived ambiguity labels from `DISTRICT_ROW` did not beat the canonical accepted `F2` baseline
- changed only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py` by removing `usage_ambiguity`, `state_specific_risk`, `placeholder_mapping`, and `assessment_slice`, then restored the accepted `F2` baseline after measurement
- saved `G3` artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-batch-scored-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-batch-pipeline-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-vs-f2-comparison.json`
- the original `G3` batch eventually completed cleanly with `455` scored rows and `0` failed requests
- shared-row comparison versus the canonical accepted `F2` baseline used all `455` rows
- overall deltas versus the canonical accepted `F2` baseline: selection rate `+0.0044`, all-row top-1 `-0.0132`, selected-row top-1 `-0.0292`
- targeted ambiguity slices did not improve: `catalog_state_specific_expected` top-1 `-0.1154`, `catalog_unspecified` top-1 `-0.0208`, assessment rows top-1 `+0.0000`, and wrong-top1-but-gold-in-shortlist rows top-1 `-0.0429`
- reject `G3`; the accepted rerank prompt baseline remains `F2` no-score exposure and no experiment is currently in progress
- the next roadmap movement is to lift exactly `G4` if continuing rerank prompt-contract experiments
- `2026-04-14 04:15 PM MST`: `G3` lifted and started; the code change and prompt/request artifacts are ready, but the live OpenAI batch is still in progress and has not produced a scored output yet
- changed only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py` by removing `usage_ambiguity`, `state_specific_risk`, `placeholder_mapping`, and `assessment_slice`
- updated `tests/test_llm_rerank.py` to assert those four derived ambiguity labels are absent while `confidence_band` and `match_selected_strategy` remain present
- verified locally:
  - `PYTHONPATH=src ./.venv/bin/python -m unittest tests/test_llm_rerank.py`
  - result: `5` tests passed
- saved `G3` pre-result artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-prompts.jsonl`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g3-derived-labels-requests.jsonl`
- live batch state at checkpoint:
  - batch id `batch_69debc7911b08190b55cccfdbba7d76c`
  - status `in_progress`
  - request counts: total `455`, completed `0`, failed `0`
- no keep / reject decision yet because the batch has not reached a terminal state
- next roadmap movement is to resume `G3` from the live batch, download and score the output when complete, compare it against the canonical `F2` baseline, and then decide accept / reject before touching `G4`
- `2026-04-14 02:40 PM MST`: `G2` measured and rejected; removing matcher-internal confidence labels did not beat the canonical accepted `F2` no-score baseline cleanly
- changed only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py` by removing `confidence_band` and `match_selected_strategy`
- saved `G2` artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-batch-scored-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-batch-pipeline-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-vs-f2-comparison.json`
- shared-row comparison versus the canonical accepted `F2` baseline used `450` shared rows because the `G2` batch returned `450` scored rows with `5` failed requests
- shared-row deltas versus the canonical accepted `F2` baseline: selection rate `-0.0111`, all-row top-1 `+0.0000`, selected-row top-1 `+0.0089`
- targeted recovery improved slightly on wrong-top1-but-gold-in-shortlist rows, but `catalog_state_specific_expected` regressed and the candidate did not clear the promotion threshold
- reject `G2`; the accepted rerank prompt baseline remains `F2` no-score exposure and the accepted comparison-contract baseline remains the row-matched `F2` artifact set
- no active experiment remains in progress after the `G2` closeout; the next roadmap movement is to restore the accepted `F2` code path and then lift exactly `G3`
- `2026-04-14 02:35 PM MST`: `G2` lifted and started; code change and prompt/request artifacts are ready, but the live OpenAI batch is still in progress and has not produced a final scored output yet
- changed only the `DISTRICT_ROW` payload in `src/curriculum_matcher/llm_rerank.py` by removing `confidence_band` and `match_selected_strategy`
- verified locally that the generated prompt payload omits both fields and that targeted rerank tests still pass
- saved `G2` pre-result artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-prompts.jsonl`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-g2-internal-labels-requests.jsonl`
- live batch state at checkpoint:
  - batch id `batch_69de7b254d408190b7bbd31563c9d421`
  - status `in_progress`
  - request counts: total `455`, completed `413`, failed `0`
- no keep / reject decision yet because the batch has not reached a terminal state
- next roadmap movement is to resume `G2` from the live batch, download/score the output when complete, compare it against the canonical `F2` baseline, and then decide accept / reject before touching `G3`
- `2026-04-14 01:55 PM MST`: `G1` accepted; the accepted `F2` no-score run has been frozen as the official row-matched rerank baseline artifact set for post-`F` experiments
- changed only the reporting / comparison contract and did not change matcher, retrieval, rerank behavior, shortlist, model, or reasoning settings
- saved `G1` artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-canonical-baseline.json`
  - `docs/analysis/2026-04-14-canonical-prompt-pack-alignment-review.md`
- accepted `F2` row-match validation: prompts `455`, scored rows `455`, prompt-minus-scored `0`, scored-minus-prompt `0`
- older pre-`F2` historical baseline remains useful for history, but not as the default direct comparison base because it still has prompt/scored row mismatch
- accept `G1`; no behavioral baseline changed, but the comparison-contract baseline now points to the accepted `F2` row-matched artifact set
- no active experiment remains in progress after the `G1` checkpoint; the next roadmap movement is to lift exactly `G2`
- `2026-04-14 01:20 PM MST`: post-`F` experiment set defined after repo cleanup checkpoint; no matcher, retrieval, rerank, or shortlist behavior changed in this pass
- confirmed the accepted rerank prompt baseline remains `F2` no-score exposure on top of matcher `C1 + C2 + C5`, retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- confirmed the `F` series is fully closed: `F2` accepted, `F1` / `F3` / `F4` rejected
- defined a new `G` series focused on rerank prompt input-contract cleanup rather than more top-level wording changes
- saved roadmap definition note:
  - `docs/roadmap/2026-04-14-post-f-experiment-set.md`
- no active experiment remains in progress after this checkpoint; the next roadmap movement is to lift exactly `G1`, then `G2` if continuing behavioral experiments
- `2026-04-06 09:34 AM MST`: `F4` measured and rejected; stricter abstention wording improved selected-row precision but lost too much overall recovery versus the accepted `F2` no-score prompt baseline
- changed only the abstention guidance sentence in `src/curriculum_matcher/llm_rerank.py`, ran the batch, and then restored the accepted `F2` prompt baseline after measurement
- saved `F4` artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f4-abstention-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f4-abstention-batch-scored-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f4-abstention-vs-f2-common424-comparison.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f4-abstention-slice-report.json`
- `F2` and `F4` aligned cleanly on the same `455` prompt rows
- shared-row deltas versus the accepted `F2` baseline: selection rate `-0.0747`, all-row top-1 `-0.0066`, selected-row top-1 `+0.0526`
- the stricter wording was too conservative for the accepted operating point: `catalog_state_specific_expected` top-1 fell `-0.0769`, and wrong-top1-but-gold-in-shortlist rows fell `-0.0214`
- reject `F4`; the accepted prompt baseline remains `F2` no-score exposure, with matcher `C1 + C2 + C5`, retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- no active experiment remains in progress after the `F4` closeout; the next roadmap movement is to define a new experiment set before changing matcher, retrieval, or rerank behavior again
- `2026-04-06 08:46 AM MST`: `F3` measured and rejected; adding a compact candidate disambiguation block did not beat the accepted `F2` no-score prompt baseline
- changed prompt structure only in `src/curriculum_matcher/llm_rerank.py` by adding a derived `CANDIDATE_DISAMBIGUATION` section, ran the batch, and then restored the accepted `F2` prompt baseline after measurement
- saved `F3` artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f3-disambiguation-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f3-disambiguation-batch-scored-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f3-disambiguation-vs-f2-common424-comparison.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f3-disambiguation-slice-report.json`
- unlike the earlier historical-baseline comparisons, the accepted `F2` scored output and the `F3` candidate aligned cleanly on the same `455` prompt rows
- shared-row deltas versus the accepted `F2` baseline: selection rate `-0.0022`, all-row top-1 `-0.0022`, selected-row top-1 `-0.0027`
- `F3` also introduced one invalid selected id where the accepted `F2` baseline had `0`
- targeted ambiguity slices did not improve overall: `catalog_state_specific_expected` stayed flat, while `Assessment`, `assessment_short_or_acronym_title`, and wrong-top1-but-gold-in-shortlist rows all regressed
- reject `F3`; the accepted prompt baseline remains `F2` no-score exposure, with matcher `C1 + C2 + C5`, retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- next roadmap movement is to lift exactly `F4` if continuing the `F` series
- `2026-04-06 07:07 AM MST`: `F2` measured and accepted; removing raw matcher scores from the rerank prompt beat the accepted baseline cleanly on the shared scored-row comparison
- changed prompt payload exposure only in `src/curriculum_matcher/llm_rerank.py` by suppressing candidate `score` while keeping candidate order unchanged
- saved `F2` artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-batch-scored-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-vs-baseline-common424-comparison.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f2-no-score-slice-report.json`
- because the accepted historical scored artifact and the current error-row prompt pack still do not align one-to-one, the keep / accept decision was made on the `424` shared `selection_identifier` rows present in both scored outputs
- shared-row deltas versus the accepted baseline: selection rate `+0.0378`, all-row top-1 `+0.0330`, selected-row top-1 `+0.0379`
- target recovery movement was strong: wrong-top1-but-gold-in-shortlist rows improved top-1 by `+0.1111`, and `catalog_state_specific_expected` improved top-1 by `+0.3333`
- accept `F2`; the rerank prompt baseline now suppresses raw matcher scores while keeping the accepted matcher, retrieval, shortlist, model, and reasoning baselines fixed
- next roadmap movement is to lift exactly `F3` if continuing the `F` series
- `2026-04-06 07:07 AM MST`: `F1` measured and rejected; tightening the rerank `SYSTEM_PROMPT` improved all-row recovery but did not beat the accepted baseline cleanly
- changed `SYSTEM_PROMPT` only in `src/curriculum_matcher/llm_rerank.py`, rebuilt the prompt pack, ran a fresh `gpt-5.4-mini` / `medium` batch, and then restored the accepted prompt after measurement
- saved `F1` artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f1-system-prompt-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f1-system-prompt-batch-scored-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f1-system-prompt-vs-baseline-common424-comparison.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-f1-system-prompt-slice-report.json`
- because the accepted historical scored artifact and the current error-row prompt pack do not align one-to-one, the keep / reject decision was made on the `424` shared `selection_identifier` rows present in both scored outputs
- shared-row deltas versus the accepted baseline: selection rate `+0.0920`, all-row top-1 `+0.0330`, selected-row top-1 `+0.0015`
- protected slices improved in several places, especially `Assessment` and `catalog_state_specific_expected`, but the candidate achieved that mostly by selecting more often rather than by becoming materially sharper once selected
- reject `F1`; accepted baselines remain locked: matcher `C1 + C2 + C5`, retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- no active experiment remains in progress after the `F1` closeout; the next roadmap movement is to decide whether to lift `F2`
- `2026-04-06 06:16 AM MST`: defined the next experiment set after the `E` series closeout; no matcher, retrieval, rerank, or shortlist behavior was changed in this pass
- read the latest handoff, the live tracker, and the `E1` rejection note before redefining the roadmap
- confirmed the accepted baselines stay locked: matcher `C1 + C2 + C5`, retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- confirmed `E1` and `E2` both remain rejected and should not be reopened without new measured evidence
- defined a new `F` series focused on the LLM rerank prompt contract, which is the cleanest remaining one-variable seam in the current Tkinter batch workflow
- saved roadmap definition note:
  - `docs/roadmap/2026-04-06-next-experiment-set.md`
- next roadmap movement is to lift exactly `F1` and benchmark a system-prompt-only rerank experiment against the accepted batch baseline
- `2026-04-06 10:09 AM MST`: `E1` measured and rejected; the approved cross-encoder cascade produced mixed results and did not beat the locked baseline cleanly
- ran the representative benchmark with the accepted retrieval baseline, shortlist `10`, and `cross-encoder/ms-marco-MiniLM-L-6-v2` as the only rerank variable
- saved benchmark artifacts:
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_cross_encoder_summary.json`
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_cross_encoder_records.csv`
- overall deltas versus the accepted char n-gram baseline: top-1 `-0.0020`, top-3 `+0.0120`, hit@10 `-0.0030`, MRR `+0.0017`, nDCG@10 `+0.0008`
- row movement was also mixed: top-1 gains `51` vs losses `53`, and top-10 gains `5` vs losses `8`
- protected slices improved in some assessment-heavy cases, but `adoption_state_high_risk` and `catalog_state_specific_expected` still lost top-1 / MRR ground, and the overall shortlist contract weakened slightly
- reject `E1`; accepted baselines remain locked: retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- no active experiment remains in this roadmap; the next movement requires defining a new next-phase experiment set before changing matcher, retrieval, or rerank behavior
- `2026-04-06 05:28 AM MST`: blocker cleared; `E1` is ready to resume and no benchmark has been run yet
- confirmed again that the stripped automation worktree still does not contain the required roadmap, handoffs, or `src/curriculum_matcher/app.py`, so the live OneDrive-backed repo remains the active project path for this roadmap
- downloaded and staged the approved offline cross-encoder model `cross-encoder/ms-marco-MiniLM-L-6-v2` into the default Hugging Face cache from the live project root
- verified that `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` can now load `CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')` successfully without network access
- stop here after environment unblocking; do not start the cascade benchmark in the same run
- accepted baselines remain locked: retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- next roadmap movement is to reopen `E1` and run the exact cross-encoder cascade benchmark against the locked baseline
- `2026-04-06 05:21 AM MST`: blocker verification only; `E1` remains blocked and no new experiment was started
- confirmed again that the stripped automation worktree still does not contain the required roadmap, handoffs, or `src/curriculum_matcher/app.py`, so the live OneDrive-backed repo remains the active project path for this roadmap
- re-ran the explicit offline cross-encoder load probe from the live project root without changing code or reopening any other roadmap item
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` still could not load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- the probe failed with `LocalEntryNotFoundError` followed by `OSError`, and the default Hugging Face cache scan still showed no staged `cross-encoder/*` artifacts in this environment
- stop here rather than substituting a different reranker or inventing a new roadmap task mid-run
- accepted baselines remain locked: retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- next roadmap movement still requires either staging an approved offline cross-encoder model for `E1` or explicitly defining a new experiment set
- `2026-04-06 04:21 AM MST`: blocker verification only; `E1` remains blocked and no new experiment was started
- confirmed the stripped automation worktree still does not contain the required roadmap, handoffs, or `src/curriculum_matcher/app.py`, so the live OneDrive-backed repo remains the active project path for this roadmap
- re-ran the explicit offline cross-encoder load probe from the live project root without changing code or reopening any other roadmap item
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` still could not load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- the probe failed with `LocalEntryNotFoundError` followed by `OSError`, and the default Hugging Face cache scan still showed no staged `cross-encoder/*` artifacts in this environment
- stop here rather than substituting a different reranker or inventing a new roadmap task mid-run
- accepted baselines remain locked: retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- next roadmap movement still requires either staging an approved offline cross-encoder model for `E1` or explicitly defining a new experiment set
- `2026-04-06 03:51 AM MST`: blocker verification only; `E1` remains blocked and no new experiment was started
- resolved the active project path to the live OneDrive-backed repo because the stripped worktree did not contain the roadmap, handoffs, or `src/curriculum_matcher/app.py`
- re-ran the explicit offline cross-encoder load probe from the live project root without changing code or reopening any other roadmap item
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` still could not load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- the probe failed with `LocalEntryNotFoundError` followed by `OSError`, and the default Hugging Face cache scan still showed no staged `cross-encoder/*` artifacts in this environment
- stop here rather than substituting a different reranker or inventing a new roadmap task mid-run
- accepted baselines remain locked: retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- next roadmap movement still requires either staging an approved offline cross-encoder model for `E1` or explicitly defining a new experiment set
- `2026-04-06 03:21 AM MST`: blocker verification only; `E1` remains blocked and no new experiment was started
- re-ran the explicit offline cross-encoder load probe from the live project root without changing code or reopening any other roadmap item
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` still could not load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- default Hugging Face cache scan still showed no staged `cross-encoder/*` artifacts in this environment
- stop here rather than substituting a different reranker or inventing a new roadmap task mid-run
- accepted baselines remain locked: retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- next roadmap movement still requires either staging an approved offline cross-encoder model for `E1` or explicitly defining a new experiment set
- `2026-04-06 02:51 AM MST`: blocker verification only; `E1` remains blocked and no new experiment was started
- re-ran the explicit offline cross-encoder load probe from the live project root without changing code or reopening any other roadmap item
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` still could not load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- default Hugging Face cache scan still showed no staged `cross-encoder/*` artifacts in this environment
- stop here rather than substituting a different reranker or inventing a new roadmap task mid-run
- accepted baselines remain locked: retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- next roadmap movement still requires either staging an approved offline cross-encoder model for `E1` or explicitly defining a new experiment set
- `2026-04-06 02:22 AM MST`: blocker verification only; `E1` remains blocked and no new experiment was started
- re-ran the explicit offline cross-encoder load probe from the live project root without changing code or reopening any other roadmap item
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` still could not load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- default Hugging Face cache scan still showed no staged `cross-encoder/*` artifacts in this environment
- stop here rather than substituting a different reranker or inventing a new roadmap task mid-run
- accepted baselines remain locked: retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- next roadmap movement still requires either staging an approved offline cross-encoder model for `E1` or explicitly defining a new experiment set
- `2026-04-06 02:21 AM MST`: blocker verification only; `E1` remains blocked and no new experiment was started
- re-ran the offline cross-encoder load probe without changing code or reopening any other roadmap item
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` still could not load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- local Hugging Face cache still shows no staged `cross-encoder/*` artifacts in this environment
- stop here rather than substituting a different reranker or inventing a new roadmap task mid-run
- accepted baselines remain locked: retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`, shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`
- next roadmap movement still requires either staging an approved offline cross-encoder model for `E1` or explicitly defining a new experiment set
- `2026-04-06 01:34 AM MST`: `E2` top-15 confirmation measured and rejected
- kept `E1` blocked after reconfirming offline mode still cannot load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- compared the locked `D4` baseline at shortlist `10` against a single higher-shortlist candidate: shortlist `15`
- saved benchmark artifacts:
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top15_shortlist_retune_summary.json`
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top15_shortlist_retune_records.csv`
- saved prompt-pack coverage artifacts:
  - `benchmarks/outputs/openai_batch_runs/historical-top10-gpt54mini-medium-batch-prompts-summary.json`
  - `benchmarks/outputs/openai_batch_runs/historical-top15-gpt54mini-medium-batch-prompts-summary.json`
- protected rerank metrics stayed flat: top-1 `+0.0000`, top-3 `+0.0000`, hit@10 `+0.0000`, nDCG@10 `+0.0000`; `MRR` moved only `+0.0003`
- `hit@15` reached `0.6320`, which added only `4` new gold-shortlist rescues at ranks `11-15`
- prompt-pack row count stayed `455`; average candidates per error row rose from `4.1714` to `4.6022`, and `59` error rows expanded beyond `10` candidates
- keep shortlist `10`; `E2` remains rejected and there is no active experiment in this roadmap
- `2026-04-06 01:24 AM MST`: `E2` measured and rejected; increasing shortlist size from `10` to `20` did not improve the locked top-10 quality contract
- representative benchmark deltas versus the accepted `D4` baseline: top-1 `+0.0000`, top-3 `+0.0000`, hit@10 `+0.0000`, nDCG@10 `+0.0000`, MRR `+0.0003`
- only `4` rows moved from miss to ranks `11-20`, so the larger shortlist adds prompt cost without improving the accepted rerank operating range
- baseline remains shortlist `10`; `E1` stays blocked pending an offline cross-encoder cache
- `2026-04-06 12:31 AM MST`: `E1` lifted from deferred, then blocked before code changes
- offline verification confirmed no cached cross-encoder model is available in this environment
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python` could not load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- stop here rather than substituting a different reranker or weakening the one-variable experiment rule
- `2026-04-06 12:20 AM MST`: overnight validation only; no active experiment was reopened
- source and tests still match the accepted `D4` baseline
- next roadmap movement requires explicitly lifting `E1` or `E2` from `deferred` before any experiment or code change

Running baseline note:

- accepted retrieval default: `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- `gpt-5.4-mini` remains the default rerank model for the next phase.
- `D4` confirmed the retrieval default stays on the accepted char n-gram blend and shortlist stays `10`.
- `gpt-5.4` does not become the default unless a later checkpoint explicitly reverses this decision.
- accepted normalization changes can update the matcher baseline as long as rerank settings stay fixed.

## Review Package Required For Every Measured Task

- updated benchmark artifact(s)
- one short delta summary
- one explicit keep / reject decision
- one note on what becomes the baseline, if anything

## Live Task Tracker

| ID | Status | Task | Single Variable / Scope | Required Review Focus | Required Output | Baseline Update Rule |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | `accepted` | Confirm locked rerank baseline | none | rerank comparison decision still holds | comparison artifact already saved | no change unless a later accepted task beats it |
| A2 | `accepted` | Record official rerank comparison decision | none | confirm `gpt-5.4-mini` remains default | running checkpoint note | no change |
| B1 | `accepted` | Verify shortlist metric contract | reporting contract only | `hit@1`, `hit@3`, `hit@10`, `MRR`, `nDCG@10` plus required slices | representative top-10 summary + bottleneck snapshot | no change |
| B2 | `accepted` | Standing comparison table template | reporting format only | every experiment can be appended without new format decisions | completed comparison table in checkpoint or review note | no change |
| B3 | `accepted` | Fixed worst bottleneck slices section | checkpoint format only | worst overall family, assessment, placeholder, and state-risk slices always called out | completed checkpoint section | no change |
| C1 | `accepted` | Publisher alias normalization experiment | publisher alias normalization only | publisher-variant rows, `Assessment`, `catalog_unspecified` | review note: `docs/analysis/2026-04-05-publisher-alias-normalization-review.md` | accepted: safe publisher alias normalization is now part of the matcher baseline |
| C2 | `accepted` | Acronym and abbreviation expansion experiment | acronym handling only | `assessment_short_or_acronym_title`, sparse rows, vendor abbreviations | review note: `docs/analysis/2026-04-05-acronym-alias-review.md` | accepted: safe product-title acronym expansion is now part of the matcher baseline |
| C3 | `rejected` | Grade normalization tightening experiment | grade canonicalization only | grade-range mismatches, elementary vs middle-grade ambiguity | review note: `docs/analysis/2026-04-05-grade-normalization-review.md` | rejected: no matcher baseline change |
| C4 | `rejected` | Edition / year extraction experiment | edition/year parsing only | near-duplicate titles, year-sensitive variants | review note: `docs/analysis/2026-04-05-edition-year-review.md` | rejected: no matcher baseline change |
| C5 | `accepted` | State-specific token normalization experiment | state-variant normalization only | `adoption_state_high_risk`, `catalog_state_specific_expected`, assessment state-specific cases | review note: `docs/analysis/2026-04-05-state-normalization-review.md` | accepted: safe state-specific token normalization is now part of the matcher baseline |
| D1 | `rejected` | Hybrid retrieval with RRF | retrieval fusion only | `hit@10`, MRR, sparse evidence, assessment slices | review note: `docs/analysis/2026-04-05-rrf-retrieval-review.md` | rejected: no retrieval baseline change |
| D2 | `rejected` | Field-aware lexical weighting experiment | lexical weighting only | title-vs-publisher confusion, title-dominant families | review note: `docs/analysis/2026-04-05-field-lexical-weighting-review.md` | rejected: no retrieval baseline change |
| D3 | `accepted` | Character n-gram retrieval experiment | n-gram retrieval only | acronym-heavy, typo-prone, short-title rows | review note: `docs/analysis/2026-04-05-char-ngram-retrieval-review.md` | accepted: char n-gram retrieval is now the retrieval baseline |
| D4 | `accepted` | Retrieval baseline decision checkpoint | compare accepted retrieval candidates only | decide default retrieval path and whether shortlist stays `10` | retrieval recommendation note: `docs/analysis/2026-04-05-retrieval-baseline-decision.md` | accepted: keep the char n-gram retrieval default and shortlist `10` |
| E1 | `rejected` | Cross-encoder pre-LLM reranker experiment | cascade reranker only | top-1 lift, shortlist quality lift, cost stability | review note: `docs/analysis/2026-04-06-cross-encoder-rerank-review.md` | rejected: keep the current rerank baseline |
| E2 | `rejected` | Shortlist-size retuning | shortlist size only | compare top-10 with any justified higher shortlist | shortlist decision note: `docs/analysis/2026-04-06-shortlist-size-retuning-review.md` | rejected: keep shortlist `10` |
| F1 | `rejected` | System prompt tightening | `SYSTEM_PROMPT` wording only | `llm_top1_accuracy_on_all_rows`, selected-row accuracy, selection-rate stability, ambiguity slices | review note: `docs/analysis/2026-04-06-system-prompt-tightening-review.md` | accepted: prompt wording becomes the rerank prompt baseline |
| F2 | `accepted` | Raw matcher-score exposure experiment | prompt score exposure only | wrong-top1-but-gold-in-shortlist rows, selection-rate stability, repaired / invalid ids | review note: `docs/analysis/2026-04-06-score-exposure-review.md` | accepted: score-display policy becomes the rerank prompt baseline |
| F3 | `rejected` | Candidate disambiguation block experiment | derived prompt comparison block only | state-specific, placeholder, and assessment ambiguity slices | review note: `docs/analysis/2026-04-06-disambiguation-block-review.md` | accepted: disambiguation block becomes the rerank prompt baseline |
| F4 | `rejected` | Abstention wording calibration | abstention guidance wording only | selection rate, all-row accuracy, selected-row accuracy, abstain reason mix | review note: `docs/analysis/2026-04-06-abstention-calibration-review.md` | accepted: abstention wording becomes the rerank prompt baseline |
| G1 | `accepted` | Canonical prompt-pack alignment checkpoint | reporting / comparison contract only | row-count parity and row-matched accepted baseline artifacts | review note: `docs/analysis/2026-04-14-canonical-prompt-pack-alignment-review.md` | accepted: `F2` row-matched artifact set becomes the default rerank comparison baseline |
| G2 | `rejected` | Matcher-internal confidence label suppression | remove `confidence_band` and `match_selected_strategy` only | all-row top-1, selected-row top-1, wrong-top1-but-gold-in-shortlist rows, selection-rate stability | review note: `docs/analysis/2026-04-14-matcher-internal-label-suppression-review.md` | accepted: trimmed row-context contract becomes the rerank prompt baseline |
| G3 | `rejected` | Derived ambiguity label suppression | remove `usage_ambiguity`, `state_specific_risk`, `placeholder_mapping`, and `assessment_slice` only | state-specific, placeholder, and assessment ambiguity slices | review note: `docs/analysis/2026-04-14-derived-ambiguity-label-suppression-review.md` | accepted: derived-row-label policy becomes the rerank prompt baseline |
| G4 | `not started` | Candidate series exposure experiment | candidate `series` exposure only | near-duplicate families, wrong-top1-but-gold-in-shortlist rows, selected-row precision | review note: `docs/analysis/2026-04-14-series-exposure-review.md` | accepted: candidate-series exposure policy becomes the rerank prompt baseline |

## Exact Review Workflow

For each task:

1. mark one task `in progress`
2. state the single variable being changed
3. run the benchmark and save artifacts
4. record overall deltas
5. record required hard-slice deltas
6. mark the task `accepted`, `rejected`, or `deferred`
7. update the “current baseline” only if accepted
8. move to the next task

Do not advance until the current task has:

- saved artifacts
- a short written interpretation
- an explicit decision

## Review Defaults

- if a task shows mixed results, default to `rejected`
- do not change core matcher scoring logic during this cycle
- do not run multiple experiment-changing tasks in parallel
- continue using the representative historical benchmark as the default comparison dataset
