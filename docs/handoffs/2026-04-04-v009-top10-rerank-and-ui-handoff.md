# Top-10 Rerank And UI Handoff

- Date: 2026-04-04
- Version: v009
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Next-session focus: build the operator/demo UI around the active Tkinter app and switch the LLM rerank workflow from `top-3` to `top-10` candidates

## What happened since v008

- The first live OpenAI Batch rerank run completed successfully on the representative historical benchmark error rows.
- Result quality showed that the LLM is not the main bottleneck yet; shortlist recall is.
- Based on that result, the next logical change is to move the rerank workflow from `top-3` to `top-10`.

## Key live Batch result

The user ran:

- `python scripts/run_openai_batch_pipeline.py --input-jsonl benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_batch_requests.jsonl --prompt-jsonl benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_prompts.jsonl --run-name historical-1000-gpt54mini-batch --poll-interval-seconds 30`

Reported pipeline result:

- uploaded file id: `file-M64Txsg3mpWZ5ZqHzqEuEG`
- batch id: `batch_69d17b7010708190b8822eb493f42e97`
- batch status: `completed`

Score summary:

- reviewed rows: `472`
- LLM selection rate: `0.4216`
- LLM top-1 accuracy on all reviewed rows: `0.1398`
- LLM top-1 accuracy on selected rows: `0.3317`

Primary reason counts:

- `likely_assessment_alias_or_subtype_ambiguity`: `83`
- `likely_evidence_sparsity`: `155`
- `likely_matcher_error`: `32`
- `likely_ui_catalog_selection_ambiguity`: `75`
- `well_supported`: `127`

Saved artifacts from that run:

- batch output:
  - `benchmarks/outputs/openai_batch_runs/historical-1000-gpt54mini-batch-batch-output.jsonl`
- scored CSV:
  - `benchmarks/outputs/openai_batch_runs/historical-1000-gpt54mini-batch-scored.csv`
- scored summary:
  - `benchmarks/outputs/openai_batch_runs/historical-1000-gpt54mini-batch-scored-summary.json`
- pipeline summary:
  - `benchmarks/outputs/openai_batch_runs/historical-1000-gpt54mini-batch-pipeline-summary.json`

## Most important interpretation from the live run

The LLM rerank layer only worked when the correct answer was already in the shortlist.

Post-run inspection showed:

- on reviewed rows where the baseline top-3 did **not** contain the gold answer:
  - `371` rows
  - LLM accuracy: effectively `0.0`
- on reviewed rows where the baseline top-3 **did** contain the gold answer:
  - `101` rows
  - LLM selection rate: about `0.9406`
  - LLM accuracy: about `0.6535`

Conclusion:

- the next best investment is not “use a bigger LLM on the same top-3”
- the next best investment is “feed the LLM a larger candidate set”

That is why `top-10` is now the priority.

## Current code state

### Existing benchmark / QA work

Still valid from v008:

- policy-aware benchmark slicing in:
  - `src/curriculum_matcher/evaluation.py`
- policy-aware QA taxonomy in:
  - `src/curriculum_matcher/qa.py`
- representative QA benchmark artifacts:
  - `benchmarks/qa/historical_07122025_representative_1000_human_qa.csv`
  - `benchmarks/qa/historical_07122025_representative_1000_human_qa_summary.json`

### Existing LLM / Batch workflow

Still valid from v008:

- `src/curriculum_matcher/llm_rerank.py`
- `src/curriculum_matcher/openai_batch.py`
- `scripts/evaluate_llm_rerank.py`
- `scripts/prepare_openai_batch_requests.py`
- `scripts/run_openai_batch.py`
- `scripts/run_openai_batch_pipeline.py`
- `scripts/score_openai_batch_results.py`

### Persistent key support

Still valid:

- repo-root `.env` is gitignored
- `.env.example` exists
- Batch scripts auto-load `.env`

Relevant files:

- `.gitignore`
- `.env.example`
- `src/curriculum_matcher/openai_batch.py`

## Top-10 rerank state

### What is complete

- decision made to move from `top-3` to `top-10`
- code already supports variable `topn_final` in the prompt-pack builder and request prep path

### What still needs to be regenerated

The current prompt pack and batch request files are still based on the `top-3` records file:

- `benchmarks/outputs/historical_07122025_representative_1000_fast_records.csv`
- `benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_prompts.jsonl`
- `benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_batch_requests.jsonl`

To fully switch to `top-10`, next session should regenerate:

1. a `top-10` historical matcher output
2. a `top-10` rerank prompt pack
3. a `top-10` Batch request file

### Command to regenerate the top-10 baseline

Use the local virtualenv:

```bash
cd "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher"
source .venv/bin/activate
python scripts/evaluate_matcher.py \
  --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv \
  --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" \
  --profile fast \
  --topn-final 10 \
  --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_summary.json \
  --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_records.csv
```

Then regenerate prompts:

```bash
python scripts/evaluate_llm_rerank.py \
  --records-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_records.csv \
  --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" \
  --output-jsonl benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_prompts.jsonl \
  --output-json benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_prompts_summary.json \
  --only-errors \
  --topn-final 10
```

Then regenerate Batch requests:

```bash
python scripts/prepare_openai_batch_requests.py \
  --prompt-jsonl benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_prompts.jsonl \
  --output-jsonl benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_batch_requests.jsonl \
  --model gpt-5.4-mini \
  --reasoning-effort low
```

## UI findings and direction

### Yes, the app already has a UI

The active app already contains a Tkinter UI with:

- file pickers
- output directory picker
- progress bar
- status label
- log pane
- start-processing button

Active UI file:

- `src/curriculum_matcher/app.py`

Archived UI references:

- `archive/legacy_gui/gui_curriculum_matcher.py`
- `archive/legacy_gui/gui_curriculum_matcher2.py`
- `archive/legacy_core/enhanced_curriculum_matcher.py`

### Best next UI strategy

Do **not** revive a legacy archive copy as the main path.

Instead:

- extend the active UI in `src/curriculum_matcher/app.py`
- keep the existing matcher workflow intact
- add a second workflow area for the LLM Batch path

## Recommended next-session UI scope

### Minimum viable demo/operator UI

1. Add a clear “LLM Batch Rerank” section or tab to the active Tkinter app.
2. Let the operator choose:
   - matcher records CSV
   - catalog CSV
   - output directory
   - run name
   - model
   - shortlist size (`3`, `10`, maybe `20`)
3. Add buttons for:
   - build prompt pack
   - build Batch request file
   - run Batch pipeline
4. Add visible progress/logging:
   - upload started / finished
   - batch id
   - poll status every cycle
   - output download path
   - scored summary metrics at the end

### Nice immediate improvement even before richer UI

Improve `scripts/run_openai_batch_pipeline.py` to print progress as it runs:

- after upload
- after batch creation
- on every poll interval
- after download
- after scoring

This would reduce the “silent terminal” problem the user experienced.

## Constraints for next session

- do not change matcher logic yet
- prioritize usability and operator feedback
- top-10 rerank path is now the intended path, not top-3
- preserve Batch API cost discipline
- preserve the active Tkinter matcher UI while extending it

## Validation status from this session

Passing checks:

- `PYTHONPATH=src python3 -m pytest tests/test_llm_rerank.py tests/test_qa.py tests/test_evaluation.py -q`
  - `20 passed`

- syntax checks passed for:
  - `scripts/run_openai_batch_pipeline.py`
  - `scripts/run_openai_batch.py`
  - `scripts/prepare_openai_batch_requests.py`
  - `scripts/score_openai_batch_results.py`
  - `src/curriculum_matcher/openai_batch.py`

## Agent-optimized kickoff prompt for next session

Resume work on the curriculum matcher from:

- `docs/handoffs/2026-04-04-v009-top10-rerank-and-ui-handoff.md`
- `docs/handoffs/2026-04-04-v008-llm-batch-and-ui-handoff.md`

Current priority:

- extend the active Tkinter UI in `src/curriculum_matcher/app.py`
- make the Batch rerank workflow usable and demo-friendly
- treat `top-10` candidates as the intended rerank shortlist size

Please:

1. preserve the existing matcher UI flow,
2. add a second UI flow for:
   - building prompt packs
   - preparing Batch request JSONL
   - running the Batch pipeline
   - showing live progress and scored results,
3. make the operator feedback much clearer than the current silent terminal polling,
4. regenerate the rerank artifacts on `top-10` if they are not already present,
5. do not start changing core matcher scoring logic yet.
