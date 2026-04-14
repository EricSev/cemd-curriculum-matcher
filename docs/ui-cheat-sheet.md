# Curriculum Matcher UI Cheat Sheet

This is the restart-from-zero guide for using the curriculum matcher UI after being away from the project.

## 1. Project Location

Main project folder:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`

## 2. Open Terminal In The Project

In Terminal:

```bash
cd "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher"
```

Optional quick check:

```bash
pwd
```

Expected result:

```text
/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher
```

## 3. Start The Virtual Environment

Activate the repo venv:

```bash
source .venv/bin/activate
```

When it is active, your prompt should usually show `(.venv)`.

If the venv is missing, stop and inspect the project before going further.

## 4. Launch The UI

Use this command from the project root:

```bash
PYTHONPATH=src python3 -m curriculum_matcher
```

If you want to force the startup profile:

```bash
PYTHONPATH=src python3 -m curriculum_matcher --profile fast
```

The Tkinter app window should open with:

- `Matcher` tab
- `LLM Batch Rerank` tab
- shared status bar
- shared log pane

## 5. Important File Locations

### Core input files

Representative benchmark source CSV:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/benchmarks/gold/historical_07122025_representative_1000.csv`

Top-10 matcher records CSV:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/benchmarks/outputs/historical_07122025_representative_1000_fast_top10_records.csv`

Top-10 matcher summary JSON:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/benchmarks/outputs/historical_07122025_representative_1000_fast_top10_summary.json`

Catalog CSV currently used for the historical benchmark:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv`

### Batch / rerank artifact locations

Artifact output folder:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/benchmarks/outputs`

Batch run output folder:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/benchmarks/outputs/openai_batch_runs`

Top-10 prompt pack:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_prompts.jsonl`

Top-10 prompt summary:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_prompts_summary.json`

Top-10 Batch request JSONL:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/benchmarks/outputs/historical_07122025_representative_1000_top10_llm_rerank_batch_requests.jsonl`

### Existing completed Batch run artifacts

Completed representative Batch run folder:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/benchmarks/outputs/openai_batch_runs`

Existing completed run outputs:

- `historical-1000-gpt54mini-batch-batch-output.jsonl`
- `historical-1000-gpt54mini-batch-pipeline-summary.json`
- `historical-1000-gpt54mini-batch-scored.csv`
- `historical-1000-gpt54mini-batch-scored-summary.json`

## 6. API Key Reminder

The Batch workflow uses repo-root `.env` loading.

Repo-root `.env` path:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/.env`

Example template:

`/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher/.env.example`

Expected key:

`OPENAI_API_KEY=...`

If the Batch flow fails immediately with an API key error, check `.env` first.

## 7. How To Use The `Matcher` Tab

Use this for the baseline matcher workflow.

Fill in:

- `Input Data File`
- `Product Catalog File`
- `Output Directory`

Check:

- weights add up to about `1.0`
- `Profile` is what you want, usually `fast` unless you specifically want `accurate`

Run:

- click `Start Processing`

What it produces:

- a timestamped CSV in the output directory

Where to look for feedback:

- status bar
- progress bar
- log pane

## 8. How To Use The `LLM Batch Rerank` Tab

This is the demo/operator path for rerank preparation and Batch execution.

### Recommended default setup

Use:

- `Matcher Records CSV`
  - `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_records.csv`
- `Product Catalog File`
  - `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv`
- `Artifact Output Directory`
  - `benchmarks/outputs`
- `Batch Run Output Directory`
  - `benchmarks/outputs/openai_batch_runs`
- `Model`
  - `gpt-5.4-mini`
- `Reasoning Effort`
  - `low`
- `Shortlist Size`
  - `10`
- `Poll Interval`
  - `30`

### Three main buttons

`Build Prompt Pack`

- builds the rerank prompt JSONL
- writes the prompt summary JSON
- useful when you want to inspect the prompt artifact before making API requests

`Prepare Batch Request JSONL`

- converts the prompt pack into OpenAI Batch request JSONL
- useful when you want to stop before upload

`Run Full Batch Pipeline`

- rebuilds prompt and request artifacts first
- uploads the request file
- creates the Batch job
- polls status
- downloads output and errors if present
- scores the output if the Batch completes successfully
- writes pipeline summary and scored outputs

### What the UI logs during a full run

You should expect log messages for:

- prompt-pack generation
- request-file creation
- upload start and uploaded file id
- batch creation and batch id
- repeated polling updates
- output download path
- error download path if present
- scored CSV path
- scored summary path
- final pipeline summary path

### Important behavior

If the selected top-10 matcher records file is missing, the UI will try to regenerate it from the representative benchmark and catalog file.

## 9. Where The Results End Up

### From the `Matcher` tab

Results are saved to the output directory you selected, with a filename like:

`Matcher_Results_YYYYMMDD_HHMMSS.csv`

### From the `LLM Batch Rerank` tab

Prompt and request artifacts are saved under:

`benchmarks/outputs`

Batch downloads and scored outputs are saved under:

`benchmarks/outputs/openai_batch_runs`

Common output files from a completed rerank run:

- `RUN_NAME-batch-output.jsonl`
- `RUN_NAME-batch-errors.jsonl`
- `RUN_NAME-scored.csv`
- `RUN_NAME-scored-summary.json`
- `RUN_NAME-pipeline-summary.json`

## 10. If You Are Returning After A Long Break

Use this checklist:

1. `cd` into the project.
2. Activate `.venv`.
3. Confirm `.env` still has `OPENAI_API_KEY` if you plan to use Batch.
4. Launch the UI with `PYTHONPATH=src python3 -m curriculum_matcher`.
5. In the rerank tab, confirm the catalog CSV path is still correct.
6. Keep `Shortlist Size` on `10` unless you are intentionally experimenting.
7. Read the log pane during long runs instead of assuming the app is stuck.

## 11. How To Close Everything Cleanly

### Close the UI

Use the window close button.

The app should save matcher settings on close.

### Leave the venv

Back in Terminal:

```bash
deactivate
```

Your prompt should stop showing `(.venv)`.

### Optional final cleanup check

If you want to be extra sure you are out of the project session:

```bash
pwd
which python3
```

## 12. One-Minute Restart Version

```bash
cd "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher"
source .venv/bin/activate
PYTHONPATH=src python3 -m curriculum_matcher
```

When done:

```bash
deactivate
```
