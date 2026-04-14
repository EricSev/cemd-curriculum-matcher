# CEMD Curriculum Matcher

This repository is the canonical home for the Python curriculum matching application. It currently contains the latest Tkinter-based matcher, packaging metadata, and an explicit archive of earlier GUI and Colab experiments.

## Active Layout

```text
curriculum-matcher/
  src/curriculum_matcher/   # Active application package
  scripts/                  # Convenience runner(s)
  docs/                     # Current repo documentation
  data_samples/             # Small non-sensitive sample data only
  tests/                    # Reserved for automated tests
  archive/                  # Legacy scripts and experiments
```

## Primary Entry Points

- Package module: `src/curriculum_matcher/app.py`
- Module run: `python -m curriculum_matcher`
- Script run: `python scripts/run_matcher.py`
- Installed CLI: `curriculum-matcher`

## What The App Does

The matcher scores messy curriculum records against a standard product catalog and writes out:

- top 3 programmatic matches
- fallback-repair audit fields when alternate field interpretations are used
- component scores for semantic name similarity, fuzzy name similarity, publisher, grade, and year
- `human_match_*` QA scores and challenge flags when a prior `product_identifier` already exists on the input row

The current matcher uses a two-stage retrieval flow:

- BM25 plus MiniLM for candidate recall
- optional MPNet reranking in the `accurate` profile

## Setup

Preferred local environment convention:

- use `.venv/` for the local machine-specific virtual environment
- do not commit `.venv/`
- recreate `.venv/` separately on each Mac or Windows machine

This repository currently declares Python `>=3.10` in `pyproject.toml`, so use a Python 3.10+ interpreter when creating the environment.

macOS / Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

If your machine only has `python3` and it is below 3.10, install a newer Python first rather than creating a mismatched environment.

## OpenAI API Key

For Batch API scripts in this repo, the simplest persistent setup is a repo-root `.env` file.

1. Copy `.env.example` to `.env`
2. Put your real `OPENAI_API_KEY` in `.env`
3. Do not commit `.env` (this repo already ignores it in `.gitignore`)

Example:

```bash
cp .env.example .env
```

## Running

GUI mode:

```bash
python -m curriculum_matcher
```

Headless mode:

```bash
python -m curriculum_matcher --headless --profile accurate
```

Evaluation mode:

```bash
PYTHONPATH=src python scripts/evaluate_matcher.py \
  --benchmark-file path/to/benchmark.csv \
  --catalog-file path/to/catalog.csv \
  --profile accurate \
  --output-json benchmarks/outputs/latest-summary.json \
  --output-csv benchmarks/outputs/latest-records.csv
```

After installing the package:

```bash
curriculum-matcher-evaluate \
  --benchmark-file path/to/benchmark.csv \
  --catalog-file path/to/catalog.csv
```

URL evidence audit:

```bash
PYTHONPATH=src python scripts/audit_source_evidence.py \
  --input-file path/to/input.csv \
  --catalog-file path/to/catalog.csv \
  --output-csv benchmarks/outputs/url-audit.csv \
  --output-json benchmarks/outputs/url-audit-summary.json
```

Human-match QA evaluation:

```bash
PYTHONPATH=src python scripts/evaluate_human_match_qa.py \
  --input-file path/to/qa-benchmark.csv \
  --catalog-file path/to/catalog.csv \
  --output-csv benchmarks/outputs/human-qa-records.csv \
  --output-json benchmarks/outputs/human-qa-summary.json
```

LLM rerank prompt-pack generation:

```bash
PYTHONPATH=src python scripts/evaluate_llm_rerank.py \
  --records-csv benchmarks/outputs/historical_07122025_representative_1000_fast_records.csv \
  --catalog-file path/to/catalog.csv \
  --output-jsonl benchmarks/outputs/llm-rerank-prompts.jsonl \
  --output-json benchmarks/outputs/llm-rerank-prompts-summary.json \
  --only-errors
```

OpenAI Batch API request prep for rerank prompts:

```bash
PYTHONPATH=src python scripts/prepare_openai_batch_requests.py \
  --prompt-jsonl benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_prompts.jsonl \
  --output-jsonl benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_batch_requests.jsonl \
  --model gpt-5.4-mini \
  --reasoning-effort low

PYTHONPATH=src python scripts/run_openai_batch.py upload \
  --input-jsonl benchmarks/outputs/historical_07122025_representative_1000_llm_rerank_batch_requests.jsonl

PYTHONPATH=src python scripts/run_openai_batch.py create \
  --input-file-id file-... \
  --endpoint /v1/responses
```

## Data Expectations

Input data should include at least:

- `product_name_raw`
- `publisher_raw`
- `grade`

Catalog data should include at least:

- `product_identifier`
- `product_name`
- `publisher`
- `publisher_prior`
- `series`
- `intended_grades`

## Notes

- Current roadmap and milestone tracker:
  - `docs/roadmap/2026-04-04-roadmap-and-checkpoints.md`
  - `docs/roadmap/2026-04-04-milestone-status.md`
- Current analysis reports:
  - `docs/analysis/2026-04-03-starter-gold-fast-error-analysis.md`
  - `docs/analysis/2026-04-03-starter-gold-field-corruption-report.md`
  - `docs/analysis/2026-04-04-repair-strategy-report.md`
  - `docs/analysis/2026-04-04-human-qa-canonical-report.md`
  - `docs/analysis/2026-04-04-human-qa-tuning-report.md`
  - `docs/analysis/2026-04-04-live-url-pilot-report.md`
- Runtime settings are stored in the repo-root `matcher_settings.json`.
- Older scripts have been moved into `archive/` to keep the active path clear.
- `rank-bm25` is now part of the declared dependencies because the current app requires it.
- Benchmark scaffolding now lives under `benchmarks/`.
- Basic unit tests can be run with `PYTHONPATH=src python -m unittest discover -s tests`.
- The old `venv/` folder should be treated as legacy machine-specific state, not the active environment standard.
