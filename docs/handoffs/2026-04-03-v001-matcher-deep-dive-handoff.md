# Matcher Deep Dive Handoff

- Date: 2026-04-03
- Version: v001
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Session focus: repo cleanup follow-through plus first focused deep dive into the active matcher

## What changed this session

- Reorganized the matcher repo so the active app now lives in `src/curriculum_matcher/app.py`.
- Archived older GUI, core, and Colab variants under `archive/`.
- Added package metadata in `pyproject.toml`.
- Added `scripts/run_matcher.py` as a stable runner.
- Updated `README.md` to reflect the current structure and entry points.
- Added `rank-bm25` to declared dependencies.
- Verified that `python3 scripts/run_matcher.py --help` works after the reorganization.

## Current active entry points

- Module entry: `python -m curriculum_matcher`
- Script entry: `python scripts/run_matcher.py`
- Main implementation: `src/curriculum_matcher/app.py`

## High-level assessment

The current matcher is a credible first-generation hybrid matcher, but it is not close to the best achievable architecture for this problem today.

It already has:

- normalization
- publisher alias handling
- grade parsing
- BM25 candidate recall
- semantic embeddings
- optional reranking profile
- QA scoring of pre-existing human matches

Its main limitations are structural and algorithmic:

- one monolithic file mixes model logic, scoring, batch processing, and Tkinter GUI
- no evaluation pipeline, benchmark harness, or regression tests
- no calibrated confidence model
- hand-weighted scoring instead of learned reranking
- limited candidate generation fields
- significant per-row inefficiencies

## Key code findings

### 1. Core logic and GUI are tightly coupled

The whole app still lives in one module: `src/curriculum_matcher/app.py`.

Relevant sections:

- matcher core starts at line 28
- GUI starts at line 368
- batch processing lives inside the GUI class starting at line 756

This makes it harder to:

- test matching logic independently
- run large experiments
- swap out scoring stages
- expose the matcher as a reusable service

### 2. Candidate generation is narrower than it needs to be

Catalog recall text is built from only:

- product name
- series
- year

Reference:

- `src/curriculum_matcher/app.py:199-203`

Notably absent from retrieval text:

- publisher
- prior publisher
- subject
- product type
- alias tables beyond a tiny hardcoded publisher map
- curated title variants or acronyms

This means the retrieval stage can miss good candidates before reranking ever gets a chance.

### 3. Stage-1 score fusion is simplistic and likely poorly scaled

The matcher combines BM25 and MiniLM recall scores with a simple:

- `0.5 * bm25 + 0.5 * semantic`

Reference:

- `src/curriculum_matcher/app.py:287-295`

This is risky because BM25 and cosine similarity do not naturally live on the same scale. Without normalization or learned weighting, one channel can dominate unpredictably depending on the dataset.

### 4. The reranker is still a hand-weighted formula

Final score is currently:

- semantic name
- fuzzy name
- publisher
- grade
- year

weighted by fixed numbers from code or settings.

Reference:

- `src/curriculum_matcher/app.py:337-343`
- settings file: `matcher_settings.json:2-7`

This is the biggest ceiling on accuracy. The code is still using a manually tuned score recipe rather than learning from corrected examples.

### 5. The current saved weights are probably not useful

Current settings sum to 0.62, not 1.0:

- semantic 0.40
- fuzzy 0.15
- publisher 0.02
- grade 0.00
- year 0.05

Reference:

- `matcher_settings.json:2-7`
- the GUI only warns when weights do not sum to 1.0; it does not enforce it
- `src/curriculum_matcher/app.py:510-524`

This means the current ranking emphasis is likely underweighting publisher and ignoring grade almost entirely.

### 6. There are performance issues inside per-row matching

Each row currently:

- re-encodes the input text for recall
- rebuilds `cat_fast` with `np.vstack(...)`
- may re-encode again with the precise model
- may re-encode again for `human_match_*`

References:

- `src/curriculum_matcher/app.py:288-291`
- `src/curriculum_matcher/app.py:298-300`
- `src/curriculum_matcher/app.py:653-666`

This will hurt scaling and experimentation speed. Several of these tensors should be prepared once per run, not per record.

### 7. `topn_stage1` is effectively ignored

The method signature includes `topn_stage1`, but the method uses an internal `recall_k` derived from profile instead.

Reference:

- `src/curriculum_matcher/app.py:279`
- `src/curriculum_matcher/app.py:294`

This is a sign that the matching API is not yet cleanly controlled from the outside.

### 8. The “cached per row” comment is misleading

The file header and inline comment imply input embedding caching, but there is no real cache structure.

References:

- header comment line 6
- `src/curriculum_matcher/app.py:289`

The code computes embeddings once per row within the method call, but does not cache repeated inputs across rows or reruns.

### 9. Headless mode is still a placeholder, not a real CLI

Headless execution uses hardcoded placeholder paths:

- `your_input_data.csv`
- `product_catalog.csv`
- `results`

Reference:

- `src/curriculum_matcher/app.py:746-754`

This is fine for an experiment, but not for a reliable batch system.

### 10. There is no formal evaluation harness in the repo

The repo currently has no:

- benchmark datasets in active structure
- automated tests
- accuracy reports
- regression checks
- slice-based error analysis

This is the main reason matcher improvement feels uncertain. Without a benchmark loop, it is hard to know whether a change actually improves production usefulness.

## Best interpretation of the current matcher

This matcher evolved in a smart direction:

- early versions used one embedding model and simpler scoring
- v3 moved to a two-stage recall/rerank setup
- QA-oriented outputs were preserved

That means the repo already contains the right instincts.

The next leap should not be “tune weights harder.”
The next leap should be:

- split pipeline stages cleanly
- add evaluation
- strengthen candidate generation
- replace fixed reranking with a learnable or at least feature-rich reranker
- add calibrated confidence bands

## Recommended next engineering sequence

### Phase 1: Make the matcher measurable

- create `tests/` coverage for normalization, grade parsing, year parsing, and score components
- build a benchmark dataset folder with:
  - trusted gold labels
  - weak historical labels
  - edge-case slices
- add a reproducible evaluation runner that outputs:
  - top-1 accuracy
  - top-3 recall
  - accuracy by confidence band
  - error slices by subject, publisher, grade, product type

### Phase 2: Untangle architecture

- split `app.py` into:
  - `normalization.py`
  - `retrieval.py`
  - `scoring.py`
  - `pipeline.py`
  - `cli.py`
  - GUI module only if the Tkinter UI is still worth keeping
- keep GUI as a thin shell over a reusable matcher pipeline

### Phase 3: Improve candidate generation

- build multiple retrieval views:
  - title-only
  - title plus series
  - publisher-aware
  - alias-expanded
  - edition/year-aware
- add explicit alias resources instead of tiny hardcoded maps

### Phase 4: Replace hand-weighting with a real reranker

- build candidate-level features
- train or tune a reranker on adjudicated examples
- compare against the current baseline using the evaluation harness

### Phase 5: Use LLMs selectively

Best LLM uses for this project:

- title normalization suggestions
- publisher/title alias generation
- hard-case adjudication among top candidates
- synthetic edge-case generation
- error clustering and taxonomy

Not recommended as first move:

- sending every row to an LLM as the primary matcher

## Open questions for the next session

- Is the Tkinter GUI still a required operating surface, or should the matcher become CLI-first?
- Where is the best available reviewed match dataset for building a gold set?
- Do we want the first modernization step to be evaluation harness or module refactor?
- Should the QA web app eventually consume matcher run metadata such as matcher version and confidence band?

## Suggested next session starting point

Start with a strict “evaluation-first” sprint:

1. define a benchmark data layout in the repo
2. add a CLI evaluation command
3. write parser and normalization unit tests
4. only then begin redesigning retrieval and reranking

That order will keep future matcher work measurable and much easier to trust.
