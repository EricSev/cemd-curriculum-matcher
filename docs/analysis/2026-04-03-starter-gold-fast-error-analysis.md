# Benchmark Error Analysis

- benchmark: `benchmarks/gold/starter_gold_100.csv`
- profile: `fast`
- record count: 100
- top-1 accuracy: 41.0%
- top-3 recall: 52.0%
- prediction rate: 74.0%

## Failure Buckets

- no prediction: 26 / 100
- top-1 wrong but top-3 right: 11 / 100
- top-3 miss: 48 / 100

## Key Patterns

- Blank publishers underperform the rest of the set: top-1 accuracy 35.1% vs 44.4%, top-3 recall 43.2% vs 57.1%.
- No-prediction rows are concentrated in a few catalog families:
- `StudySync` / `StudySync ELA`: 8
- `IXL` / `IXL Language Arts`: 2
- `Reach for Reading` / `Reach for Reading`: 2
- `Get Ready!` / `Get Ready! (MS)`: 1
- `Benchmark Ready to Advance` / `Benchmark Ready to Advance`: 1
- `Learning A-Z` / `Raz-Kids`: 1
- `Fountas & Pinnell Classroom` / `Fountas & Pinnell Classroom`: 1
- `NoRedInk` / `NoRedInk`: 1
- Wrong top-1 / right top-3 cases are almost entirely same-family confusions: 11 of 11 have the expected and predicted top-1 rows in the same catalog series.
- Top-3 misses are also concentrated in a few families:
- `StudySync` / `StudySync ELA`: 8
- `Benchmark Advance` / `Benchmark Advance`: 4
- `Edmentum ELA` / `Edmentum  ELA: Unspecified`: 3
- `Wonders` / `Wonders: Unspecified`: 2
- `Summit English` / `Summit English 9-12`: 2
- `Benchmark Advance` / `Benchmark Advance: Unspecified`: 2
- `IXL` / `IXL Language Arts`: 2
- `Reach for Reading` / `Reach for Reading`: 2

## Gate Spot Check

- A direct offline spot check against the expected catalog row for the 26 no-prediction cases showed that 25 of 26 would have been rejected by the hard negative gate in `match_record` before final ranking.
- Several of those expected rows still produced plausible downstream scores once publisher and grade were included, which means they were likely recoverable:
- `StudySync` grade 12 vs `StudySync ELA`: semantic `0.6432`, fuzzy `0.3600`, publisher `1.0`, grade `1.0`, weighted score about `0.71`, but still gated out.
- `Get Ready` grade 7 vs `Get Ready! (MS)`: semantic `0.5514`, fuzzy `0.5333`, publisher `1.0`, grade `1.0`, weighted score about `0.70`, but still gated out.
- `Study Sync` grade 8 vs `StudySync ELA`: semantic `0.4512`, fuzzy `0.3600`, publisher `1.0`, grade `1.0`, weighted score about `0.63`, but still gated out.
- This makes the no-prediction bucket look more like a title-normalization and gate-threshold problem than a pure weighting problem.

## Publisher Signal

- No-prediction publisher distribution:
- `[missing]`: 10
- `McGraw Hill`: 5
- `IXL Learning`: 2
- `National Geographic`: 2
- `Vista Higher Learning`: 1
- `McGraw-Hill`: 1
- `Benchmark Education`: 1
- `Houghton Mifflin`: 1
- Top-3 miss publisher distribution:
- `[missing]`: 21
- `McGraw Hill`: 8
- `National Geographic`: 3
- `Houghton Mifflin`: 2
- `Summit`: 2
- `IXL Learning`: 2
- `Vista Higher Learning`: 1
- `McGraw-Hill`: 1

## Representative Cases

- No prediction examples:
- `Study Sync` | publisher=`McGraw Hill` | grade=`8` | expected=`StudySync ELA` | predicted=`[blank]` | expected_rank=`0`
- `Get Ready` | publisher=`Vista Higher Learning` | grade=`7` | expected=`Get Ready! (MS)` | predicted=`[blank]` | expected_rank=`0`
- `StudySync with Integrated ELD (Program 2)` | publisher=`McGraw-Hill` | grade=`5` | expected=`StudySync ELA` | predicted=`[blank]` | expected_rank=`0`
- `Ready to Advance` | publisher=`Benchmark Education` | grade=`TK` | expected=`Benchmark Ready to Advance` | predicted=`[blank]` | expected_rank=`0`
- `Raz Kids` | publisher=`[blank]` | grade=`7` | expected=`Raz-Kids` | predicted=`[blank]` | expected_rank=`0`
- `StudySync` | publisher=`McGraw Hill` | grade=`6` | expected=`StudySync ELA` | predicted=`[blank]` | expected_rank=`0`
- Wrong top-1 but top-3 right examples:
- `Benchmark Listos y Adelante` | publisher=`Benchmark Education Co.` | grade=`6` | expected=`Benchmark Listos y Adelante: Math` | predicted=`Benchmark Adelante` | expected_rank=`3`
- `EL Education/ Expeditionary Learning Education` | publisher=`EL Education` | grade=`K` | expected=`EL Education K-5 Language Arts: Unspecified` | predicted=`EL Education 6-8 Language Arts` | expected_rank=`3`
- `Wit and Wisdom` | publisher=`[blank]` | grade=`2` | expected=`Wit & Wisdom: Unspecified` | predicted=`Wit & Wisdom` | expected_rank=`2`
- `Benchmark Listos y Adelante` | publisher=`Benchmark Education Co.` | grade=`1` | expected=`Benchmark Listos y Adelante: Math` | predicted=`Benchmark Listos y Adelante` | expected_rank=`2`
- `EngageNY` | publisher=`[blank]` | grade=`9` | expected=`EngageNY English Language Arts` | predicted=`EngageNY: A|G|A` | expected_rank=`3`
- `Journeys` | publisher=`Houghton Mifflin Harcourt` | grade=`1` | expected=`Journeys` | predicted=`Journeys` | expected_rank=`2`
- Top-3 miss examples:
- `Study Sync` | publisher=`McGraw Hill` | grade=`8` | expected=`StudySync ELA` | predicted=`[blank]` | expected_rank=`0`
- `Get Ready` | publisher=`Vista Higher Learning` | grade=`7` | expected=`Get Ready! (MS)` | predicted=`[blank]` | expected_rank=`0`
- `Edgenuity` | publisher=`[blank]` | grade=`8` | expected=`Edgenuity ELA` | predicted=`Edgenuity Mathematics` | expected_rank=`0`
- `McGraw Hill MyON` | publisher=`McGraw Hill` | grade=`10` | expected=`myON` | predicted=`McGraw-Hill My Math` | expected_rank=`0`
- `StudySync with Integrated ELD (Program 2)` | publisher=`McGraw-Hill` | grade=`5` | expected=`StudySync ELA` | predicted=`[blank]` | expected_rank=`0`
- `Benchmark Advance/Adelante` | publisher=`Benchmark` | grade=`5` | expected=`Benchmark Advance` | predicted=`Benchmark Adelante` | expected_rank=`0`

## Takeaways

- The current matcher is losing a large share of recoverable rows before final ranking, especially where title normalization is weak or the publisher is blank.
- The first ranking problem is not random confusion across unrelated products; it is mostly within-program grade, edition, or series-sibling confusion.
- The best next matcher changes are likely: stronger title normalization and aliasing, softer handling for missing publishers, and family-aware grade tie-breaking inside the same catalog series.

## Reproduce

```bash
./.venv/bin/python scripts/analyze_benchmark_errors.py \
  --records-csv benchmarks/outputs/starter-gold-fast-records.csv \
  --summary-json benchmarks/outputs/starter-gold-fast-summary.json \
  --output-md docs/analysis/2026-04-03-starter-gold-fast-error-analysis.md
```
