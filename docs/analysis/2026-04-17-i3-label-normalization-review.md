# I3 Narrow Label-Normalization Candidate Review

- Task: `I3` narrow label-normalization candidate
- Single variable: split `catalog_unspecified` into named-series versus structural-placeholder labels
- row-level source rows: `330`
- changed labels: `56`
- behavior changed: `no candidate generation, scoring, shortlist, model, or prompt wording changed`

## Baseline Placeholder Counts

| baseline_label | count | rate |
| --- | --- | --- |
| catalog_unspecified | 56 | 0.1697 |
| no_information_available | 1 | 0.003 |
| standard_catalog_mapping | 273 | 0.8273 |

## Candidate Placeholder Counts

| candidate_label | count | rate |
| --- | --- | --- |
| catalog_unspecified_named_series | 46 | 0.1394 |
| catalog_unspecified_structural_placeholder | 10 | 0.0303 |
| no_information_available | 1 | 0.003 |
| standard_catalog_mapping | 273 | 0.8273 |

## Candidate Slice Metrics

| candidate_label | count | top1_accuracy | top10_recall |
| --- | --- | --- | --- |
| catalog_unspecified_named_series | 46 | 0.1087 | 0.413 |
| catalog_unspecified_structural_placeholder | 10 | 0.0 | 0.0 |
| no_information_available | 1 | 0.0 | 0.0 |
| standard_catalog_mapping | 273 | 0.3846 | 0.4689 |

## Example Changed Rows

- `0096e117f95b2b3426ae12d32bd16c7727dd164aaeae2b3b2d44979f67bc7814`: `Core Curriculum` `Benchmark Workshop` -> `Benchmark Advance: Unspecified` / `Benchmark Advance`; `catalog_unspecified` -> `catalog_unspecified_named_series`; top10 `False`, top1 `False`
- `006b929da661a0f1af4747f7919b1052ca397334b3bf472be6077dc407993aa5`: `Core Curriculum` `Journeys` -> `Journeys: Unspecified` / `Journeys`; `catalog_unspecified` -> `catalog_unspecified_named_series`; top10 `True`, top1 `False`
- `000059a4514304b33d389ec533c8b05cf5b3d716ce6df7ae418cfe0830c917f5`: `Core Curriculum` `McGraw Hill` -> `MGH ELA Curriculum: Unspecified` / `MGH ELA`; `catalog_unspecified` -> `catalog_unspecified_structural_placeholder`; top10 `False`, top1 `False`
- `0018f7693e6a5ec327ac0e92c908bd3570971271700ad4d568745e73a3893f48`: `Core Curriculum` `Wonders McGraw-Hill` -> `Wonders: Unspecified` / `Wonders`; `catalog_unspecified` -> `catalog_unspecified_named_series`; top10 `False`, top1 `False`
- `00009bbe6b6e266a9677f549e5298fca0178f42643c1b186c514d1b0d9cb7a1e`: `Core Curriculum` `Bilingual: Maravillas & Wonders` -> `Wonders: Unspecified` / `Wonders`; `catalog_unspecified` -> `catalog_unspecified_named_series`; top10 `False`, top1 `False`
- `001cc7577fe769a770ce78c2547bec4a873c4ee31003d8fac94383372d0a948a`: `Core Curriculum` `Wonders` -> `Wonders: Unspecified` / `Wonders`; `catalog_unspecified` -> `catalog_unspecified_named_series`; top10 `False`, top1 `False`
- `001fff84404c968accf947ac6188c5a4d9d637d21fe9ca75df271a6655c10b13`: `Core Curriculum` `National Geographic 6th -8th` -> `NatGeo/Cengage ELA Curriculum: Unspecified` / `NatGeo/Cengage ELA`; `catalog_unspecified` -> `catalog_unspecified_structural_placeholder`; top10 `False`, top1 `False`
- `00651cbc5f942463812a319366933d94fa75955dc9b06e0e54dc94569f67c05c`: `Core Curriculum` `Wonders` -> `Wonders: Unspecified` / `Wonders`; `catalog_unspecified` -> `catalog_unspecified_named_series`; top10 `False`, top1 `False`
- `008d32e4847526eaac992809f9e95b54108603512de658f63762911d1ef3c72e`: `Core Curriculum` `My Perspectives` -> `myPerspectives: Unspecified` / `myPerspectives`; `catalog_unspecified` -> `catalog_unspecified_structural_placeholder`; top10 `False`, top1 `False`
- `008deaa29d63f1abc00956724bc081248c9f9471d31f8b38f88c6aada79023bb`: `Core Curriculum` `National Geographic` -> `NatGeo/Cengage ELA Curriculum: Unspecified` / `NatGeo/Cengage ELA`; `catalog_unspecified` -> `catalog_unspecified_structural_placeholder`; top10 `False`, top1 `False`

## Decision

- accept `I3` as a diagnostic label-normalization candidate
- no retrieval, scoring, shortlist, model, reasoning, or Tkinter behavior changed
- preserve the split for future prompt/evaluation diagnostics: `46` named-series rows versus `10` structural-placeholder rows
