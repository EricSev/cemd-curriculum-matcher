# I1 Assessment and Catalog-Unspecified Taxonomy Audit

- Task: `I1` diagnostic checkpoint
- records rows: `1000`
- focal rows: `316`
- LLM joined rows: `455`
- behavior changed: `no`

## Bucket Definitions

- `likely_retrieval_miss`: gold is absent from top-10 and no stronger catalog-structure or ambiguity signal is present.
- `catalog_label_or_placeholder_structure`: expected catalog row or derived placeholder label points to `Unspecified`, no-information, district-created, or similar catalog structure.
- `ambiguous_historical_ground_truth`: row evidence is too sparse, acronym-like, assessment-subtype-like, or publisher-missing to treat the historical gold label as a clean retrieval target without review.
- `gold_present_selection_failure`: gold is present in top-10, but the ranker or joined canonical `F2` LLM output did not select it.
- `already_correct_or_recovered`: accepted ranker or joined LLM output already selected the expected gold id.

## Slice Summary

| slice | row_count | top10_absent_count | top10_absent_rate | gold_present_selection_failure_bucket_count | llm_joined_count |
| --- | --- | --- | --- | --- | --- |
| Assessment | 260 | 142 | 0.5462 | 4 | 136 |
| catalog_unspecified | 56 | 37 | 0.6607 | 6 | 48 |

## Overall Focal Bucket Counts

| bucket | count | rate |
| --- | --- | --- |
| already_correct_or_recovered | 127 | 0.4019 |
| ambiguous_historical_ground_truth | 135 | 0.4272 |
| catalog_label_or_placeholder_structure | 38 | 0.1203 |
| gold_present_selection_failure | 10 | 0.0316 |
| likely_retrieval_miss | 6 | 0.019 |

## `Assessment` Bucket Counts

| bucket | count | rate |
| --- | --- | --- |
| already_correct_or_recovered | 114 | 0.4385 |
| ambiguous_historical_ground_truth | 135 | 0.5192 |
| catalog_label_or_placeholder_structure | 1 | 0.0038 |
| gold_present_selection_failure | 4 | 0.0154 |
| likely_retrieval_miss | 6 | 0.0231 |

## `catalog_unspecified` Bucket Counts

| bucket | count | rate |
| --- | --- | --- |
| already_correct_or_recovered | 13 | 0.2321 |
| catalog_label_or_placeholder_structure | 37 | 0.6607 |
| gold_present_selection_failure | 6 | 0.1071 |

## Key Overlaps

### Bucket by Placeholder Mapping

```json
{
  "already_correct_or_recovered": {
    "catalog_unspecified": 13,
    "no_information_available": 0,
    "standard_catalog_mapping": 114
  },
  "ambiguous_historical_ground_truth": {
    "catalog_unspecified": 0,
    "no_information_available": 0,
    "standard_catalog_mapping": 135
  },
  "catalog_label_or_placeholder_structure": {
    "catalog_unspecified": 37,
    "no_information_available": 1,
    "standard_catalog_mapping": 0
  },
  "gold_present_selection_failure": {
    "catalog_unspecified": 6,
    "no_information_available": 0,
    "standard_catalog_mapping": 4
  },
  "likely_retrieval_miss": {
    "catalog_unspecified": 0,
    "no_information_available": 0,
    "standard_catalog_mapping": 6
  }
}
```

### Bucket by Assessment Slice

```json
{
  "already_correct_or_recovered": {
    "assessment_other": 4,
    "assessment_publisher_missing": 29,
    "assessment_short_or_acronym_title": 75,
    "assessment_state_specific_expected": 6,
    "not_assessment": 13
  },
  "ambiguous_historical_ground_truth": {
    "assessment_other": 1,
    "assessment_publisher_missing": 49,
    "assessment_short_or_acronym_title": 84,
    "assessment_state_specific_expected": 1,
    "not_assessment": 0
  },
  "catalog_label_or_placeholder_structure": {
    "assessment_other": 0,
    "assessment_publisher_missing": 0,
    "assessment_short_or_acronym_title": 1,
    "assessment_state_specific_expected": 0,
    "not_assessment": 37
  },
  "gold_present_selection_failure": {
    "assessment_other": 0,
    "assessment_publisher_missing": 3,
    "assessment_short_or_acronym_title": 1,
    "assessment_state_specific_expected": 0,
    "not_assessment": 6
  },
  "likely_retrieval_miss": {
    "assessment_other": 6,
    "assessment_publisher_missing": 0,
    "assessment_short_or_acronym_title": 0,
    "assessment_state_specific_expected": 0,
    "not_assessment": 0
  }
}
```

## Example Rows

### `likely_retrieval_miss`

- `001de09bad8760f0e38cad4082fd668b32f977c85400b1153f3758a7571b6c0e`: `Assessment` `Arizona's Academic Standards Assessment` / expected `State Created Assessments`; labels `standard_catalog_mapping`, `assessment_other`, `rich`; top10 `False`, top1 `False`; LLM `abstain` `likely_matcher_error`
- `002bf91b0f9d846d1ff779cc4db772b9cd83ec23aaa8270339df2c8e04aa2532`: `Assessment` `Renaissance Star Testing ELA` / expected `Star Assessments`; labels `standard_catalog_mapping`, `assessment_other`, `rich`; top10 `False`, top1 `False`; LLM `select_candidate` `likely_assessment_alias_or_subtype_ambiguity`
- `00a82b49972de76d42b9ea6d086d8333cad5d172b36e2e742afca0cac96de085`: `Assessment` `NWEA: MAP Growth: ELA: 3-11` / expected `NWEA: MAP Growth`; labels `standard_catalog_mapping`, `assessment_other`, `rich`; top10 `False`, top1 `False`; LLM `abstain` `likely_evidence_sparsity`
- `00d2f695e296466fd09ccdbc1ba7f0704e31078eab603f9443e436e254e7e8ce`: `Assessment` `Maryland Comprehensive Assessment Program` / expected `State Created Assessments`; labels `standard_catalog_mapping`, `assessment_other`, `rich`; top10 `False`, top1 `False`; LLM `not_joined` `not_joined`
- `01dc75ae75c2a1943f33198ef26ce78b4ab0fed5424651d08bf7bf2d7503ee10`: `Assessment` `i-Ready Math` / expected `i-Ready Assessment`; labels `standard_catalog_mapping`, `assessment_other`, `medium`; top10 `False`, top1 `False`; LLM `select_candidate` `likely_assessment_alias_or_subtype_ambiguity`

### `catalog_label_or_placeholder_structure`

- `000fb8d311a31fd1aa20e4184b2b610b2fc0394c1db64abe573457120d1e7e84`: `Assessment` `Not Available` / expected `Assessment Information Not Publicly Available`; labels `no_information_available`, `assessment_short_or_acronym_title`, `policy_placeholder_no_info`; top10 `False`, top1 `False`; LLM `not_joined` `not_joined`
- `0096e117f95b2b3426ae12d32bd16c7727dd164aaeae2b3b2d44979f67bc7814`: `Core Curriculum` `Benchmark Workshop` / expected `Benchmark Advance: Unspecified`; labels `catalog_unspecified`, `not_assessment`, `sparse`; top10 `False`, top1 `False`; LLM `select_candidate` `well_supported`
- `000059a4514304b33d389ec533c8b05cf5b3d716ce6df7ae418cfe0830c917f5`: `Core Curriculum` `McGraw Hill` / expected `MGH ELA Curriculum: Unspecified`; labels `catalog_unspecified`, `not_assessment`, `sparse`; top10 `False`, top1 `False`; LLM `abstain` `likely_evidence_sparsity`
- `0018f7693e6a5ec327ac0e92c908bd3570971271700ad4d568745e73a3893f48`: `Core Curriculum` `Wonders McGraw-Hill` / expected `Wonders: Unspecified`; labels `catalog_unspecified`, `not_assessment`, `medium`; top10 `False`, top1 `False`; LLM `not_joined` `not_joined`
- `00009bbe6b6e266a9677f549e5298fca0178f42643c1b186c514d1b0d9cb7a1e`: `Core Curriculum` `Bilingual: Maravillas & Wonders` / expected `Wonders: Unspecified`; labels `catalog_unspecified`, `not_assessment`, `rich`; top10 `False`, top1 `False`; LLM `select_candidate` `likely_assessment_alias_or_subtype_ambiguity`

### `ambiguous_historical_ground_truth`

- `00013ac4a2c830963fbfa1f6a2e3affb30cef924a6a00365319867836c80260b`: `Assessment` `MAP` / expected `NWEA: MAP Suite`; labels `standard_catalog_mapping`, `assessment_short_or_acronym_title`, `sparse`; top10 `False`, top1 `False`; LLM `abstain` `likely_assessment_alias_or_subtype_ambiguity`
- `000cabeb3e8de7971ea3576515165625d031d9ea5281dc136dd08dde58b359d7`: `Assessment` `Measures of Academic Progress (MAP)  Growth` / expected `NWEA: MAP Growth`; labels `standard_catalog_mapping`, `assessment_publisher_missing`, `medium`; top10 `False`, top1 `False`; LLM `not_joined` `not_joined`
- `0038157425d060437adf8ce36ec7d4ebe941fb6746b86ade39c1d0ff0fecf984`: `Assessment` `ELPA21` / expected `Other Assessment - Not in Catalog`; labels `standard_catalog_mapping`, `assessment_short_or_acronym_title`, `sparse`; top10 `False`, top1 `False`; LLM `select_candidate` `well_supported`
- `001bc951e763a6de602f6ea466f1cd7d89bc2e0a1db3313e3638e3cb893c1c56`: `Assessment` `AZELLA` / expected `State Created Assessments`; labels `standard_catalog_mapping`, `assessment_short_or_acronym_title`, `sparse`; top10 `False`, top1 `False`; LLM `abstain` `likely_evidence_sparsity`
- `00018814494f2fd3d04bec642dd5a91f2920b98340fe8fff188b20e2b79826d9`: `Assessment` `Summative Alternative ELPAC` / expected `State Created Assessments`; labels `standard_catalog_mapping`, `assessment_publisher_missing`, `medium`; top10 `False`, top1 `False`; LLM `abstain` `likely_matcher_error`

### `gold_present_selection_failure`

- `0023cae523990781b129ba898a1f1eadb38adbda8d986fc01db33b8bcca0897a`: `Assessment` `EOY NWEA MAP: Reading/Math/Science` / expected `NWEA: MAP Suite`; labels `standard_catalog_mapping`, `assessment_publisher_missing`, `medium`; top10 `True`, top1 `False`; LLM `select_candidate` `likely_assessment_alias_or_subtype_ambiguity`
- `002132c5b6925aba95593336619087b5bfe0c3a2dd10b2281f446c276aae50b3`: `Assessment` `NWEA MAP` / expected `NWEA: MAP Reading Fluency`; labels `standard_catalog_mapping`, `assessment_short_or_acronym_title`, `medium`; top10 `True`, top1 `False`; LLM `select_candidate` `likely_assessment_alias_or_subtype_ambiguity`
- `000375fb35bff87b8785631752465dae33f4ef0624c6d73330eddaace850093c`: `Assessment` `Georgia Alternate Assessment 2.0` / expected `State Created Assessments`; labels `standard_catalog_mapping`, `assessment_publisher_missing`, `medium`; top10 `True`, top1 `False`; LLM `abstain` `likely_assessment_alias_or_subtype_ambiguity`
- `00094c066fa9fa4b5880e712a8e4c8e2f3ef38a762c17fee1c701c3bb974ef35`: `Assessment` `TCAP Alternate Assessments` / expected `State Created Assessments`; labels `standard_catalog_mapping`, `assessment_publisher_missing`, `medium`; top10 `True`, top1 `False`; LLM `abstain` `likely_assessment_alias_or_subtype_ambiguity`
- `006b929da661a0f1af4747f7919b1052ca397334b3bf472be6077dc407993aa5`: `Core Curriculum` `Journeys` / expected `Journeys: Unspecified`; labels `catalog_unspecified`, `not_assessment`, `sparse`; top10 `True`, top1 `False`; LLM `select_candidate` `likely_evidence_sparsity`

## Interpretation

- The focal audit covers `316` rows across `Assessment` and `catalog_unspecified` slices.
- Top-10 absence is still substantial: `179` focal rows have gold absent from the accepted top-10 shortlist.
- `Assessment` is mostly an ambiguity / historical-ground-truth problem in this taxonomy: `135 / 260` assessment rows land in `ambiguous_historical_ground_truth`, while only `6 / 260` land in the clean `likely_retrieval_miss` bucket.
- `catalog_unspecified` is mostly a catalog-label / placeholder-structure problem: `37 / 56` rows land in `catalog_label_or_placeholder_structure`.
- Gold-present primary selection failures account for `10` focal rows: `4` assessment rows and `6` `catalog_unspecified` rows.

## Decision

- accept `I1` as a diagnostic checkpoint
- no behavioral baseline change
- use this artifact to decide whether `I2` should inspect catalog-label consistency before any retrieval change
