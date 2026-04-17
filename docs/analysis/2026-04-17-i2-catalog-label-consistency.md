# I2 Catalog-Label Consistency Audit

- Task: `I2` diagnostic checkpoint
- records rows: `1000`
- focal rows: `330`
- behavior changed: `no`

## Label Families

| family | count | rate |
| --- | --- | --- |
| assessment_like | 273 | 0.8273 |
| catalog_unspecified | 56 | 0.1697 |
| no_information_available | 1 | 0.003 |

## Consistency Issues

| issue | count | rate |
| --- | --- | --- |
| assessment_gold_present_selection_issue | 22 | 0.0667 |
| assessment_like_consistent | 249 | 0.7545 |
| assessment_usage_catalog_type_mismatch | 2 | 0.0061 |
| policy_no_information_placeholder | 1 | 0.003 |
| unspecified_label_but_retrievable | 19 | 0.0576 |
| unspecified_structural_placeholder | 37 | 0.1121 |

## Family by Issue

```json
{
  "assessment_like": {
    "assessment_gold_present_selection_issue": 22,
    "assessment_like_consistent": 249,
    "assessment_usage_catalog_type_mismatch": 2,
    "policy_no_information_placeholder": 0,
    "unspecified_label_but_retrievable": 0,
    "unspecified_structural_placeholder": 0
  },
  "catalog_unspecified": {
    "assessment_gold_present_selection_issue": 0,
    "assessment_like_consistent": 0,
    "assessment_usage_catalog_type_mismatch": 0,
    "policy_no_information_placeholder": 0,
    "unspecified_label_but_retrievable": 19,
    "unspecified_structural_placeholder": 37
  },
  "no_information_available": {
    "assessment_gold_present_selection_issue": 0,
    "assessment_like_consistent": 0,
    "assessment_usage_catalog_type_mismatch": 0,
    "policy_no_information_placeholder": 1,
    "unspecified_label_but_retrievable": 0,
    "unspecified_structural_placeholder": 0
  }
}
```

## Example Rows

### `unspecified_structural_placeholder`

- `0096e117f95b2b3426ae12d32bd16c7727dd164aaeae2b3b2d44979f67bc7814`: `Core Curriculum` `Benchmark Workshop` -> expected `Benchmark Advance: Unspecified` / `Benchmark Advance` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `False`, top1 `False`
- `000059a4514304b33d389ec533c8b05cf5b3d716ce6df7ae418cfe0830c917f5`: `Core Curriculum` `McGraw Hill` -> expected `MGH ELA Curriculum: Unspecified` / `MGH ELA` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `False`, top1 `False`
- `0018f7693e6a5ec327ac0e92c908bd3570971271700ad4d568745e73a3893f48`: `Core Curriculum` `Wonders McGraw-Hill` -> expected `Wonders: Unspecified` / `Wonders` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `False`, top1 `False`
- `00009bbe6b6e266a9677f549e5298fca0178f42643c1b186c514d1b0d9cb7a1e`: `Core Curriculum` `Bilingual: Maravillas & Wonders` -> expected `Wonders: Unspecified` / `Wonders` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `False`, top1 `False`
- `001cc7577fe769a770ce78c2547bec4a873c4ee31003d8fac94383372d0a948a`: `Core Curriculum` `Wonders` -> expected `Wonders: Unspecified` / `Wonders` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `False`, top1 `False`
- `001fff84404c968accf947ac6188c5a4d9d637d21fe9ca75df271a6655c10b13`: `Core Curriculum` `National Geographic 6th -8th` -> expected `NatGeo/Cengage ELA Curriculum: Unspecified` / `NatGeo/Cengage ELA` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `False`, top1 `False`

### `unspecified_label_but_retrievable`

- `006b929da661a0f1af4747f7919b1052ca397334b3bf472be6077dc407993aa5`: `Core Curriculum` `Journeys` -> expected `Journeys: Unspecified` / `Journeys` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `True`, top1 `False`
- `0037cbfe50e8c083816a224e296aa279a3786e0d4123cfe9bffa546415b66249`: `Core Curriculum` `95% Group` -> expected `95 Percent Group: Unspecified` / `95 Percent Group` (`Supplemental`); labels `catalog_unspecified`, `not_assessment`; top10 `True`, top1 `True`
- `0135bd985bd9cc5da4a46f73f251773fb605569ce00ff80546cc6f6fc760beb2`: `Core Curriculum` `i-Ready` -> expected `i-Ready, ELA: Unspecified` / `i-Ready` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `True`, top1 `False`
- `00dbe3b52a6667bcf9fe5b9d3999d1a7e5941a7f3d5ac84dcf856393666d66c5`: `Core Curriculum` `SpringBoard Mathematics` -> expected `Springboard Mathematics: Unspecified` / `SpringBoard Mathematics` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `True`, top1 `False`
- `0004835bef6c2280b24c4cb8b2cdcd68696a25b176f0555d641b15671124f59b`: `Core Curriculum` `Everyday Mathematics` -> expected `Everyday Mathematics: Unspecified` / `Everyday Mathematics` (`Core Curriculum`); labels `catalog_unspecified`, `not_assessment`; top10 `True`, top1 `False`
- `0042eba1a071bb6ea8cb076b8527753db2115c7a7304e9b57cd501646f9c0fc7`: `Core Curriculum` `Renaissance Learning STAR Math` -> expected `Renaissance Learning: Unspecified` / `Renaissance` (`Supplemental`); labels `catalog_unspecified`, `not_assessment`; top10 `True`, top1 `True`

### `assessment_bucket_gold_label`

- none

### `assessment_usage_catalog_type_mismatch`

- `00a2b1ad5b5a797f5c52d830edfcc432ef3d33b24f3692248ff062170db92510`: `Assessment` `IXL` -> expected `IXL Language Arts` / `IXL` (`Supplemental`); labels `standard_catalog_mapping`, `assessment_short_or_acronym_title`; top10 `True`, top1 `False`
- `0085d9ae5574dcde5b283a0355cf9860f9cf49164dddf91c76cefefc8ac969a0`: `Assessment` `Success Maker` -> expected `SuccessMaker Math` / `SuccessMaker` (`Supplemental`); labels `standard_catalog_mapping`, `assessment_short_or_acronym_title`; top10 `True`, top1 `True`

### `policy_no_information_placeholder`

- `000fb8d311a31fd1aa20e4184b2b610b2fc0394c1db64abe573457120d1e7e84`: `Assessment` `Not Available` -> expected `Assessment Information Not Publicly Available` / `Assessment Information Not Publicly Available` (`Assessment`); labels `no_information_available`, `assessment_short_or_acronym_title`; top10 `False`, top1 `False`

## Interpretation

- `catalog_unspecified` is not a normal retrieval family: `37` rows are unresolved `Unspecified` catalog placeholders, while `19` are already retrievable despite the label.
- Assessment ambiguity is concentrated in catalog bucket labels: `0` rows point to broad gold labels such as `State Created Assessments` or `Other Assessment - Not in Catalog`.
- Assessment usage/catalog-type mismatch accounts for `2` rows.

## Decision

- accept `I2` as a diagnostic checkpoint
- no behavioral baseline change
- use this audit to drive one narrow derived-label normalization in `I3`
