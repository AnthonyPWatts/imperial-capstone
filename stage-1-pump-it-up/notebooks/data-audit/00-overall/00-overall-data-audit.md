---
status: current
updated: 2026-08-20
scope: supplied Pump It Up training and test data
---

# Pump It Up data audit: findings and modelling decisions

This report records the conclusions from the raw-data audit before model building. The predictor notebooks hold the executable evidence; this document explains what that evidence means for preparation, validation and the first baseline.

The three structural removals are settled for the supplied schema. All other feature treatments remain provisional until they earn their place on a frozen training and validation split.

## Contents

- [Executive summary](#executive-summary)
- [Audit scope](#audit-scope)
- [Structural missingness](#structural-missingness)
- [Target label](#target-label)
- [Structural removals](#structural-removals)
- [Cross-cutting findings](#cross-cutting-findings)
- [Predictor decision register](#predictor-decision-register)
- [Highest-priority relationship checks](#highest-priority-relationship-checks)
- [Modelling consequences](#modelling-consequences)
- [Decisions deferred until after the split](#decisions-deferred-until-after-the-split)
- [Evidence map](#evidence-map)

## Executive summary

- The supplied data contains 59,400 labelled training rows, 14,850 test rows, an identifier and 39 raw predictors. Training and test use the same predictor schema.
- The target contains three complete classes. An always-`functional` prediction reaches 54.31% accuracy, while the majority class is 7.47 times the size of the smallest class.
- Three predictors have direct structural evidence for removal: `recorded_by` is constant, `payment` is a fixed relabelling of `payment_type`, and `quantity_group` duplicates `quantity` row for row.
- Source blanks occupy 1.94% of training predictor cells and affect 53.18% of training rows. Test rates are almost identical, so there is no material structural-missingness shift between the supplied frames.
- Many apparent values encode missingness or collection state. Examples include zero coordinates, zero construction year, blank booleans and literal tokens such as `None`, `unknown` and `0`. The preparation pipeline must preserve those states until training-partition evidence supports a treatment.
- Geography carries strong signal and strong leakage risk. Random validation may reward local memorisation, so the project needs an LGA or region-grouped sensitivity check alongside the main stratified split.
- Several categorical families form deterministic hierarchies. The baseline should choose a sensible granular representation, then compare coarser alternatives by ablation instead of retaining every level automatically.
- Raw names such as `wpt_name`, `scheme_name` and `subvillage` are too sparse for unrestricted one-hot encoding. Any frequency, hashing, text or target-based treatment must fit inside each training fold.

## Audit scope

| Item | Result |
|---|---:|
| Training predictor rows | 59,400 |
| Training label rows | 59,400 |
| Test predictor rows | 14,850 |
| Raw columns in each predictor file | 40 |
| Raw predictors, excluding `id` | 39 |
| Candidate predictors after structural removals | 36 |
| Structural removals | 3 |
| Target classes | 3 |
| Generated predictor notebooks | 117 |

The catalogue covers every raw non-identifier predictor:

| Audit type | Predictors |
|---|---:|
| Categorical | 22 |
| High-cardinality categorical | 6 |
| Numeric | 4 |
| Binary | 2 |
| Coordinate | 2 |
| Date | 1 |
| Year | 1 |
| Constant | 1 |

Each predictor has a basic breakdown, a noteworthy-findings notebook and a related-features notebook. The [predictor index](../README.md) links to all 39 folders.

## Structural missingness

The raw CSVs are loaded with `keep_default_na=False` so that blank source cells remain distinguishable from pandas nulls. Structural missingness combines those two representations but deliberately excludes semantic sentinels such as numeric zero and literal `unknown` values.

| Frame | Cells | Structural missing cells | Missing cells (%) | Rows affected | Affected rows (%) | Mean missing cells per affected row | Maximum in one row |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 2,376,000 | 46,094 | 1.94% | 31,587 | 53.18% | 1.459 | 6 |
| Test | 594,000 | 11,464 | 1.93% | 7,906 | 53.24% | 1.450 | 6 |

The affected-row mean is the interpretable form of “total missing cells divided by rows containing missing values”. It is useful only alongside the counts and percentages above: a low cell rate can still touch many rows.

Seven source columns contain structural missingness:

| Predictor | Training | Test | Test minus training |
|---|---:|---:|---:|
| `scheme_name` | 47.42% | 47.76% | +0.34 pp |
| `scheme_management` | 6.53% | 6.53% | 0.00 pp |
| `installer` | 6.15% | 5.91% | -0.24 pp |
| `funder` | 6.12% | 5.85% | -0.27 pp |
| `public_meeting` | 5.61% | 5.53% | -0.08 pp |
| `permit` | 5.14% | 4.96% | -0.18 pp |
| `subvillage` | 0.62% | 0.67% | +0.05 pp |

The close training/test rates are reassuring, but they do not settle how any feature should be treated. The focused audits separately examine blank-state meaning, numeric sentinels, literal placeholder tokens and invalid feature combinations. The [structural missingness audit](supporting-audits/05-structural-missingness-audit.ipynb) contains the executable evidence and uses the reusable summaries in [`predictor_audit.py`](../../../src/predictor_audit.py).

## Target label

The target audit found one complete label for every training identifier and exactly the three documented classes:

| Target label | Rows | Share |
|---|---:|---:|
| `functional` | 32,259 | 54.31% |
| `non functional` | 22,824 | 38.42% |
| `functional needs repair` | 4,317 | 7.27% |

The class balance sets four requirements for the baseline:

1. Create a reproducible stratified split before inspecting feature-to-target relationships further.
2. Use 54.31% accuracy as the non-model reference.
3. Report the confusion matrix and per-class recall alongside overall accuracy, with particular attention to `functional needs repair`.
4. Fit any resampling, class weighting or target-informed transformation inside the training partition or current cross-validation fold.

The [target-label analysis](../00-target-label-analysis/00-target-label-analysis.ipynb) contains the executable checks and distribution plot.

## Structural removals

Only three columns have evidence strong enough for removal before model comparison.

| Remove | Evidence | Retain | Safety condition |
|---|---|---|---|
| `recorded_by` | Every training and test row contains `GeoData Consultants Ltd`. | Nothing; the field has no variance. | Reject a future source that introduces another value until the policy is reviewed. |
| `payment` | It is a one-to-one verbose relabelling of `payment_type` in both supplied predictor files. | `payment_type` | Validate the fixed mapping against each raw input. |
| `quantity_group` | It equals `quantity` row for row, including the missing-value pattern, in both supplied predictor files. | `quantity` | Validate equality before removing the duplicate. |

The payment mapping is stable across training and test data:

| `payment` | `payment_type` |
|---|---|
| `pay annually` | `annually` |
| `pay monthly` | `monthly` |
| `pay per bucket` | `per bucket` |
| `pay when scheme fails` | `on failure` |
| `never pay` | `never pay` |
| `other` | `other` |
| `unknown` | `unknown` |

The fixed removal policy reduces the raw predictor frame from 40 columns to 37, including `id`. The modelling workflow then separates `id` as metadata and works with 36 candidate predictors. [`raw_feature_column_policy.json`](../../../src/raw_feature_column_policy.json) records the schema and [`data_preparation.py`](../../../src/data_preparation.py) applies the guarded removal.

## Cross-cutting findings

### Placeholder values carry information

The source uses several representations for unavailable, unknown or unrecorded values. They do not all mean the same thing.

- Numeric zero marks likely unavailability in `amount_tsh`, `gps_height`, `population`, `construction_year` and the coordinate pair, but a genuine zero remains possible for some fields.
- Blank values in `public_meeting` and `permit` behave as a third state rather than a safe substitute for `false`.
- Literal strings such as `None`, `unknown`, `not known` and `0` coexist with source blanks in organisation and scheme fields.

The first pipeline should create explicit availability or sentinel states where the evidence supports them. Imputation must fit inside each training fold.

### Geography is predictive and easy to overuse

Coordinates, basin, region, LGA, ward and subvillage describe overlapping views of location. Region-level functional rates differ substantially, all 125 training LGAs appear in the test data, and local names can act as geographic identifiers.

Keep interpretable geographic back-offs such as region and LGA. Treat coordinates as one validated pair. Compare the main stratified split with an LGA or region-grouped sensitivity check so local memorisation does not masquerade as generalisation.

### High-cardinality names need fold-safe treatment

`wpt_name`, `subvillage`, `scheme_name`, `funder` and `installer` contain sparse levels, inconsistent spelling, sentinel tokens or unseen test values. Raw one-hot encoding would create a large, fragile feature space.

Apply only conservative case and whitespace normalisation at first. Keep unseen and rare values explicit. Test frequency, hashing or text treatments separately, and fit every learned mapping inside the current training fold.

### Categorical hierarchies need ablation

Five families contain deterministic or near-deterministic levels of granularity:

| Family | Granular representation | Coarser alternatives | Initial position |
|---|---|---|---|
| Extraction | `extraction_type` | `extraction_type_group`, `extraction_type_class` | Start granular; compare one coarse alternative. |
| Management | `management` | `management_group` | Start with `management`; test whether the group generalises better. |
| Water quality | `water_quality` | `quality_group` | Start granular; retain the group for ablation. |
| Source | `source` | `source_type`, `source_class` | Start with `source`; test `source_type` as a lower-cardinality alternative. |
| Waterpoint form | `waterpoint_type` | `waterpoint_type_group` | Start granular; compare the grouped form. |

Keeping every hierarchy level would add deterministic redundancy. Removing every coarse level now would also be premature because simpler groupings may handle rare categories better.

### Collection and administration can proxy the outcome

`date_recorded`, geographic fields, funder, installer, permit and management variables partly describe how and where the survey operated. They may predict the target without representing a transferable physical mechanism.

Keep them as candidates, but interpret their importance cautiously. Grouped geographic checks and time-aware inspection will show whether performance depends on survey operations.

### Numeric fields need state plus magnitude

`amount_tsh`, `gps_height`, `population` and `num_private` combine large sentinel spikes with skewed positive values. One numeric transform cannot represent both behaviours well.

Test explicit availability or special-value indicators alongside raw or `log1p` positive magnitude. Learn imputation and scaling after the split.

## Predictor decision register

The following tables bring every catalogue decision into one readable register. Markdown wraps each cell instead of replacing evidence and decisions with ellipses.

### Measurement, time and binary administration

| Predictor | Finding and provisional treatment | Main risk |
|---|---|---|
| `amount_tsh` | **Finding:** Zero dominates the supplied values and positive amounts are strongly right-skewed.<br>**Treatment:** Keep an amount-recorded indicator and compare raw or transformed positive magnitude inside validation. | Zero can mean no recorded amount rather than a genuine measured zero. |
| `date_recorded` | **Finding:** Every supplied date parses, but collection timing is concentrated in survey waves.<br>**Treatment:** Replace the raw date with `days_since_recorded` from the fixed 2015-02-02 competition-era reference; do not derive separate calendar components. | Recording time can proxy survey operations and geography rather than waterpoint condition. |
| `gps_height` | **Finding:** Zero affects roughly a third of rows and overlaps a broader missing-measurement block.<br>**Treatment:** Flag zero, impute it inside folds where necessary and preserve negative measured values initially. | Some zeros can be genuine low elevation and the feature is geographically structured. |
| `num_private` | **Finding:** The undocumented field is almost entirely zero with a small number of extreme positive values.<br>**Treatment:** Keep a non-zero flag and magnitude candidate, then test an early omission ablation. | Zero has no documented missing-value meaning and positive support is sparse. |
| `population` | **Finding:** Zero and one are large special spikes; positive values are strongly right-skewed.<br>**Treatment:** Keep zero and one flags plus a raw or `log1p` magnitude candidate. | Zero is probably often missing while one may be a recorded placeholder or real small population. |
| `public_meeting` | **Finding:** About 5.6% of training values are blank and blanks differ from false.<br>**Treatment:** Encode true, false and blank as three explicit states. | Missingness may reflect the survey process rather than pump condition. |
| `permit` | **Finding:** About 5.1% of training values are blank and blanks are independent of most `public_meeting` blanks.<br>**Treatment:** Encode three explicit states independently of `public_meeting`. | Permit status may proxy administration and geography. |
| `construction_year` | **Finding:** Year zero affects about a third of rows and a small number of derived ages are negative.<br>**Treatment:** Keep unknown and inconsistent flags plus valid pump age or construction cohort. | Age is missing for many rows and is confounded with technology and geography. |

### Geography and location

| Predictor | Finding and provisional treatment | Main risk |
|---|---|---|
| `longitude` | **Finding:** Longitude zero participates in the paired missing-location sentinel.<br>**Treatment:** Create one coordinate-availability flag and transform longitude and latitude together. | Using either coordinate independently breaks location meaning and encourages spatial memorisation. |
| `latitude` | **Finding:** Latitude near zero participates in the paired missing-location sentinel.<br>**Treatment:** Create one coordinate-availability flag and transform latitude and longitude together. | Using either coordinate independently breaks location meaning and encourages spatial memorisation. |
| `basin` | **Finding:** Nine complete levels have stable train/test coverage and meaningful target differences.<br>**Treatment:** Retain as a categorical feature and compare its contribution with administrative geography. | Hydrological and administrative geography overlap without forming a strict hierarchy. |
| `subvillage` | **Finding:** The field is extremely sparse, has reused names and exposes many test rows to unseen levels.<br>**Treatment:** Use only a separately validated hashing or frequency treatment with explicit missingness. | Direct encoding encourages local memorisation and poor unseen coverage. |
| `region` | **Finding:** All 21 levels are covered and functional rates differ substantially by region.<br>**Treatment:** Retain and add an LGA or region-grouped validation sensitivity check. | Random validation can reward geographic memorisation. |
| `region_code` | **Finding:** The numeric-looking code has categorical meaning and is not a one-to-one copy of region.<br>**Treatment:** Treat as unordered and compare region, code and their combination by validation. | Scaling or distance-based treatment would impose a false ordering. |
| `district_code` | **Finding:** The integer code is reused across regions and zero is not universally missing.<br>**Treatment:** Treat as categorical and prefer a region-code and district-code composite if retained. | The same raw code does not identify one national district. |
| `lga` | **Finding:** All 125 training levels are covered in test and each maps to one region.<br>**Treatment:** Retain and compare with the coarser region representation. | Strong geographic target differences require grouped robustness checks. |
| `ward` | **Finding:** Raw unseen exposure is small, but many ward names are reused and low-frequency.<br>**Treatment:** Prefer an LGA and ward composite or an encoder with explicit rare and unseen handling. | Raw ward names can be ambiguous outside their administrative context. |

### Organisations, names and governance

| Predictor | Finding and provisional treatment | Main risk |
|---|---|---|
| `funder` | **Finding:** Effective missingness is about 7.5% and a small but material test share uses unseen funders.<br>**Treatment:** Keep blank and sentinel states distinct, normalise conservatively and handle unseen levels explicitly. | Naive target encoding leaks and unrestricted fuzzy merging can combine different organisations. |
| `installer` | **Finding:** The field contains case and whitespace fragmentation as well as blank and sentinel values.<br>**Treatment:** Normalise case and whitespace, retain separately from funder and map unseen values explicitly. | Aliases need a reviewed mapping; fuzzy merging can erase meaningful distinctions. |
| `wpt_name` | **Finding:** Most levels are sparse and more than half of test rows use unseen exact names.<br>**Treatment:** Exclude raw one-hot names initially; test cross-fitted frequency or text features separately. | In-sample lookup performance is dominated by memorisation and geographic proxying. |
| `recorded_by` | **Finding:** Every supplied row contains the same recording organisation.<br>**Treatment:** Remove before modelling. | Revalidate the assumption if a future source introduces another recorder. |
| `scheme_management` | **Finding:** The field has source blanks and strongly overlaps `management` without being equivalent.<br>**Treatment:** Keep blank explicit and compare its incremental value with `management`. | Parallel management fields may mostly add redundancy. |
| `scheme_name` | **Finding:** Nearly half the source values are blank and the names are high-cardinality and inconsistent with `scheme_management`.<br>**Treatment:** Preserve blank and literal sentinels separately; test only fold-fitted high-cardinality treatments. | Scheme identity can memorise projects and geography. |
| `management` | **Finding:** Twelve complete levels map deterministically to five management groups.<br>**Treatment:** Use as the initial management feature and compare `scheme_management` incrementally. | Some sparse levels may require grouping. |
| `management_group` | **Finding:** Five coarse groups are deterministically derived from `management` and retain much less descriptive association.<br>**Treatment:** Prefer `management` and keep this as a coarse ablation candidate. | The coarse feature may generalise better for rare management levels. |

### Infrastructure, service and categorical hierarchies

| Predictor | Finding and provisional treatment | Main risk |
|---|---|---|
| `extraction_type` | **Finding:** Eighteen levels map deterministically to two coarser hierarchy levels.<br>**Treatment:** Start with the granular type and compare a coarse alternative by ablation. | Keeping every hierarchy level adds deterministic redundancy. |
| `extraction_type_group` | **Finding:** The intermediate group is deterministic from `extraction_type` and retains nearly the same descriptive association.<br>**Treatment:** Compare it against the granular type; do not keep both automatically. | The coarser grouping may generalise better even though it loses detail. |
| `extraction_type_class` | **Finding:** Seven broad classes are deterministically derived from the finer extraction hierarchy.<br>**Treatment:** Use as a compact alternative or back-off, not an automatic extra feature. | Coarsening can hide useful method-level differences. |
| `payment` | **Finding:** The field is a fixed verbose relabelling of `payment_type` in both supplied feature sets.<br>**Treatment:** Remove `payment` and retain `payment_type`. | Revalidate the fixed mapping against any future source schema. |
| `payment_type` | **Finding:** Seven fully covered levels include informative `unknown` and `other` states.<br>**Treatment:** Retain as nominal and do not reintroduce the removed `payment` alias. | Payment patterns may proxy local administration and income. |
| `water_quality` | **Finding:** Eight levels map deterministically to `quality_group` and preserve important within-group differences.<br>**Treatment:** Retain granular quality with rare handling and compare the coarse parent by ablation. | The unknown state may reflect data quality as much as water quality. |
| `quality_group` | **Finding:** Six coarse groups are deterministic from `water_quality` and hide useful granular distinctions.<br>**Treatment:** Prefer `water_quality` and keep this only as a coarse ablation candidate. | The coarser feature may generalise better for rare quality values. |
| `quantity` | **Finding:** Five fully covered nominal levels include a strong `dry` state and an explicit `unknown` state.<br>**Treatment:** Retain as nominal and do not reintroduce the duplicate `quantity_group`. | The strong dry association is predictive but should not be interpreted causally. |
| `quantity_group` | **Finding:** The field is an exact duplicate of `quantity` in every supplied training and test row.<br>**Treatment:** Remove `quantity_group` and retain `quantity`. | Revalidate equality if a future source schema changes. |
| `source` | **Finding:** Ten fully covered levels map deterministically upward and preserve differences hidden by broader types.<br>**Treatment:** Retain `source` and compare `source_type` as a lower-cardinality alternative. | Keeping every source hierarchy level adds deterministic redundancy. |
| `source_type` | **Finding:** Seven intermediate types are deterministic from `source` and remove useful within-type detail.<br>**Treatment:** Use only as an alternative to `source`, not automatically alongside it. | The coarser representation may generalise better despite losing detail. |
| `source_class` | **Finding:** Three broad classes are deterministic from the finer source hierarchy and have weak descriptive association.<br>**Treatment:** Omit from the first granular baseline and retain as a coarse ablation. | Coarsening may help simple models but discards substantial source detail. |
| `waterpoint_type` | **Finding:** Seven levels map deterministically to a coarser group and preserve standpipe distinctions.<br>**Treatment:** Retain with infrequent handling and compare the coarse group by ablation. | The rare `dam` level is too sparse for a stable standalone interpretation. |
| `waterpoint_type_group` | **Finding:** Six coarse groups are deterministic from `waterpoint_type` and hide single-versus-multiple standpipe differences.<br>**Treatment:** Prefer `waterpoint_type` and keep this as a coarse ablation candidate. | The grouped feature may generalise better for rare granular values. |

## Highest-priority relationship checks

| Relationship | Question for validation | Planned comparison |
|---|---|---|
| `longitude` and `latitude` | Does the pair add value beyond named geography without overfitting local coordinates? | Coordinates available or unavailable; raw or transformed pair; named geography only. |
| `date_recorded` and `construction_year` | Does valid pump age add value beyond elapsed recording time? | `days_since_recorded`, valid pump age, unknown-age flags and an elapsed-date ablation. |
| `funder` and `installer` | Does each organisation add independent signal after conservative normalisation? | Each alone, both together and rare-grouped variants. |
| `region`, `lga` and `ward` | Which level balances signal, interpretability and unseen handling? | Coarse, granular and grouped-validation results. |
| `management`, `management_group` and `scheme_management` | Does the parallel scheme field add value beyond the main management label? | Granular management, coarse management and incremental scheme management. |
| Extraction, source and waterpoint families | Do several physical hierarchies add complementary information or redundant category labels? | One preferred level per family, then targeted coarse or combined ablations. |

## Modelling consequences

The audit supports this order of work:

```mermaid
flowchart LR
    A["Immutable source CSVs"] --> B["Validate schema and identifiers"]
    B --> C["Remove three structural columns"]
    C --> D["Separate IDs and target"]
    D --> E["Create stratified split"]
    E --> F["Fit preprocessing within folds"]
    F --> G["Compare baseline models"]
    G --> H["Check errors and geographic robustness"]
```

1. Reload the immutable source CSVs and validate their schema and identifiers.
2. Apply the three guarded structural removals independently to training and test predictors.
3. Separate `id` as metadata and align `status_group` by identifier.
4. Create the reproducible stratified split.
5. Fit missing-value handling, category grouping, encoding and scaling inside the training partition or current cross-validation fold.
6. Establish the majority-class reference, then compare a transparent baseline, a decision tree and a tree ensemble.
7. Inspect overall accuracy, per-class recall and the confusion matrix.
8. Run geographic sensitivity checks and targeted feature-family ablations before accepting a final feature set.

No cleaned CSV should become a second source of truth. Each run should recreate the modelling frame from the untouched competition files.

## Decisions deferred until after the split

- Alternative missing-value representations and imputation strategies.
- Rare-category thresholds and treatment of unseen values.
- Frequency, hashing, text or target-informed treatment for high-cardinality fields.
- Geographic feature engineering and ablation of the elapsed-date feature.
- Choice between granular and coarse categorical hierarchy levels.
- Removal of weak candidates such as `num_private`.
- Model selection, class weighting, resampling and tuning ranges.

These decisions need training-partition or cross-validation evidence. The structural removals do not.

## Evidence map

- [Data inventory and checksums](../../../data/README.md)
- [Structural missingness audit](supporting-audits/05-structural-missingness-audit.ipynb)
- [Target-label analysis](../00-target-label-analysis/00-target-label-analysis.ipynb)
- [Predictor audit index](../README.md)
- [Predictor catalogue](../../../src/predictor_audit_catalogue.json)
- [Structural column policy](../../../src/raw_feature_column_policy.json)
- [Data-preparation decisions and next steps](../../../reports/data-preparation-next-steps.md)
- [Prior executable overall audit](supporting-audits/10-prior-overall-data-audit.ipynb)
- [Feature-family audit notebooks](feature-family-audits/)

## Maintaining this report

This Markdown file is the maintained findings report. `generate_data_audit_notebooks.py` rebuilds the 117 per-predictor notebooks and their index, but it does not overwrite this narrative. Update the report when new executed evidence changes a finding, decision or risk in the predictor catalogue.
