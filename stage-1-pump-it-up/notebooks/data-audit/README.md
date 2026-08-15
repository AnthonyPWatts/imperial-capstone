# Pump It Up data audit

This directory is the canonical home of the data audit.

Start with the role-based project-level analyses:

1. [`00-target-label-analysis/`](00-target-label-analysis/) contains the dedicated target-label analysis. The folder and notebook use the generic modelling role rather than the dataset-specific `status_group` column name.
2. [`00-overall-data-audit.md`](00-overall/00-overall-data-audit.md) is the maintained findings report. Its folder also preserves supporting executable and feature-family audits.

Every raw non-`id` predictor then has a CSV-order-prefixed folder.

Each predictor folder contains:

1. `01-<feature>-basic-breakdown.ipynb` — the analysis always applied to that feature type.
2. `02-<feature>-noteworthy-findings.ipynb` — supported observations, interpretation and decision.
3. `03-<feature>-related-features.ipynb` — nominated relationships and pairwise diagnostics.

The three structurally removed columns are intentionally included: this is where
the evidence for removal is recorded before preprocessing.

## Predictor index

| # | Predictor | Type | Role | Folder |
|---:|---|---|---|---|
| 1 | `amount_tsh` | `numeric` | `candidate` | [`01-amount_tsh/`](01-amount_tsh/) |
| 2 | `date_recorded` | `date` | `candidate` | [`02-date_recorded/`](02-date_recorded/) |
| 3 | `funder` | `high-cardinality-category` | `candidate` | [`03-funder/`](03-funder/) |
| 4 | `gps_height` | `numeric` | `candidate` | [`04-gps_height/`](04-gps_height/) |
| 5 | `installer` | `high-cardinality-category` | `candidate` | [`05-installer/`](05-installer/) |
| 6 | `longitude` | `coordinate` | `candidate` | [`06-longitude/`](06-longitude/) |
| 7 | `latitude` | `coordinate` | `candidate` | [`07-latitude/`](07-latitude/) |
| 8 | `wpt_name` | `high-cardinality-category` | `candidate` | [`08-wpt_name/`](08-wpt_name/) |
| 9 | `num_private` | `numeric` | `candidate` | [`09-num_private/`](09-num_private/) |
| 10 | `basin` | `category` | `candidate` | [`10-basin/`](10-basin/) |
| 11 | `subvillage` | `high-cardinality-category` | `candidate` | [`11-subvillage/`](11-subvillage/) |
| 12 | `region` | `category` | `candidate` | [`12-region/`](12-region/) |
| 13 | `region_code` | `category` | `candidate` | [`13-region_code/`](13-region_code/) |
| 14 | `district_code` | `category` | `candidate` | [`14-district_code/`](14-district_code/) |
| 15 | `lga` | `category` | `candidate` | [`15-lga/`](15-lga/) |
| 16 | `ward` | `high-cardinality-category` | `candidate` | [`16-ward/`](16-ward/) |
| 17 | `population` | `numeric` | `candidate` | [`17-population/`](17-population/) |
| 18 | `public_meeting` | `binary` | `candidate` | [`18-public_meeting/`](18-public_meeting/) |
| 19 | `recorded_by` | `constant` | `structural-removal` | [`19-recorded_by/`](19-recorded_by/) |
| 20 | `scheme_management` | `category` | `candidate` | [`20-scheme_management/`](20-scheme_management/) |
| 21 | `scheme_name` | `high-cardinality-category` | `candidate` | [`21-scheme_name/`](21-scheme_name/) |
| 22 | `permit` | `binary` | `candidate` | [`22-permit/`](22-permit/) |
| 23 | `construction_year` | `year` | `candidate` | [`23-construction_year/`](23-construction_year/) |
| 24 | `extraction_type` | `category` | `candidate` | [`24-extraction_type/`](24-extraction_type/) |
| 25 | `extraction_type_group` | `category` | `candidate` | [`25-extraction_type_group/`](25-extraction_type_group/) |
| 26 | `extraction_type_class` | `category` | `candidate` | [`26-extraction_type_class/`](26-extraction_type_class/) |
| 27 | `management` | `category` | `candidate` | [`27-management/`](27-management/) |
| 28 | `management_group` | `category` | `candidate` | [`28-management_group/`](28-management_group/) |
| 29 | `payment` | `category` | `structural-removal` | [`29-payment/`](29-payment/) |
| 30 | `payment_type` | `category` | `candidate` | [`30-payment_type/`](30-payment_type/) |
| 31 | `water_quality` | `category` | `candidate` | [`31-water_quality/`](31-water_quality/) |
| 32 | `quality_group` | `category` | `candidate` | [`32-quality_group/`](32-quality_group/) |
| 33 | `quantity` | `category` | `candidate` | [`33-quantity/`](33-quantity/) |
| 34 | `quantity_group` | `category` | `structural-removal` | [`34-quantity_group/`](34-quantity_group/) |
| 35 | `source` | `category` | `candidate` | [`35-source/`](35-source/) |
| 36 | `source_type` | `category` | `candidate` | [`36-source_type/`](36-source_type/) |
| 37 | `source_class` | `category` | `candidate` | [`37-source_class/`](37-source_class/) |
| 38 | `waterpoint_type` | `category` | `candidate` | [`38-waterpoint_type/`](38-waterpoint_type/) |
| 39 | `waterpoint_type_group` | `category` | `candidate` | [`39-waterpoint_type_group/`](39-waterpoint_type_group/) |

## Rebuild

Run `generate_data_audit_notebooks.py` from the project environment, then
execute the generated predictor notebooks before treating their recorded outputs as current.
The generator rebuilds the 117 predictor notebooks and this index; it does not
overwrite the maintained overall findings report.
The catalogue in `../../src/predictor_audit_catalogue.json` is the single source
of truth for ordering, audit types, dispositions and related-feature selections.
