# Pump It Up: competition reference

The public problem description and rules were checked on 15 August 2026. The
local download inventory and source-page snapshots date from 29 July 2026.

## Problem

Predict the operating condition of waterpoints in Tanzania from information
collected by Taarifa and supplied by the Tanzanian Ministry of Water. This is a
multiclass tabular classification problem with three target values:

- `functional`
- `functional needs repair`
- `non functional`

The practical aim is to help identify which waterpoints work, which need
repair, and which do not work.

## Files and submission

The competition data is available only after joining. The expected submission
has one row per test-set waterpoint and two columns:

```text
id,status_group
```

The downloaded filenames, row counts and checksums are recorded in
`../data/README.md`.

The primary evaluation metric is **classification rate**: the proportion of
test cases assigned the correct class. In other words, this competition is
ranked by multiclass accuracy. The account currently has a limit of three
submissions per UTC day.

## Feature dictionary

The public problem page describes 39 predictor columns:

| Feature | Meaning |
|---|---|
| `amount_tsh` | Total static head available to the waterpoint |
| `date_recorded` | Date the row was entered |
| `funder` | Organisation funding the well |
| `gps_height` | Altitude of the well |
| `installer` | Organisation that installed the well |
| `longitude`, `latitude` | Waterpoint coordinates |
| `wpt_name` | Waterpoint name, if present |
| `num_private` | Undocumented on the public page |
| `basin` | Geographic water basin |
| `subvillage` | Geographic location |
| `region`, `region_code` | Geographic location and encoded region |
| `district_code` | Encoded geographic district |
| `lga` | Local government area |
| `ward` | Geographic ward |
| `population` | Population around the well |
| `public_meeting` | Whether a public meeting was held |
| `recorded_by` | Organisation entering the row |
| `scheme_management`, `scheme_name` | Management organisation and scheme |
| `permit` | Whether the waterpoint is permitted |
| `construction_year` | Year the waterpoint was constructed |
| `extraction_type`, `extraction_type_group`, `extraction_type_class` | Extraction method at three levels of grouping |
| `management`, `management_group` | Management method and its higher-level group |
| `payment`, `payment_type` | Payment arrangements in two related encodings |
| `water_quality`, `quality_group` | Water quality and its higher-level group |
| `quantity`, `quantity_group` | Water quantity and its higher-level group |
| `source`, `source_type`, `source_class` | Water source at three levels of grouping |
| `waterpoint_type`, `waterpoint_type_group` | Waterpoint type and its higher-level group |

The feature table is a working paraphrase. Use the live data dictionary if its
wording or schema differs from downloaded files.

## Rules relevant to the project

- Joining means agreeing to DrivenData's competition rules; this must be done by
  the account holder.
- The public rules restrict data use to the competition unless the competition
  site grants an exception. They also require participants to prevent access by
  people who have not accepted the rules.
- Treat the supplied data as competition-restricted. Do not publish or commit
  the downloaded files.
- The public rules prohibit external data for this competition.
- Winning solutions must be made available under the MIT licence to qualify for
  recognition or any prize offered.
- Check the live rules before publishing code, forming a team or submitting.

Suggested citation:

> DrivenData. (2015). *Pump it Up: Data Mining the Water Table.* Retrieved
> 29 July 2026 from
> https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table.

## Authoritative links

- [Competition home](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)
- [About Taarifa](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/24/)
- [Problem description and data dictionary](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/)
- [Rules](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/rules/)
- [Leaderboard](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/leaderboard/)
- [DrivenData community category](https://community.drivendata.org/c/pump-it-up-data-mining-the-water-table/11)
