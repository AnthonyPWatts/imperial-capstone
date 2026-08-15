"""Build the focused predictor-audit notebooks from one consistent template."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parent
KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.14.5"},
}


def markdown(source: str, cell_id: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n", id=cell_id)


def code(source: str, cell_id: str):
    return nbformat.v4.new_code_cell(dedent(source).strip() + "\n", id=cell_id)


def common_cells(title: str, scope: str, features: list[str], review_type: str):
    feature_list = ", ".join(f"`{feature}`" for feature in features)
    return [
        markdown(
            f"""
            # {title}

            This focused notebook completes the initial **{review_type}** review of
            {feature_list}. It describes the supplied training and test predictors,
            then uses the labelled training rows to identify relationships worth
            carrying into a leakage-safe modelling pipeline.

            {scope}

            This is exploratory evidence, not fitted preprocessing. Category pooling,
            imputation, encoding and scaling must be learned inside each training fold.
            """,
            "audit-intro",
        ),
        markdown(
            """
            ## Consistent audit contract

            Every focused predictor audit answers the same questions before adding
            type-specific checks:

            1. What is explicitly missing, and what looks like a sentinel?
            2. What range or category coverage is present in training and test?
            3. How much of the test set is exposed to unseen training levels?
            4. Does the labelled distribution vary enough to justify retaining the field?
            5. What exact baseline treatment follows from the evidence?

            Target-rate tables flag support rather than treating tiny groups as reliable.
            Train/test comparisons are descriptive and do not use the hidden test labels.
            """,
            "audit-contract",
        ),
        code(
            f"""
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            source_directory = str(Path("../src").resolve())
            if source_directory not in sys.path:
                sys.path.insert(0, source_directory)

            from predictor_audit import (
                MISSING_CATEGORY,
                analysis_categories,
                categorical_summary,
                categorical_target_profile,
                category_frequency_table,
                cramer_v,
                hierarchy_conflicts,
                hierarchy_summary,
                numeric_summary,
                numeric_target_summary,
                normalise_categories,
                sentinel_mask,
                source_blank_mask,
                text_normalisation_summary,
            )
            from source_data_validation import (
                validate_aligned_ids,
                validate_label_frame,
                validate_raw_feature_schema,
            )

            data_directory = Path("../data")
            training_features = pd.read_csv(
                data_directory / "TrainingSetValues.csv",
                keep_default_na=False,
            )
            training_labels = pd.read_csv(
                data_directory / "TrainingSetLabels.csv",
                keep_default_na=False,
            )
            test_features = pd.read_csv(
                data_directory / "TestSetValues.csv",
                keep_default_na=False,
            )

            validate_raw_feature_schema(training_features)
            validate_raw_feature_schema(test_features)
            validate_label_frame(training_labels)
            validate_aligned_ids(training_features, training_labels)

            training_data = training_features.merge(
                training_labels,
                on="id",
                validate="one_to_one",
            )
            audited_features = {features!r}
            assert set(audited_features).issubset(training_features.columns)

            pd.set_option("display.max_columns", 30)
            pd.set_option("display.max_colwidth", 80)
            print(
                f"Validated {{len(training_features):,}} training rows and "
                f"{{len(test_features):,}} test rows for {{len(audited_features)}} predictors."
            )
            """,
            "load-and-validate",
        ),
    ]


def decision_cells(decisions: list[dict[str, str]], caveat: str):
    return [
        markdown(
            """
            ## Decision register

            The register separates observed evidence from the proposed baseline action.
            A retained field is still a candidate: later validation must show whether it
            improves generalisation and whether a coarser related representation is safer.
            """,
            "decision-heading",
        ),
        code(
            f"""
            decision_register = pd.DataFrame({decisions!r})
            display(decision_register.set_index("feature"))
            """,
            "decision-register",
        ),
        markdown(
            f"""
            ### Handoff to modelling

            {caveat}

            - Preserve raw source frames and implement the stated sentinel rules on copies.
            - Fit imputers, rare-level grouping and encoders on each training fold only.
            - Map unseen validation or test categories to an explicit fallback.
            - Compare the stated baseline treatment with a simple omission ablation.
            - Revisit target-rate observations after the reproducible stratified split exists.
            """,
            "modelling-handoff",
        ),
    ]


NUMERIC_DECISIONS = [
    {
        "feature": "gps_height",
        "quality finding": "Zero affects 34.41% of training rows; 1,496 negative heights are stable in test.",
        "baseline treatment": "Flag zero, impute it inside each fold, and preserve negative values initially.",
        "risk to verify": "Some zeros may be genuine; elevation is geographically structured.",
    },
    {
        "feature": "num_private",
        "quality finding": "The field is undocumented and 98.73% zero; only 757 training rows are positive.",
        "baseline treatment": "Keep a non-zero flag plus magnitude candidate, with an early omission ablation.",
        "risk to verify": "Zero has no documented missing-value meaning.",
    },
    {
        "feature": "population",
        "quality finding": "Zero affects 36.00%; value one is a separate 11.83% spike with 53.52% non-functional.",
        "baseline treatment": "Keep zero and one flags plus log1p magnitude; do not merge the two special states.",
        "risk to verify": "Local population estimates may be noisy and geographically dependent.",
    },
]


def build_numeric_notebook():
    features = ["gps_height", "num_private", "population"]
    cells = common_cells(
        "Numeric predictors: scale, sentinels and stability",
        "The audit keeps true numeric magnitude separate from encoded identifiers such as region codes.",
        features,
        "numeric",
    )
    cells += [
        markdown(
            """
            ## 1. Distribution and sentinel audit

            The zero counts are shown for every field. Domain meaning determines whether
            zero is missing: it is treated as a sentinel for elevation and population,
            but remains merely suspicious for the undocumented `num_private` field.
            """,
            "numeric-summary-heading",
        ),
        code(
            """
            numeric_sentinels = {
                "gps_height": [0],
                "num_private": [],
                "population": [0],
            }
            display(
                numeric_summary(
                    training_features,
                    test_features,
                    audited_features,
                    sentinel_values_by_column=numeric_sentinels,
                )
            )
            """,
            "numeric-summary",
        ),
        code(
            """
            shared_zero = (
                training_features["gps_height"].eq(0)
                & training_features["population"].eq(0)
                & training_features["construction_year"].eq(0)
            )
            shared_zero_test = (
                test_features["gps_height"].eq(0)
                & test_features["population"].eq(0)
                & test_features["construction_year"].eq(0)
            )
            shared_measurement_state = pd.DataFrame({
                "training rows": [shared_zero.sum()],
                "training (%)": [shared_zero.mean() * 100],
                "test rows": [shared_zero_test.sum()],
                "test (%)": [shared_zero_test.mean() * 100],
                "height-zero rows also population-zero (%)": [
                    training_features.loc[training_features["gps_height"].eq(0), "population"].eq(0).mean() * 100
                ],
                "height-zero rows also year-zero (%)": [
                    training_features.loc[training_features["gps_height"].eq(0), "construction_year"].eq(0).mean() * 100
                ],
            }, index=["shared zero-measurement block"]).round(3)
            display(shared_measurement_state)
            """,
            "shared-zero-state",
        ),
        code(
            """
            fig, axes = plt.subplots(len(audited_features), 2, figsize=(12, 9))
            for row, feature in enumerate(audited_features):
                training_values = pd.to_numeric(training_features[feature], errors="coerce")
                test_values = pd.to_numeric(test_features[feature], errors="coerce")
                upper = training_values.quantile(0.99)
                for column, (label, values, colour) in enumerate(
                    (("training", training_values, "#2b6cb0"), ("test", test_values, "#dd6b20"))
                ):
                    axes[row, column].hist(values.clip(upper=upper), bins=35, color=colour, alpha=0.8)
                    axes[row, column].set_title(f"{feature}: {label} (capped at training p99)")
                    axes[row, column].set_xlabel(feature)
                    axes[row, column].set_ylabel("rows")
            plt.tight_layout()
            plt.show()
            """,
            "numeric-plots",
        ),
        markdown(
            """
            ## 2. Relationship with `status_group`

            Medians and p90 values compare the numeric magnitude by class. The second
            table isolates zero from positive values so a sentinel's apparent signal is
            not mistaken for an ordinary numeric relationship.
            """,
            "numeric-target-heading",
        ),
        code(
            """
            for feature in audited_features:
                print()
                print(feature)
                display(
                    numeric_target_summary(
                        training_data,
                        feature,
                        sentinel_values=numeric_sentinels[feature],
                    )
                )
                zero_state = np.where(training_data[feature].eq(0), "zero", "non-zero")
                zero_profile = pd.crosstab(
                    zero_state,
                    training_data["status_group"],
                    normalize="index",
                ).mul(100).round(2)
                zero_profile.insert(0, "rows", pd.Series(zero_state).value_counts())
                display(zero_profile)
            """,
            "numeric-target",
        ),
        code(
            """
            state_definitions = {
                "gps_height state": np.select(
                    [training_data["gps_height"].lt(0), training_data["gps_height"].eq(0)],
                    ["negative", "zero"],
                    default="positive",
                ),
                "num_private state": np.where(training_data["num_private"].eq(0), "zero", "positive"),
                "population state": np.select(
                    [
                        training_data["population"].eq(0),
                        training_data["population"].eq(1),
                        training_data["population"].between(2, 10),
                    ],
                    ["zero", "one", "2-10"],
                    default="11+",
                ),
            }
            for state_name, states in state_definitions.items():
                state_table = pd.crosstab(
                    states,
                    training_data["status_group"],
                    normalize="index",
                ).mul(100).round(2)
                state_table.insert(0, "rows", pd.Series(states).value_counts())
                print()
                print(state_name)
                display(state_table)
            """,
            "numeric-special-states",
        ),
        markdown(
            """
            ## 3. Training/test stability

            The Kolmogorov-Smirnov statistic compares marginal numeric distributions.
            It is a drift flag, not a hypothesis test for feature usefulness. A second
            value excludes agreed sentinels so missingness drift and value drift remain visible.
            """,
            "numeric-drift-heading",
        ),
        code(
            """
            from scipy.stats import ks_2samp

            drift_records = []
            for feature in audited_features:
                training_values = pd.to_numeric(training_features[feature], errors="coerce").dropna()
                test_values = pd.to_numeric(test_features[feature], errors="coerce").dropna()
                all_values = ks_2samp(training_values, test_values)
                sentinels = set(numeric_sentinels[feature])
                training_observed = training_values.loc[~training_values.isin(sentinels)]
                test_observed = test_values.loc[~test_values.isin(sentinels)]
                observed_values = ks_2samp(training_observed, test_observed)
                drift_records.append({
                    "feature": feature,
                    "KS including sentinels": all_values.statistic,
                    "KS excluding agreed sentinels": observed_values.statistic,
                    "training sentinel (%)": training_values.isin(sentinels).mean() * 100,
                    "test sentinel (%)": test_values.isin(sentinels).mean() * 100,
                })
            display(pd.DataFrame(drift_records).set_index("feature").round(4))
            """,
            "numeric-drift",
        ),
    ]
    cells += decision_cells(
        NUMERIC_DECISIONS,
        "Keep zero-as-missing logic feature-specific. Do not globally replace numeric zeros.",
    )
    return notebook(cells)


TEMPORAL_DECISIONS = [
    {
        "feature": "date_recorded",
        "quality finding": "All dates parse; training spans 2002-10-14 to 2013-12-03 with strong survey-month differences.",
        "baseline treatment": "Derive recording year/month and elapsed days; do not one-hot the raw date string.",
        "risk to verify": "Collection timing can proxy geography or survey operations.",
    },
    {
        "feature": "construction_year",
        "quality finding": "Year zero affects 34.86%; valid pump age has median 13 and nine negative training cases.",
        "baseline treatment": "Keep unknown/inconsistent flags and valid pump age; compare construction cohort.",
        "risk to verify": "Imputation and age construction must be fitted or applied without validation leakage.",
    },
]


def build_temporal_notebook():
    features = ["date_recorded", "construction_year"]
    cells = common_cells(
        "Temporal predictors: recording date and construction age",
        "The two raw columns are audited together because pump age is their meaningful derived relationship.",
        features,
        "temporal",
    )
    cells += [
        markdown(
            """
            ## 1. Parsing, coverage and unknown years

            `date_recorded` is parsed with errors surfaced. A construction year of zero
            is handled as unknown, not as a historically meaningful year.
            """,
            "temporal-quality-heading",
        ),
        code(
            """
            training_dates = pd.to_datetime(training_features["date_recorded"], errors="coerce")
            test_dates = pd.to_datetime(test_features["date_recorded"], errors="coerce")

            date_summary = pd.DataFrame({
                "rows": [len(training_dates), len(test_dates)],
                "invalid or missing": [training_dates.isna().sum(), test_dates.isna().sum()],
                "minimum": [training_dates.min(), test_dates.min()],
                "maximum": [training_dates.max(), test_dates.max()],
                "unique dates": [training_dates.nunique(), test_dates.nunique()],
                "years": [training_dates.dt.year.nunique(), test_dates.dt.year.nunique()],
                "months represented": [training_dates.dt.month.nunique(), test_dates.dt.month.nunique()],
            }, index=["training", "test"])
            display(date_summary)
            display(
                numeric_summary(
                    training_features,
                    test_features,
                    ["construction_year"],
                    sentinel_values_by_column={"construction_year": [0]},
                )
            )
            """,
            "temporal-quality",
        ),
        code(
            """
            recording_coverage = pd.crosstab(
                training_dates.dt.year.rename("year"),
                training_dates.dt.month.rename("month"),
            )
            display(recording_coverage)

            monthly_share = pd.DataFrame({
                "training (%)": training_dates.dt.month.value_counts(normalize=True).sort_index() * 100,
                "test (%)": test_dates.dt.month.value_counts(normalize=True).sort_index() * 100,
            }).fillna(0).round(2)
            monthly_share.plot(kind="bar", figsize=(10, 4), color=["#2b6cb0", "#dd6b20"])
            plt.title("Recording-month distribution")
            plt.ylabel("rows (%)")
            plt.tight_layout()
            plt.show()
            display(monthly_share)
            """,
            "temporal-coverage",
        ),
        markdown(
            """
            ## 2. Derived pump age and validity

            Pump age is `recording year - construction year`. It is missing when the
            construction year is zero and invalid when negative. Keeping an availability
            flag distinguishes unknown age from a genuinely new pump.
            """,
            "age-heading",
        ),
        code(
            """
            def derive_pump_age(features):
                dates = pd.to_datetime(features["date_recorded"], errors="coerce")
                construction_year = pd.to_numeric(features["construction_year"], errors="coerce")
                known_year = construction_year.gt(0)
                age = dates.dt.year.sub(construction_year).where(known_year)
                valid_age = age.where(age.ge(0))
                return pd.DataFrame({
                    "recording_year": dates.dt.year,
                    "recording_month": dates.dt.month,
                    "known_construction_year": known_year,
                    "raw_pump_age": age,
                    "pump_age": valid_age,
                })

            training_temporal = derive_pump_age(training_features)
            test_temporal = derive_pump_age(test_features)
            age_summary = pd.DataFrame({
                "known construction year (%)": [
                    training_temporal["known_construction_year"].mean() * 100,
                    test_temporal["known_construction_year"].mean() * 100,
                ],
                "negative ages": [
                    training_temporal["raw_pump_age"].lt(0).sum(),
                    test_temporal["raw_pump_age"].lt(0).sum(),
                ],
                "median valid age": [
                    training_temporal["pump_age"].median(),
                    test_temporal["pump_age"].median(),
                ],
                "p90 valid age": [
                    training_temporal["pump_age"].quantile(0.90),
                    test_temporal["pump_age"].quantile(0.90),
                ],
                "maximum valid age": [
                    training_temporal["pump_age"].max(),
                    test_temporal["pump_age"].max(),
                ],
            }, index=["training", "test"]).round(2)
            display(age_summary)
            """,
            "derive-age",
        ),
        markdown(
            """
            ## 3. Relationship with `status_group`

            Age bands are descriptive and deliberately broad. The missing band is kept
            visible because the absence of a construction year may itself be informative.
            """,
            "temporal-target-heading",
        ),
        code(
            """
            labelled_temporal = training_temporal.copy()
            labelled_temporal["status_group"] = training_labels["status_group"].to_numpy()
            labelled_temporal["age band"] = pd.cut(
                labelled_temporal["pump_age"],
                bins=[-0.001, 0, 5, 10, 20, 30, 40, np.inf],
                labels=["0", "1-5", "6-10", "11-20", "21-30", "31-40", "41+"],
            ).astype("string").fillna("unknown/invalid")

            age_status = pd.crosstab(
                labelled_temporal["age band"],
                labelled_temporal["status_group"],
                normalize="index",
            ).mul(100).round(2)
            age_status.insert(0, "rows", labelled_temporal["age band"].value_counts())
            display(age_status)

            recording_year_status = pd.crosstab(
                labelled_temporal["recording_year"],
                labelled_temporal["status_group"],
                normalize="index",
            ).mul(100).round(2)
            recording_year_status.insert(
                0, "rows", labelled_temporal["recording_year"].value_counts()
            )
            display(recording_year_status)
            """,
            "temporal-target",
        ),
        markdown(
            """
            ## 4. Training/test stability

            The test set must be representable by transformations defined from training.
            Coverage is compared for recording year, month and valid pump age.
            """,
            "temporal-drift-heading",
        ),
        code(
            """
            temporal_drift = pd.DataFrame({
                "training": [
                    training_temporal["recording_year"].min(),
                    training_temporal["recording_year"].max(),
                    training_temporal["pump_age"].median(),
                    training_temporal["known_construction_year"].mean() * 100,
                ],
                "test": [
                    test_temporal["recording_year"].min(),
                    test_temporal["recording_year"].max(),
                    test_temporal["pump_age"].median(),
                    test_temporal["known_construction_year"].mean() * 100,
                ],
            }, index=[
                "minimum recording year",
                "maximum recording year",
                "median valid pump age",
                "known construction year (%)",
            ]).round(2)
            display(temporal_drift)
            """,
            "temporal-drift",
        ),
    ]
    cells += decision_cells(
        TEMPORAL_DECISIONS,
        "Compute age row by row, then fit any age imputation only on the training fold.",
    )
    return notebook(cells)


BINARY_DECISIONS = [
    {
        "feature": "public_meeting",
        "quality finding": "The source is blank for 3,334 training rows (5.61%) and 821 test rows.",
        "baseline treatment": "Retain true/false/unknown as three explicit states.",
        "risk to verify": "Missingness may reflect the data-collection process rather than pump condition.",
    },
    {
        "feature": "permit",
        "quality finding": "The source is blank for 3,056 training rows (5.15%) and 737 test rows.",
        "baseline treatment": "Retain true/false/unknown as three explicit states.",
        "risk to verify": "Permit status may proxy geography or administration.",
    },
]


def build_binary_notebook():
    features = ["public_meeting", "permit"]
    cells = common_cells(
        "Binary predictors: observed false versus unknown",
        "Missing values remain a third auditable state; they are not silently coerced to `False`.",
        features,
        "binary",
    )
    cells += [
        markdown("## 1. Value coverage and missingness", "binary-quality-heading"),
        code(
            """
            display(categorical_summary(training_features, test_features, audited_features))
            for feature in audited_features:
                print()
                print(feature)
                display(category_frequency_table(training_features, test_features, feature, top_n=5))
            """,
            "binary-quality",
        ),
        markdown(
            """
            ## 2. Relationship with `status_group`

            The class percentages are conditional on each true/false/unknown state.
            Row support is shown before any interpretation of percentage differences.
            """,
            "binary-target-heading",
        ),
        code(
            """
            for feature in audited_features:
                print()
                print(feature)
                display(
                    categorical_target_profile(
                        training_data,
                        feature,
                        minimum_support=100,
                    )
                )
            """,
            "binary-target",
        ),
        markdown("## 3. Training/test stability", "binary-drift-heading"),
        code(
            """
            binary_stability = categorical_summary(
                training_features,
                test_features,
                audited_features,
            )[[
                "training explicit missing",
                "test explicit missing",
                "test-only levels",
                "test rows in unseen levels (%)",
                "marginal total-variation distance",
            ]]
            display(binary_stability)
            """,
            "binary-drift",
        ),
    ]
    cells += decision_cells(
        BINARY_DECISIONS,
        "Use nullable booleans or an explicit unknown category; never apply Python truthiness to missing values.",
    )
    return notebook(cells)


def category_notebook(
    *,
    title: str,
    scope: str,
    features: list[str],
    review_type: str,
    relationships: list[tuple[str, str]],
    decisions: list[dict[str, str]],
    minimum_support: int,
    sentinel_tokens_by_column: dict[str, list[str]] | None = None,
    extra_cells: list | None = None,
    caveat: str,
):
    cells = common_cells(title, scope, features, review_type)
    sentinel_tokens_by_column = sentinel_tokens_by_column or {}
    cells += [
        markdown(
            """
            ## 1. Missingness, cardinality and test coverage

            Source blanks, pandas nulls and configured sentinel strings are reported separately.
            Semantic sentinels such as `unknown` stay visible in frequency and target tables;
            they are not silently merged with blank values.
            Rare means fewer than 50 training rows; it is a diagnostic threshold, not
            a preprocessing choice. Total-variation distance compares marginal shares.
            """,
            "categorical-quality-heading",
        ),
        code(
            f"""
            category_overview = categorical_summary(
                training_features,
                test_features,
                audited_features,
                rare_threshold=50,
                sentinel_tokens_by_column={sentinel_tokens_by_column!r},
            )
            display(category_overview)
            """,
            "categorical-quality",
        ),
        markdown("## 2. Most common values", "frequency-heading"),
        code(
            """
            for feature in audited_features:
                print()
                print(feature)
                display(category_frequency_table(training_features, test_features, feature, top_n=10))
            """,
            "frequency-tables",
        ),
        markdown(
            f"""
            ## 3. Relationship with `status_group`

            The tables display the most supported levels first and mark whether each
            level has at least {minimum_support:,} training rows. Small groups are leads
            for later validation, not stable target encodings.
            """,
            "categorical-target-heading",
        ),
        code(
            f"""
            for feature in audited_features:
                print()
                print(feature)
                profile = categorical_target_profile(
                    training_data,
                    feature,
                    minimum_support={minimum_support},
                )
                display(profile.head(15))
            """,
            "categorical-target",
        ),
    ]
    if relationships:
        cells += [
            markdown(
                """
                ## 4. Related-field consistency

                A deterministic child-to-parent mapping makes the parent derivable from
                the child in this dataset. That is redundancy evidence, not automatic
                permission to discard the child: granularity, unseen levels and model
                behaviour still determine which representation is safer.
                """,
                "hierarchy-heading",
            ),
            code(
                f"""
                hierarchy_relationships = {relationships!r}
                display(
                    hierarchy_summary(
                        training_features,
                        test_features,
                        hierarchy_relationships,
                    )
                )
                for child, parent in hierarchy_relationships:
                    conflicts = hierarchy_conflicts(training_features, child, parent)
                    if not conflicts.empty:
                        print()
                        print(f"Training conflicts for {{child}} -> {{parent}}")
                        display(conflicts)
                """,
                "hierarchy-summary",
            ),
        ]
    if extra_cells:
        cells += extra_cells
    cells += [
        markdown("## Training/test handoff", "categorical-drift-heading"),
        code(
            """
            display(
                category_overview[[
                    "training levels",
                    "test levels",
                    "test-only levels",
                    "test rows in unseen levels (%)",
                    "training rows in rare levels (%)",
                    "marginal total-variation distance",
                ]].sort_values("test rows in unseen levels (%)", ascending=False)
            )
            """,
            "categorical-drift",
        ),
    ]
    cells += decision_cells(decisions, caveat)
    return notebook(cells)


GEOGRAPHY_FEATURES = [
    "basin",
    "subvillage",
    "region",
    "region_code",
    "district_code",
    "lga",
    "ward",
]
GEOGRAPHY_DECISIONS = [
    {
        "feature": feature,
        "quality finding": finding,
        "baseline treatment": treatment,
        "risk to verify": risk,
    }
    for feature, finding, treatment, risk in [
        ("basin", "Nine complete levels; train/test total-variation distance is 1.23%.", "Retain as a stable low-cardinality category.", "Hydrological basin crosses administrative geography."),
        ("subvillage", "19,287 levels; 87.32% of rows are in <50 groups and 16.09% of test rows are unseen.", "Exclude from first one-hot baseline; validate hashing/frequency treatment separately.", "Extreme sparsity and reused names encourage memorisation."),
        ("region", "Twenty-one complete levels; functional rates range from 29.75% to 78.22%.", "Retain as an interpretable low-cardinality back-off.", "Random validation may reward geographic memorisation."),
        ("region_code", "Twenty-seven categorical codes; region and code are not simple duplicates.", "Cast to category and compare with named region.", "Do not scale or interpret code distance."),
        ("district_code", "Twenty reused numeric labels; code zero is not universally missing.", "Cast to category and combine with region_code if used.", "The same code can occur in different regions."),
        ("lga", "125 complete levels, no unseen test levels, and deterministic mapping to region.", "Retain; compare its signal against region back-off.", "Add an LGA/region-grouped validation sensitivity check."),
        ("ward", "2,092 levels; raw unseen exposure is 0.07%, or 0.08% for LGA+ward.", "Retain via LGA+ward with fold-fitted rare/unseen handling.", "Shared ward names need the LGA context."),
    ]
]


def build_geography_notebook():
    extra = [
        markdown(
            """
            ## 5. Geographic code semantics

            `region_code` and `district_code` are labels despite their integer dtype.
            District codes are checked both alone and as a region/district composite.
            """,
            "code-semantics-heading",
        ),
        code(
            """
            training_geo = training_features.copy()
            test_geo = test_features.copy()
            for frame in (training_geo, test_geo):
                frame["region_district"] = (
                    frame["region_code"].astype("string")
                    + ":"
                    + frame["district_code"].astype("string")
                )
            composite_summary = categorical_summary(
                training_geo,
                test_geo,
                ["region_district"],
            )
            display(composite_summary)

            lga_mapping = hierarchy_summary(
                training_geo,
                test_geo,
                [("lga", "region_district")],
            )
            display(lga_mapping)
            """,
            "code-semantics",
        ),
    ]
    return category_notebook(
        title="Geographic categories: hierarchy, codes and coverage",
        scope="Coordinates have their own paired audit; this notebook examines the named and encoded geographic context.",
        features=GEOGRAPHY_FEATURES,
        review_type="geographic categorical",
        relationships=[
            ("region", "region_code"),
            ("lga", "region"),
            ("ward", "lga"),
            ("subvillage", "ward"),
        ],
        decisions=GEOGRAPHY_DECISIONS,
        minimum_support=200,
        sentinel_tokens_by_column={},
        extra_cells=extra,
        caveat="Treat administrative codes as unordered categories and compare nested levels by validation rather than retaining every hierarchy level by default.",
    )


ORGANISATION_FEATURES = ["funder", "installer", "wpt_name"]
ORGANISATION_DECISIONS = [
    {
        "feature": "funder",
        "quality finding": "Effective missingness is 7.48%; 1.71% of test rows have unseen funders.",
        "baseline treatment": "Keep null/sentinel distinct, normalise conservatively, then rare-pool inside folds.",
        "risk to verify": "Naive target encoding leaks; direct one-hot encoding is unstable.",
    },
    {
        "feature": "installer",
        "quality finding": "Effective missingness is 7.54%; case normalisation reduces 2,145 to 1,918 levels.",
        "baseline treatment": "Keep separately from funder; trim/casefold/whitespace-normalise and rare-pool.",
        "risk to verify": "Aliases require cautious, reviewable normalisation rather than fuzzy merging.",
    },
    {
        "feature": "wpt_name",
        "quality finding": "56.90% of test names are unseen; leave-one-out lookup accuracy is only 55.01%.",
        "baseline treatment": "Exclude raw exact names from baseline; test cross-fitted text/frequency features later.",
        "risk to verify": "Memorisation and geographic proxying can overstate local validation value.",
    },
]


def build_organisation_notebook():
    extra = [
        markdown(
            """
            ## 4. Safe text normalisation and cross-field overlap

            Only whitespace trimming and Unicode-aware case folding are counted here.
            Fuzzy spelling merges would change identity and need an explicit reviewed map.
            """,
            "normalisation-heading",
        ),
        code(
            """
            display(text_normalisation_summary(training_features, test_features, audited_features))

            funder_missing = (
                source_blank_mask(training_features["funder"])
                | sentinel_mask(
                    training_features["funder"],
                    ["0", "none", "unknown", "not known"],
                )
            )
            installer_missing = (
                source_blank_mask(training_features["installer"])
                | sentinel_mask(
                    training_features["installer"],
                    ["0", "unknown", "not known", "-", "unknown installer"],
                )
            )
            normalised_funder = normalise_categories(training_features["funder"]).mask(funder_missing)
            normalised_installer = normalise_categories(training_features["installer"]).mask(installer_missing)
            both_observed = normalised_funder.notna() & normalised_installer.notna()
            same_normalised_value = normalised_funder.eq(normalised_installer).fillna(False)
            overlap = pd.DataFrame({
                "rows": [len(training_features)],
                "both observed": [both_observed.sum()],
                "same normalised value": [same_normalised_value.sum()],
                "same among observed (%)": [
                    same_normalised_value.sum() / both_observed.sum() * 100
                ],
            }, index=["training"]).round(2)
            display(overlap)

            pair_counts = pd.DataFrame({
                "funder": normalised_funder,
                "installer": normalised_installer,
            }).dropna().value_counts().rename("rows").head(15)
            display(pair_counts.to_frame())
            """,
            "normalisation-overlap",
        ),
    ]
    return category_notebook(
        title="Organisation and waterpoint names: high-cardinality audit",
        scope="These strings need coverage and memorisation checks before any encoding choice.",
        features=ORGANISATION_FEATURES,
        review_type="high-cardinality text/category",
        relationships=[],
        decisions=ORGANISATION_DECISIONS,
        minimum_support=100,
        sentinel_tokens_by_column={
            "funder": ["0", "none", "unknown", "not known"],
            "installer": ["0", "unknown", "not known", "-", "unknown installer"],
            "wpt_name": ["none", "unknown", "not known"],
        },
        extra_cells=extra,
        caveat="Start with explicit unknown and rare buckets. Any frequency, hashing or target encoding must be fitted within folds and justified by validation.",
    )


MANAGEMENT_FEATURES = [
    "scheme_management",
    "scheme_name",
    "management",
    "management_group",
]
MANAGEMENT_DECISIONS = [
    {
        "feature": "scheme_management",
        "quality finding": "3,877 training rows are blank; it overlaps management but is not equivalent.",
        "baseline treatment": "Retain unknown as a level; compare against management by ablation.",
        "risk to verify": "Parallel management fields may add redundancy more than signal.",
    },
    {
        "feature": "scheme_name",
        "quality finding": "28,166 rows are blank, 644 literal None; 1.26% of test rows use unseen names.",
        "baseline treatment": "Omit raw one-hot form initially; preserve blank/None and test fold-fitted pooling later.",
        "risk to verify": "Scheme identity can memorise local geography.",
    },
    {
        "feature": "management",
        "quality finding": "Twelve complete levels map deterministically to five management_group levels.",
        "baseline treatment": "Use as the initial granular management representation.",
        "risk to verify": "Some sparse levels may need grouping.",
    },
    {
        "feature": "management_group",
        "quality finding": "Deterministic coarse parent retains much less descriptive target association.",
        "baseline treatment": "Treat as likely redundant; compare coarse versus granular representation.",
        "risk to verify": "Coarser levels may generalise better to rare management methods.",
    },
]


def build_management_notebook():
    extra = [
        markdown(
            """
            ## 5. Preserve source blanks and literal scheme-name sentinels

            Loading with `keep_default_na=False` prevents pandas from merging the literal
            string `None` with an empty CSV cell. The states below remain separate because
            their target distributions differ and only the blank is structurally absent.
            """,
            "scheme-source-states-heading",
        ),
        code(
            """
            def scheme_source_state(series):
                raw = series.astype("string")
                return pd.Series(
                    np.select(
                        [
                            raw.str.strip().eq(""),
                            raw.eq("None"),
                            raw.eq("none"),
                            raw.str.strip().str.casefold().eq("no scheme"),
                        ],
                        ["blank", "literal None", "literal none", "no scheme"],
                        default="other recorded name",
                    ),
                    index=series.index,
                )

            for feature in ["scheme_management", "scheme_name"]:
                source_state = scheme_source_state(training_data[feature])
                source_profile = pd.crosstab(
                    source_state,
                    training_data["status_group"],
                    normalize="index",
                ).mul(100).round(2)
                source_profile.insert(0, "rows", source_state.value_counts())
                print()
                print(feature)
                display(source_profile)
            """,
            "scheme-source-states",
        ),
        markdown("## 6. Management-field overlap", "management-overlap-heading"),
        code(
            """
            scheme = analysis_categories(training_features["scheme_management"])
            management = analysis_categories(training_features["management"])
            overlap_table = pd.crosstab(scheme, management)
            display(overlap_table)
            display(pd.DataFrame({
                "Cramer's V": [cramer_v(overlap_table)],
                "exact normalised matches (%)": [scheme.eq(management).mean() * 100],
            }, index=["scheme_management vs management"]).round(3))
            """,
            "management-overlap",
        ),
    ]
    return category_notebook(
        title="Management and scheme predictors: overlap and identity",
        scope="The audit separates low-cardinality management types from the high-cardinality scheme identifier.",
        features=MANAGEMENT_FEATURES,
        review_type="related categorical",
        relationships=[("management", "management_group"), ("scheme_name", "scheme_management")],
        decisions=MANAGEMENT_DECISIONS,
        minimum_support=150,
        extra_cells=extra,
        caveat="Prefer one validated representation per management concept; treat scheme identity as a high-cardinality feature with an explicit omission baseline.",
    )


INFRASTRUCTURE_FEATURES = [
    "extraction_type",
    "extraction_type_group",
    "extraction_type_class",
    "source",
    "source_type",
    "source_class",
]
INFRASTRUCTURE_DECISIONS = [
    {
        "feature": feature,
        "quality finding": finding,
        "baseline treatment": treatment,
        "risk to verify": "Related hierarchy levels are redundant when used together without evidence.",
    }
    for feature, finding, treatment in [
        ("extraction_type", "Eighteen levels map deterministically upward; `other` is 80.79% non-functional.", "Keep the granular type; rare-pool if validation requires it."),
        ("extraction_type_group", "Deterministic from type and only 0.0013 descriptive MI bits lower.", "Treat as a strong redundancy candidate beside extraction_type."),
        ("extraction_type_class", "Seven deterministic broad classes give a compact back-off.", "Compare type plus class against type alone; do not keep all three."),
        ("source", "Ten fully test-covered levels retain lake/river differences hidden by source_type.", "Keep granular source and compare with the intermediate type."),
        ("source_type", "Deterministic from source and removes useful within-type detail.", "Use only as a lower-cardinality ablation candidate."),
        ("source_class", "Three deterministic classes have weak descriptive association.", "First source-hierarchy level to omit from baseline."),
    ]
]


def build_infrastructure_notebook():
    return category_notebook(
        title="Extraction and source predictors: categorical hierarchies",
        scope="Two three-level hierarchies describe how water is extracted and where it originates.",
        features=INFRASTRUCTURE_FEATURES,
        review_type="categorical hierarchy",
        relationships=[
            ("extraction_type", "extraction_type_group"),
            ("extraction_type_group", "extraction_type_class"),
            ("source", "source_type"),
            ("source_type", "source_class"),
        ],
        decisions=INFRASTRUCTURE_DECISIONS,
        minimum_support=150,
        caveat="Do not one-hot every level of a deterministic hierarchy by default. Compare granular and coarse alternatives on the same split.",
    )


SERVICE_FEATURES = [
    "payment_type",
    "water_quality",
    "quality_group",
    "quantity",
    "waterpoint_type",
    "waterpoint_type_group",
]
SERVICE_DECISIONS = [
    {
        "feature": feature,
        "quality finding": finding,
        "baseline treatment": treatment,
        "risk to verify": risk,
    }
    for feature, finding, treatment, risk in [
        ("payment_type", "Seven test-covered levels; unknown is 51.45% non-functional versus 38.42% overall.", "Retain unknown/other explicitly after removing duplicate payment.", "Payment patterns may proxy local administration."),
        ("water_quality", "Eight levels map deterministically upward; unknown is 84.06% non-functional.", "Keep granular quality and rare-pool the 17-row fluoride-abandoned level.", "Sentinel signal may reflect collection quality."),
        ("quality_group", "Deterministic parent hides the salty versus salty-abandoned repair difference.", "Treat as likely redundant beside water_quality.", "Coarsening can remove useful detail."),
        ("quantity", "Dry covers 6,246 rows and is 96.89% non-functional; quantity_group is its duplicate.", "Retain as nominal; preserve unknown and never reintroduce quantity_group.", "Strong association is legitimate but not causal evidence."),
        ("waterpoint_type", "Seven levels; `other` is 82.24% non-functional and `dam` has only seven rows.", "Keep granular type with infrequent handling.", "The dam estimate is too sparse to interpret."),
        ("waterpoint_type_group", "Deterministic parent hides single-versus-multiple standpipe differences.", "Treat as likely redundant; use only as coarse ablation.", "Coarsening can remove useful detail."),
    ]
]


def build_service_notebook():
    return category_notebook(
        title="Service predictors: payment, water state and waterpoint type",
        scope="This notebook audits the retained canonical quantity/payment fields and two related categorical pairs.",
        features=SERVICE_FEATURES,
        review_type="low-cardinality categorical",
        relationships=[
            ("water_quality", "quality_group"),
            ("waterpoint_type", "waterpoint_type_group"),
        ],
        decisions=SERVICE_DECISIONS,
        minimum_support=150,
        caveat="Keep `payment_type` and `quantity`; the duplicate `payment` and `quantity_group` columns remain excluded by the settled structural policy.",
    )


def notebook(cells):
    result = nbformat.v4.new_notebook(cells=cells, metadata=KERNEL_METADATA)
    nbformat.validate(result)
    return result


NOTEBOOK_BUILDERS = {
    "01-data-audit-numeric.ipynb": build_numeric_notebook,
    "01-data-audit-temporal.ipynb": build_temporal_notebook,
    "01-data-audit-binary.ipynb": build_binary_notebook,
    "01-data-audit-geographic-categories.ipynb": build_geography_notebook,
    "01-data-audit-organisation-and-name.ipynb": build_organisation_notebook,
    "01-data-audit-management.ipynb": build_management_notebook,
    "01-data-audit-infrastructure.ipynb": build_infrastructure_notebook,
    "01-data-audit-service-categories.ipynb": build_service_notebook,
}


def all_feature_register_rows():
    rows = [
        {
            "feature": "amount_tsh",
            "review type": "numeric",
            "focused audit": "01-data-audit-amount_tsh.ipynb",
            "audit status": "complete",
            "key finding": "Zero dominates and positive values are strongly right-skewed.",
            "baseline treatment": "Keep amount availability and compare transformed positive values.",
        },
        {
            "feature": "longitude + latitude",
            "review type": "paired geographic",
            "focused audit": "01-data-audit-coordinates.ipynb",
            "audit status": "complete",
            "key finding": "The pair (0, approximately 0) encodes 1,812 missing training locations.",
            "baseline treatment": "Keep availability; replace the paired sentinel and engineer location jointly.",
        },
    ]

    configs = [
        ("01-data-audit-numeric.ipynb", "numeric", NUMERIC_DECISIONS),
        ("01-data-audit-temporal.ipynb", "temporal", TEMPORAL_DECISIONS),
        ("01-data-audit-binary.ipynb", "binary", BINARY_DECISIONS),
        ("01-data-audit-geographic-categories.ipynb", "geographic categorical", GEOGRAPHY_DECISIONS),
        ("01-data-audit-organisation-and-name.ipynb", "high-cardinality categorical", ORGANISATION_DECISIONS),
        ("01-data-audit-management.ipynb", "related categorical", MANAGEMENT_DECISIONS),
        ("01-data-audit-infrastructure.ipynb", "categorical hierarchy", INFRASTRUCTURE_DECISIONS),
        ("01-data-audit-service-categories.ipynb", "low-cardinality categorical", SERVICE_DECISIONS),
    ]
    for filename, review_type, decisions in configs:
        for decision in decisions:
            rows.append(
                {
                    "feature": decision["feature"],
                    "review type": review_type,
                    "focused audit": filename,
                    "audit status": "complete",
                    "key finding": decision["quality finding"],
                    "baseline treatment": decision["baseline treatment"],
                }
            )
    return rows


def update_main_audit():
    path = NOTEBOOK_DIRECTORY / "01-data-audit.ipynb"
    audit = nbformat.read(path, as_version=4)
    register_rows = all_feature_register_rows()

    replacements = {
        "load-code": code(
            """
            # Preserve source blanks and literal strings such as "None" separately.
            training_set_values = pd.read_csv(
                training_set_values_path,
                keep_default_na=False,
            )
            training_set_labels = pd.read_csv(
                training_set_labels_path,
                keep_default_na=False,
            )
            test_set_values = pd.read_csv(
                test_set_values_path,
                keep_default_na=False,
            )
            submission_format = pd.read_csv(
                submission_format_path,
                keep_default_na=False,
            )

            print("Data loaded ok")
            print("==============")
            print("training_set_values:", training_set_values.shape[0])
            print("training_set_labels:", training_set_labels.shape[0])
            print("test_set_values:", test_set_values.shape[0])
            print("submission_format:", submission_format.shape[0])
            """,
            "load-code",
        ),
        "first-look-code": code(
            """
            # Display a small sample and an aligned column-level schema summary.
            display(training_set_values.head().T)

            column_summary = pd.DataFrame({
                "dtype": training_set_values.dtypes.astype("string"),
                "example": training_set_values.iloc[0],
                "source blank rows": [
                    int(training_set_values[column].astype("string").str.strip().eq("").sum())
                    for column in training_set_values.columns
                ],
                "unique values": training_set_values.nunique(dropna=False),
            })
            display(column_summary)
            """,
            "first-look-code",
        ),
        "feature-register-code": code(
            f"""
            feature_register = pd.DataFrame({register_rows!r})
            assert len(feature_register) == 35  # 36 candidate columns; coordinates are one paired row.
            assert feature_register["audit status"].eq("complete").all()
            display(feature_register)
            """,
            "feature-register-code",
        ),
        "numeric-code": code(
            """
            display(feature_register.loc[feature_register["review type"].eq("numeric")])
            print("Executed distribution, sentinel, target-state and drift evidence: 01-data-audit-numeric.ipynb")
            """,
            "numeric-code",
        ),
        "categorical-code": code(
            """
            categorical_rows = feature_register["review type"].str.contains(
                "categorical|hierarchy|related",
                case=False,
                regex=True,
            )
            display(feature_register.loc[categorical_rows])
            print("Focused notebooks use a shared 50-row rarity diagnostic and support-aware target tables.")
            """,
            "categorical-code",
        ),
        "binary-code": code(
            """
            display(feature_register.loc[feature_register["review type"].eq("binary")])
            print("public_meeting and permit keep true, false and source-blank states distinct.")
            """,
            "binary-code",
        ),
        "date-code": code(
            """
            display(feature_register.loc[feature_register["review type"].eq("temporal")])
            print("All dates parse. Pump age is missing for unknown years and invalid for nine negative training cases.")
            """,
            "date-code",
        ),
        "identifier-code": code(
            """
            display(
                feature_register.loc[
                    feature_register["review type"].eq("high-cardinality categorical")
                ]
            )
            print("Raw exact waterpoint names are excluded from the first baseline because 56.90% of test rows are unseen.")
            """,
            "identifier-code",
        ),
        "geography-code": code(
            """
            display(
                feature_register.loc[
                    feature_register["review type"].str.contains("geographic")
                ]
            )
            print("Regional functional rates span 29.75% to 78.22%; add an LGA/region-grouped robustness check.")
            """,
            "geography-code",
        ),
        "hierarchy-code": code(
            """
            hierarchy_rows = feature_register["review type"].isin([
                "categorical hierarchy",
                "related categorical",
                "low-cardinality categorical",
            ])
            display(feature_register.loc[hierarchy_rows])
            print("Documented extraction, source, quality, management and waterpoint child-parent mappings are deterministic; scheme_name to scheme_management is not.")
            """,
            "hierarchy-code",
        ),
        "missing-code": code(
            """
            missing_data_register = pd.DataFrame([
                {"feature/state": "amount_tsh = 0", "training rows": 41639, "test rows": 10410, "treatment": "availability plus transformed positive magnitude"},
                {"feature/state": "coordinate pair sentinel", "training rows": 1812, "test rows": 457, "treatment": "paired availability then missing coordinates"},
                {"feature/state": "gps_height = 0", "training rows": 20438, "test rows": 5211, "treatment": "availability then fold-fitted imputation"},
                {"feature/state": "population = 0", "training rows": 21381, "test rows": 5453, "treatment": "keep separate from population=1"},
                {"feature/state": "construction_year = 0", "training rows": 20709, "test rows": 5260, "treatment": "unknown-year and valid-age features"},
                {"feature/state": "public_meeting blank", "training rows": 3334, "test rows": 821, "treatment": "third categorical state"},
                {"feature/state": "permit blank", "training rows": 3056, "test rows": 737, "treatment": "third categorical state"},
                {"feature/state": "funder blank/sentinel", "training rows": 4445, "test rows": 1079, "treatment": "keep blank and sentinel distinct"},
                {"feature/state": "installer blank/sentinel", "training rows": 4481, "test rows": 1087, "treatment": "keep blank and sentinel distinct"},
                {"feature/state": "scheme_name blank", "training rows": 28166, "test rows": 7092, "treatment": "distinct from literal None/none"},
            ])
            missing_data_register["training (%)"] = (
                missing_data_register["training rows"] / len(training_set_values) * 100
            ).round(2)
            missing_data_register["test (%)"] = (
                missing_data_register["test rows"] / len(test_set_values) * 100
            ).round(2)
            display(missing_data_register)
            """,
            "missing-code",
        ),
        "drift-code": code(
            """
            predictor_drift_summary = pd.DataFrame([
                {"family": "numeric", "evidence": "maximum full-distribution KS D = 0.0112", "decision": "No material marginal shift; retain range and sentinel guards."},
                {"family": "temporal", "evidence": "recording-year/month total variation < 0.9 percentage points", "decision": "Use training-defined date derivations."},
                {"family": "binary", "evidence": "total variation <= 0.28 percentage points", "decision": "Keep explicit unknown handling."},
                {"family": "low-cardinality categories", "evidence": "supplied test levels are broadly covered", "decision": "Still configure handle_unknown for validation/future data."},
                {"family": "subvillage", "evidence": "16.09% of test rows use unseen levels", "decision": "Exclude raw one-hot form from baseline."},
                {"family": "wpt_name", "evidence": "56.90% of test rows use unseen exact names", "decision": "Exclude raw exact names from baseline."},
                {"family": "scheme_name", "evidence": "about 1.1% unseen after conservative normalisation", "decision": "Use only a fold-fitted high-cardinality experiment."},
            ])
            display(predictor_drift_summary)
            """,
            "drift-code",
        ),
        "consistency-code": code(
            """
            consistency_summary = pd.DataFrame([
                {"relationship": "quantity_group / quantity", "finding": "exact duplicate", "action": "drop quantity_group"},
                {"relationship": "payment / payment_type", "finding": "fixed relabelling", "action": "drop payment"},
                {"relationship": "recorded_by", "finding": "constant", "action": "drop recorded_by"},
                {"relationship": "management / management_group", "finding": "deterministic child-to-parent", "action": "prefer management; validate coarse ablation"},
                {"relationship": "extraction/source/quality/waterpoint families", "finding": "deterministic documented hierarchies", "action": "do not keep every level automatically"},
                {"relationship": "lga / region", "finding": "LGA maps deterministically to region", "action": "retain region as low-cardinality back-off"},
                {"relationship": "region / region_code", "finding": "not simple duplicates", "action": "treat code categorically and compare"},
                {"relationship": "scheme_name / scheme_management", "finding": "non-deterministic", "action": "do not derive one mechanically from the other"},
            ])
            display(consistency_summary)
            """,
            "consistency-code",
        ),
        "review-order": markdown(
            """
            ### Focused review order and coverage

            The remaining audit is organised by analytical treatment rather than CSV order:

            1. [`amount_tsh`](01-data-audit-amount_tsh.ipynb) and the paired [`longitude` / `latitude`](01-data-audit-coordinates.ipynb) audits.
            2. [Numeric magnitude](01-data-audit-numeric.ipynb), [temporal fields](01-data-audit-temporal.ipynb) and [nullable binary fields](01-data-audit-binary.ipynb).
            3. [Geographic categories](01-data-audit-geographic-categories.ipynb) and [high-cardinality organisations/names](01-data-audit-organisation-and-name.ipynb).
            4. Related [management](01-data-audit-management.ipynb), [extraction/source](01-data-audit-infrastructure.ipynb) and [service](01-data-audit-service-categories.ipynb) categories.

            Together these notebooks cover all 36 candidate predictors after the fixed structural removal, counting longitude and latitude separately. The templates below remain the governing checklist; the focused notebooks contain their executed evidence.
            """,
            "review-order",
        ),
        "decision-log": markdown(
            """
            ## 18. Consolidated predictor findings and decision log

            The detailed evidence lives in the focused notebooks. This section keeps
            the cross-cutting findings needed by notebook 02 in one place.
            """,
            "decision-log",
        ),
        "decision-code": code(
            """
            audit_decision_summary = pd.DataFrame([
                {"finding": "candidate predictor coverage", "evidence": "36 of 36", "action": "Proceed to a fixed split and preprocessing baseline."},
                {"finding": "settled raw-column removals", "evidence": "quantity_group, payment, recorded_by", "action": "Apply the shared fixed removal to training and test."},
                {"finding": "source-token preservation", "evidence": "28,166 blank scheme names versus 644 literal None values", "action": "Load with keep_default_na=False; distinguish blanks from semantic sentinels."},
                {"finding": "shared zero-measurement block", "evidence": "19,668 training rows (33.11%) have zero height, population and construction year", "action": "Create feature-specific flags; never replace all numeric zeros globally."},
                {"finding": "population special states", "evidence": "population=1 covers 7,025 rows and is 53.52% non-functional", "action": "Keep zero and one distinct from ordinary positive magnitude."},
                {"finding": "nullable binary fields", "evidence": "public_meeting blank 5.61%; permit blank 5.15%", "action": "Preserve unknown as a third state independently for each field."},
                {"finding": "temporal derivation", "evidence": "valid pump age median 13; non-functional rises from 23.73% at 1-5 years to 67.24% at 41+", "action": "Derive valid age and inconsistency/unknown flags; fit imputation inside folds."},
                {"finding": "sparse location/name fields", "evidence": "unseen test rows: subvillage 16.09%, wpt_name 56.90%", "action": "Exclude raw one-hot forms from baseline; test fold-safe hashing/frequency approaches separately."},
                {"finding": "waterpoint-name optimism", "evidence": "84.70% in-sample category-mode accuracy falls to 55.01% leave-one-out lookup", "action": "Do not use raw exact wpt_name in the first baseline."},
                {"finding": "scheme-name coverage", "evidence": "47.42% source blank; unseen test names affect 1.26%", "action": "Keep blank/None distinct and test only fold-fitted high-cardinality treatments."},
                {"finding": "deterministic hierarchies", "evidence": "extraction, source, quality, management and waterpoint child-parent mappings", "action": "Prefer a validated granular or coarse level; do not automatically retain all levels."},
                {"finding": "strong service-state associations", "evidence": "quantity=dry is 96.89% non-functional; water_quality=unknown is 84.06%", "action": "Retain as nominal categories with explicit unknown and support-aware interpretation."},
                {"finding": "geographic dependence", "evidence": "regional functional rate spans 29.75% to 78.22%", "action": "Keep the stratified split and add an LGA/region-grouped robustness check."},
                {"finding": "aggregate train/test stability", "evidence": "small numeric KS, binary TV and broad categorical marginal differences", "action": "Still configure unseen handling; absence of current drift does not guarantee future coverage."},
            ])
            display(audit_decision_summary)
            """,
            "decision-code",
        ),
        "completion-checklist": markdown(
            """
            ## Completion checklist

            - [x] Source files and identifier alignment are validated.
            - [x] The target has a separate integrity and class-balance audit.
            - [x] Each of the 36 candidate predictors has an entry in the feature register.
            - [x] Explicit missing values and suspected sentinels are separated.
            - [x] Related categorical and geographic features have paired consistency checks.
            - [x] Training/test ranges, levels and unseen-category exposure are compared.
            - [x] The fixed structural removal returns 37 columns while retaining `id`.
            - [x] Every proposed baseline treatment points to a focused audit finding.
            - [ ] Recheck target relationships after the stratified development/validation split is frozen.
            - [ ] Compare high-cardinality, hierarchy and geographic choices by held-out validation.
            """,
            "completion-checklist",
        ),
    }

    for index, cell in enumerate(audit.cells):
        if cell.id in replacements:
            audit.cells[index] = replacements[cell.id]

    nbformat.validate(audit)
    nbformat.write(audit, path)


def main():
    for filename, builder in NOTEBOOK_BUILDERS.items():
        notebook_path = NOTEBOOK_DIRECTORY / filename
        nbformat.write(builder(), notebook_path)
        print(f"Wrote {notebook_path.name}")
    update_main_audit()
    print("Updated 01-data-audit.ipynb")


if __name__ == "__main__":
    main()
