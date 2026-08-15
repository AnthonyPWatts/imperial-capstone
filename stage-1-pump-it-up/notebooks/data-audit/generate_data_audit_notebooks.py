"""Generate the explicit three-notebook audit set for every raw predictor."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat


DATA_AUDIT_DIRECTORY = Path(__file__).resolve().parent
STAGE_DIRECTORY = DATA_AUDIT_DIRECTORY.parents[1]
CATALOGUE_PATH = STAGE_DIRECTORY / "src" / "predictor_audit_catalogue.json"
POLICY_PATH = STAGE_DIRECTORY / "src" / "raw_feature_column_policy.json"

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.14.5"},
}


TYPE_CHECKLISTS = {
    "numeric": [
        "dtype and numeric coercion",
        "explicit missingness and feature-specific sentinels",
        "quantiles, skew, zero/negative support and extreme values",
        "training/test distribution stability",
    ],
    "coordinate": [
        "numeric range and precision",
        "paired coordinate sentinel handling",
        "training/test geographic coverage",
        "agreement with named administrative geography",
    ],
    "date": [
        "strict parsing and invalid values",
        "minimum, maximum and unique dates",
        "year/month survey-wave coverage",
        "training/test calendar stability",
    ],
    "year": [
        "numeric range and zero-year sentinel",
        "construction cohorts",
        "valid age relative to date_recorded",
        "training/test coverage",
    ],
    "binary": [
        "true, false and source-blank support",
        "no implicit coercion of blank to false",
        "class percentages with row support",
        "training/test state stability",
    ],
    "category": [
        "source blanks and semantic sentinel tokens",
        "cardinality, rare levels and common values",
        "test-only levels and unseen-row exposure",
        "support-aware target rates",
    ],
    "high-cardinality-category": [
        "source blanks and semantic sentinel tokens",
        "cardinality, singletons and rare-level row coverage",
        "normalisation effects and test-only levels",
        "memorisation risk and fold-fitted encoding options",
    ],
    "constant": [
        "unique values in training and test",
        "source-schema assumption",
        "proof that the field cannot separate outcomes",
        "future-data revalidation condition",
    ],
}


def markdown(source: str, cell_id: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n", id=cell_id)


def code(source: str, cell_id: str):
    return nbformat.v4.new_code_cell(dedent(source).strip() + "\n", id=cell_id)


def notebook(cells, *, feature: str | None = None, purpose: str | None = None):
    metadata = dict(KERNEL_METADATA)
    metadata["pump_it_up_data_audit"] = {
        "feature": feature,
        "purpose": purpose,
        "generated_from": "src/predictor_audit_catalogue.json",
    }
    result = nbformat.v4.new_notebook(cells=cells, metadata=metadata)
    nbformat.validate(result)
    return result


def load_catalogue():
    catalogue = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    features = catalogue["features"]
    names = [feature["name"] for feature in features]
    expected = policy["expected_columns"][1:]
    if names != expected:
        raise ValueError("Predictor catalogue order does not match the fixed raw schema.")
    if len(names) != len(set(names)):
        raise ValueError("Predictor catalogue contains duplicate feature names.")
    known = set(names)
    unknown_relationships = [
        (feature["name"], relationship["feature"])
        for feature in features
        for relationship in feature["related"]
        if relationship["feature"] not in known
    ]
    if unknown_relationships:
        raise ValueError(f"Unknown related features: {unknown_relationships!r}")
    return catalogue


def folder_name(feature):
    return f"{feature['order']:02d}-{feature['name']}"


def notebook_filename(feature, sequence: int, suffix: str):
    return f"{sequence:02d}-{feature['name']}-{suffix}.ipynb"


def common_load_cell(feature, catalogue):
    feature_types = {
        item["name"]: item["audit_type"] for item in catalogue["features"]
    }
    return code(
        f"""
        import sys
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display


        def find_stage_directory():
            start = Path.cwd().resolve()
            for candidate in (start, *start.parents):
                if (
                    (candidate / "data" / "TrainingSetValues.csv").exists()
                    and (candidate / "src" / "source_data_validation.py").exists()
                ):
                    return candidate
            raise FileNotFoundError("Could not locate the stage-1-pump-it-up directory.")


        stage_directory = find_stage_directory()
        source_directory = str((stage_directory / "src").resolve())
        if source_directory not in sys.path:
            sys.path.insert(0, source_directory)

        from predictor_audit import (
            analysis_categories,
            categorical_summary,
            categorical_target_profile,
            category_frequency_table,
            numeric_summary,
            numeric_target_summary,
            related_feature_summary,
            sentinel_mask,
            source_blank_mask,
            text_normalisation_summary,
        )
        from source_data_validation import (
            validate_aligned_ids,
            validate_label_frame,
            validate_raw_feature_schema,
        )

        data_directory = stage_directory / "data"
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

        feature = {feature['name']!r}
        feature_metadata = {feature!r}
        feature_types = {feature_types!r}
        assert feature in training_features.columns
        print(
            f"Validated {{len(training_features):,}} training rows and "
            f"{{len(test_features):,}} test rows for {{feature}}."
        )
        """,
        "load-and-validate",
    )


def checklist_markdown(feature):
    items = "\n".join(
        f"- {item}" for item in TYPE_CHECKLISTS[feature["audit_type"]]
    )
    return markdown(
        f"""
        ## Standard checks for `{feature['audit_type']}`

        Every predictor with this audit type receives the same basic checks:

        {items}

        This notebook stays descriptive. Target interpretation belongs in notebook 02,
        and cross-feature relationships belong in notebook 03.
        """,
        "type-checklist",
    )


def numeric_basic_cells(feature):
    sentinels = feature.get("sentinel_values", [])
    return [
        markdown("## Distribution, sentinels and coverage", "numeric-heading"),
        code(
            f"""
            sentinel_values = {sentinels!r}
            summary = numeric_summary(
                training_features,
                test_features,
                [feature],
                sentinel_values_by_column={{feature: sentinel_values}},
            )
            display(summary)
            """,
            "numeric-summary",
        ),
        code(
            """
            training_values = pd.to_numeric(training_features[feature], errors="coerce")
            test_values = pd.to_numeric(test_features[feature], errors="coerce")
            lower = training_values.quantile(0.01)
            upper = training_values.quantile(0.99)
            bins = np.linspace(lower, upper, 36)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
            axes[0].hist(training_values.clip(lower, upper), bins=bins, color="#2b6cb0")
            axes[0].set_title(f"{feature}: training (p01-p99 capped)")
            axes[1].hist(test_values.clip(lower, upper), bins=bins, color="#dd6b20")
            axes[1].set_title(f"{feature}: test on training bounds")
            for axis in axes:
                axis.set_xlabel(feature)
                axis.set_ylabel("rows")
            plt.tight_layout()
            plt.show()

            top_values = pd.DataFrame({
                "training rows": training_values.value_counts().head(12),
                "test rows": test_values.value_counts().head(12),
            }).fillna(0).astype(int)
            display(top_values)
            """,
            "numeric-distribution",
        ),
    ]


def date_basic_cells():
    return [
        markdown("## Parsing and calendar coverage", "date-heading"),
        code(
            """
            training_dates = pd.to_datetime(training_features[feature], errors="coerce")
            test_dates = pd.to_datetime(test_features[feature], errors="coerce")
            date_summary = pd.DataFrame({
                "rows": [len(training_dates), len(test_dates)],
                "invalid or missing": [training_dates.isna().sum(), test_dates.isna().sum()],
                "minimum": [training_dates.min(), test_dates.min()],
                "maximum": [training_dates.max(), test_dates.max()],
                "unique dates": [training_dates.nunique(), test_dates.nunique()],
                "years": [training_dates.dt.year.nunique(), test_dates.dt.year.nunique()],
            }, index=["training", "test"])
            display(date_summary)

            month_share = pd.DataFrame({
                "training (%)": training_dates.dt.month.value_counts(normalize=True).sort_index() * 100,
                "test (%)": test_dates.dt.month.value_counts(normalize=True).sort_index() * 100,
            }).fillna(0).round(2)
            display(month_share)
            month_share.plot(kind="bar", figsize=(10, 4), color=["#2b6cb0", "#dd6b20"])
            plt.ylabel("rows (%)")
            plt.title(f"{feature}: recording-month coverage")
            plt.tight_layout()
            plt.show()
            """,
            "date-summary",
        ),
    ]


def categorical_basic_cells(feature):
    sentinels = feature.get("sentinel_tokens", [])
    return [
        markdown("## Missingness, cardinality and category coverage", "category-heading"),
        code(
            f"""
            sentinel_tokens = {sentinels!r}
            overview = categorical_summary(
                training_features,
                test_features,
                [feature],
                rare_threshold=50,
                sentinel_tokens_by_column={{feature: sentinel_tokens}},
            )
            display(overview)
            display(
                category_frequency_table(
                    training_features,
                    test_features,
                    feature,
                    top_n=15,
                    sentinel_tokens=sentinel_tokens,
                )
            )
            """,
            "category-summary",
        ),
        code(
            """
            top = category_frequency_table(
                training_features,
                test_features,
                feature,
                top_n=12,
                sentinel_tokens=sentinel_tokens,
            )[["training (%)", "test (%)"]]
            top.plot(kind="bar", figsize=(12, 4), color=["#2b6cb0", "#dd6b20"])
            plt.ylabel("rows (%)")
            plt.title(f"{feature}: most common normalised values")
            plt.tight_layout()
            plt.show()
            """,
            "category-plot",
        ),
    ]


def build_basic_notebook(feature, catalogue):
    checklist = checklist_markdown(feature)
    cells = [
        markdown(
            f"""
            # `{feature['name']}` 01: basic `{feature['audit_type']}` breakdown

            **Purpose:** apply the analysis that is always performed for this feature
            type, without mixing in cross-feature interpretation.

            **Current role:** `{feature['role']}`

            **Provisional disposition:** {feature['disposition']}
            """,
            "basic-intro",
        ),
        checklist,
        common_load_cell(feature, catalogue),
    ]
    if feature["audit_type"] == "date":
        cells += date_basic_cells()
    elif feature["audit_type"] in {"numeric", "coordinate", "year"}:
        cells += numeric_basic_cells(feature)
    else:
        cells += categorical_basic_cells(feature)
    cells += [
        markdown(
            """
            ## Basic-breakdown handoff

            The outputs above establish shape, missingness, support and supplied
            train/test coverage. They do not by themselves establish usefulness or a
            causal explanation. Notebook 02 interprets noteworthy single-feature
            evidence; notebook 03 tests the explicitly related predictors.
            """,
            "basic-handoff",
        )
    ]
    return notebook(cells, feature=feature["name"], purpose="basic-breakdown")


def numeric_noteworthy_cells(feature):
    sentinels = feature.get("sentinel_values", [])
    return [
        code(
            f"""
            sentinel_values = {sentinels!r}
            display(
                numeric_target_summary(
                    training_data,
                    feature,
                    sentinel_values=sentinel_values,
                )
            )

            values = pd.to_numeric(training_data[feature], errors="coerce")
            state = pd.Series("observed", index=training_data.index, dtype="string")
            state.loc[values.isna()] = "missing"
            state.loc[values.isin(sentinel_values)] = "configured sentinel"
            state_profile = pd.crosstab(
                state,
                training_data["status_group"],
                normalize="index",
            ).mul(100).round(2)
            state_profile.insert(0, "rows", state.value_counts())
            display(state_profile)
            """,
            "numeric-target-evidence",
        )
    ]


def date_noteworthy_cells():
    return [
        code(
            """
            dates = pd.to_datetime(training_data[feature], errors="coerce")
            for component_name, component in (
                ("recording year", dates.dt.year),
                ("recording month", dates.dt.month),
            ):
                profile = pd.crosstab(
                    component,
                    training_data["status_group"],
                    normalize="index",
                ).mul(100).round(2)
                profile.insert(0, "rows", component.value_counts())
                print(component_name)
                display(profile)
            """,
            "date-target-evidence",
        )
    ]


def categorical_noteworthy_cells(feature):
    sentinels = feature.get("sentinel_tokens", [])
    return [
        code(
            f"""
            sentinel_tokens = {sentinels!r}
            target_profile = categorical_target_profile(
                training_data,
                feature,
                minimum_support=100,
                sentinel_tokens=sentinel_tokens,
            )
            display(target_profile.head(20))

            supported = target_profile.loc[target_profile["meets support threshold"]].copy()
            non_functional_column = "non functional (%)"
            if non_functional_column in supported:
                display(
                    supported.sort_values(non_functional_column, ascending=False)
                    .head(12)[["rows", non_functional_column]]
                )
            """,
            "category-target-evidence",
        )
    ]


def build_noteworthy_notebook(feature, catalogue):
    cells = [
        markdown(
            f"""
            # `{feature['name']}` 02: noteworthy single-feature findings

            **Purpose:** identify and discuss supported points that stand out after the
            standard `{feature['audit_type']}` breakdown. Target relationships here are
            exploratory and must be rechecked after the split is frozen.
            """,
            "noteworthy-intro",
        ),
        common_load_cell(feature, catalogue),
        markdown("## Supported target evidence", "target-heading"),
    ]
    if feature["audit_type"] == "date":
        cells += date_noteworthy_cells()
    elif feature["audit_type"] in {"numeric", "coordinate", "year"}:
        cells += numeric_noteworthy_cells(feature)
    else:
        cells += categorical_noteworthy_cells(feature)
    cells += [
        markdown(
            f"""
            ## Observation

            {feature['finding']}

            ## Interpretation

            The supported single-feature patterns make this field worth the stated
            treatment, but they do not prove causation or independent predictive value.
            High-cardinality and geographic fields are especially vulnerable to
            memorisation under a random split.

            ## Provisional decision

            {feature['decision']}

            **Risk to carry forward:** {feature['risk']}
            """,
            "finding-discussion",
        ),
        code(
            """
            decision_record = pd.DataFrame([{
                "feature": feature,
                "role": feature_metadata["role"],
                "disposition": feature_metadata["disposition"],
                "finding": feature_metadata["finding"],
                "decision": feature_metadata["decision"],
                "risk": feature_metadata["risk"],
            }])
            display(decision_record.set_index("feature"))
            """,
            "decision-record",
        ),
    ]
    return notebook(cells, feature=feature["name"], purpose="noteworthy-findings")


def special_relationship_cell(feature):
    name = feature["name"]
    if name in {"longitude", "latitude"}:
        return code(
            """
            usable_coordinates = (
                training_features["longitude"].between(28, 42)
                & training_features["latitude"].between(-13, 0)
            )
            coordinate_evidence = pd.DataFrame({
                "training rows": [len(training_features)],
                "usable coordinate rows": [usable_coordinates.sum()],
                "paired sentinel/out-of-bounds rows": [(~usable_coordinates).sum()],
                "paired sentinel/out-of-bounds (%)": [(~usable_coordinates).mean() * 100],
            }, index=["longitude + latitude"]).round(2)
            display(coordinate_evidence)
            """,
            "coordinate-pair-check",
        )
    if name in {"date_recorded", "construction_year"}:
        return code(
            """
            recording_year = pd.to_datetime(
                training_features["date_recorded"],
                errors="coerce",
            ).dt.year
            construction_year = pd.to_numeric(
                training_features["construction_year"],
                errors="coerce",
            )
            age = recording_year.sub(construction_year).where(construction_year.gt(0))
            age_check = pd.DataFrame({
                "known construction years": [construction_year.gt(0).sum()],
                "unknown year rows": [construction_year.eq(0).sum()],
                "negative derived ages": [age.lt(0).sum()],
                "median non-negative age": [age.where(age.ge(0)).median()],
                "maximum non-negative age": [age.where(age.ge(0)).max()],
            }, index=["date_recorded - construction_year"])
            display(age_check)
            """,
            "pump-age-check",
        )
    if name in {"payment", "payment_type"}:
        return code(
            """
            payment_mapping = {
                "pay annually": "annually",
                "pay monthly": "monthly",
                "pay per bucket": "per bucket",
                "pay when scheme fails": "on failure",
                "never pay": "never pay",
                "other": "other",
                "unknown": "unknown",
            }
            expected_payment_type = training_features["payment"].map(payment_mapping)
            payment_check = pd.DataFrame({
                "training rows": [len(training_features)],
                "mapped rows matching payment_type": [
                    expected_payment_type.eq(training_features["payment_type"]).sum()
                ],
                "mismatches": [
                    expected_payment_type.ne(training_features["payment_type"]).sum()
                ],
            }, index=["payment -> payment_type"])
            display(payment_check)
            """,
            "payment-alias-check",
        )
    if name in {"quantity", "quantity_group"}:
        return code(
            """
            quantity_check = pd.DataFrame({
                "training rows": [len(training_features)],
                "training equal rows": [
                    training_features["quantity"].eq(training_features["quantity_group"]).sum()
                ],
                "test rows": [len(test_features)],
                "test equal rows": [
                    test_features["quantity"].eq(test_features["quantity_group"]).sum()
                ],
            }, index=["quantity == quantity_group"])
            display(quantity_check)
            """,
            "quantity-duplicate-check",
        )
    if name == "recorded_by":
        return code(
            """
            recorder_check = pd.DataFrame({
                "training unique": [training_features[feature].nunique(dropna=False)],
                "test unique": [test_features[feature].nunique(dropna=False)],
                "training value": [training_features[feature].iloc[0]],
                "test value": [test_features[feature].iloc[0]],
            }, index=[feature])
            display(recorder_check)
            """,
            "constant-check",
        )
    return None


def build_related_notebook(feature, catalogue):
    relationships = feature["related"]
    relationship_lines = "\n".join(
        f"- `{item['feature']}` — {item['reason']}" for item in relationships
    )
    cells = [
        markdown(
            f"""
            # `{feature['name']}` 03: related-feature analysis

            **Purpose:** identify predictors that represent the same concept, form a
            hierarchy, share a missingness process or plausibly interact with
            `{feature['name']}`.

            ## Relationships selected in advance

            {relationship_lines}
            """,
            "related-intro",
        ),
        common_load_cell(feature, catalogue),
        code(
            """
            relationship_inventory = pd.DataFrame(feature_metadata["related"])
            display(relationship_inventory)

            relationship_evidence = related_feature_summary(
                training_features,
                feature,
                feature_metadata["audit_type"],
                feature_metadata["related"],
                feature_types,
            )
            display(relationship_evidence)
            """,
            "relationship-summary",
        ),
    ]
    special = special_relationship_cell(feature)
    if special is not None:
        cells.append(special)
    cells += [
        markdown(
            f"""
            ## Discussion and modelling consequence

            The relationships above were nominated before inspecting the pairwise
            coefficients. A strong association can mean useful interaction, hierarchy,
            shared collection behaviour or redundancy; it is not a reason to keep both
            fields automatically.

            For `{feature['name']}`, carry the relationships into controlled ablations
            and fit every learned grouping or encoding inside the training fold. The
            current provisional disposition remains: **{feature['disposition']}**.
            """,
            "relationship-discussion",
        )
    ]
    return notebook(cells, feature=feature["name"], purpose="related-features")


def build_overall_notebook(catalogue):
    features = catalogue["features"]
    return notebook(
        [
            markdown(
                """
                # Overall Pump It Up predictor audit

                This is the index and decision log for the raw predictor audit. Each of
                the 39 non-`id` raw columns has its own folder and three explicitly named
                notebooks: standard type-specific breakdown, noteworthy single-feature
                findings, and related-feature analysis.

                `status_group` remains a target rather than a predictor and has a separate
                role-named audit at `../00-target-label-analysis/00-target-label-analysis.ipynb`.
                """,
                "overall-intro",
            ),
            code(
                f"""
                import pandas as pd
                from pathlib import Path
                from IPython.display import display

                feature_catalogue = pd.DataFrame({features!r})
                display(feature_catalogue[[
                    "order",
                    "name",
                    "audit_type",
                    "role",
                    "disposition",
                ]])
                """,
                "catalogue",
            ),
            markdown("## Supplied data scope", "scope-heading"),
            code(
                """
                stage_directory = next(
                    candidate for candidate in [Path.cwd(), *Path.cwd().parents]
                    if (candidate / "data" / "TrainingSetValues.csv").is_file()
                )
                data_directory = stage_directory / "data"
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

                dataset_scope = pd.DataFrame({
                    "training rows": [len(training_features)],
                    "test rows": [len(test_features)],
                    "raw feature columns": [training_features.shape[1]],
                    "raw non-id predictors": [training_features.shape[1] - 1],
                    "candidate predictors": [feature_catalogue["role"].eq("candidate").sum()],
                    "structural removals": [
                        feature_catalogue["role"].eq("structural-removal").sum()
                    ],
                    "target classes": [training_labels["status_group"].nunique()],
                }, index=["supplied competition data"])
                display(dataset_scope)

                assert list(training_features.columns) == list(test_features.columns)
                assert training_features.shape[1] - 1 == len(feature_catalogue)
                """,
                "dataset-scope",
            ),
            markdown("## Coverage and audit contract", "coverage-heading"),
            code(
                """
                coverage = feature_catalogue.groupby(
                    ["role", "audit_type"],
                    dropna=False,
                ).size().rename("predictors").to_frame()
                display(coverage)
                assert len(feature_catalogue) == 39
                assert feature_catalogue["name"].is_unique
                print("Catalogue coverage: 39 of 39 raw non-id predictors.")
                """,
                "coverage",
            ),
            markdown(
                """
                ## Key findings and provisional decisions

                This register pulls the noteworthy single-feature conclusion back into
                one reviewable place. The detailed evidence remains in each predictor's
                `02` notebook and the relationship evidence remains in its `03` notebook.
                """,
                "findings-heading",
            ),
            code(
                """
                key_findings = feature_catalogue.set_index("name")[[
                    "audit_type",
                    "role",
                    "finding",
                    "decision",
                    "risk",
                ]]
                display(key_findings)
                """,
                "key-findings",
            ),
            markdown(
                """
                ## Structural removals are audited, not skipped

                Removal is a processing decision that needs evidence. The folders for
                `recorded_by`, `payment`, and `quantity_group` therefore contain the same
                three-part audit as retained candidates, including the cross-feature proof
                that supports each removal.
                """,
                "removal-heading",
            ),
            code(
                """
                structural_removals = feature_catalogue.loc[
                    feature_catalogue["role"].eq("structural-removal"),
                    ["name", "finding", "decision", "risk"],
                ]
                display(structural_removals.set_index("name"))
                """,
                "removals",
            ),
            markdown(
                """
                ## Cross-cutting conclusions

                - Preserve source blanks separately from literal tokens such as `None`,
                  `unknown` and `0`.
                - Treat numeric-looking administrative codes as categories.
                - Compare granular and coarse hierarchy levels by ablation rather than
                  retaining deterministic parents automatically.
                - Keep raw sparse names out of the first one-hot baseline.
                - Recheck geographic and target-rate findings after the split is frozen,
                  including an LGA/region-grouped robustness check.
                """,
                "overall-conclusions",
            ),
        ],
        purpose="overall-data-audit",
    )


def build_readme(catalogue):
    lines = [
        "# Pump It Up data audit",
        "",
        "This directory is the canonical home of the data audit.",
        "",
        "Start with the role-based project-level analyses:",
        "",
        "1. [`00-target-label-analysis/`](00-target-label-analysis/) contains the dedicated target-label analysis. The folder and notebook use the generic modelling role rather than the dataset-specific `status_group` column name.",
        "2. [`00-overall/`](00-overall/) contains the dataset scope, overall predictor finding/decision register and preserved feature-family deep dives.",
        "",
        "Every raw non-`id` predictor then has a CSV-order-prefixed folder.",
        "",
        "Each predictor folder contains:",
        "",
        "1. `01-<feature>-basic-breakdown.ipynb` — the analysis always applied to that feature type.",
        "2. `02-<feature>-noteworthy-findings.ipynb` — supported observations, interpretation and decision.",
        "3. `03-<feature>-related-features.ipynb` — nominated relationships and pairwise diagnostics.",
        "",
        "The three structurally removed columns are intentionally included: this is where",
        "the evidence for removal is recorded before preprocessing.",
        "",
        "## Predictor index",
        "",
        "| # | Predictor | Type | Role | Folder |",
        "|---:|---|---|---|---|",
    ]
    for feature in catalogue["features"]:
        folder = folder_name(feature)
        lines.append(
            f"| {feature['order']} | `{feature['name']}` | `{feature['audit_type']}` | "
            f"`{feature['role']}` | [`{folder}/`]({folder}/) |"
        )
    lines += [
        "",
        "## Rebuild",
        "",
        "Run `generate_data_audit_notebooks.py` from the project environment, then",
        "execute the generated notebooks before treating their recorded outputs as current.",
        "The catalogue in `../../src/predictor_audit_catalogue.json` is the single source",
        "of truth for ordering, audit types, dispositions and related-feature selections.",
        "",
    ]
    return "\n".join(lines)


def write_notebook(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(value, path)


def main():
    catalogue = load_catalogue()
    overall_directory = DATA_AUDIT_DIRECTORY / "00-overall"
    write_notebook(
        overall_directory / "00-overall-data-audit.ipynb",
        build_overall_notebook(catalogue),
    )

    written = 1
    for feature in catalogue["features"]:
        directory = DATA_AUDIT_DIRECTORY / folder_name(feature)
        write_notebook(
            directory / notebook_filename(feature, 1, "basic-breakdown"),
            build_basic_notebook(feature, catalogue),
        )
        write_notebook(
            directory / notebook_filename(feature, 2, "noteworthy-findings"),
            build_noteworthy_notebook(feature, catalogue),
        )
        write_notebook(
            directory / notebook_filename(feature, 3, "related-features"),
            build_related_notebook(feature, catalogue),
        )
        written += 3

    (DATA_AUDIT_DIRECTORY / "README.md").write_text(
        build_readme(catalogue),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {written} canonical notebooks for 39 predictors plus the overall audit.")


if __name__ == "__main__":
    main()
