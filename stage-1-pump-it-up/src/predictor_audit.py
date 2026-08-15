"""Reusable summaries for the Pump It Up predictor-audit notebooks.

The functions in this module describe supplied training and test data. They do
not fit preprocessing state or make feature-selection decisions. Any learned
transformation still belongs inside a later training fold.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_SENTINEL_TOKENS = frozenset({"none", "unknown", "not known"})
MISSING_CATEGORY = "<missing/blank>"


def normalise_categories(series: pd.Series) -> pd.Series:
    """Return trimmed, case-folded strings while preserving missing values."""

    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
        .replace("", pd.NA)
    )


def sentinel_mask(
    series: pd.Series,
    sentinel_tokens: Iterable[str] = DEFAULT_SENTINEL_TOKENS,
) -> pd.Series:
    """Identify configured categorical sentinel tokens after light normalisation."""

    tokens = {str(token).strip().casefold() for token in sentinel_tokens}
    return normalise_categories(series).isin(tokens).fillna(False)


def source_blank_mask(series: pd.Series) -> pd.Series:
    """Identify blank strings preserved from source CSV cells."""

    return series.astype("string").str.strip().eq("").fillna(False)


def analysis_categories(
    series: pd.Series,
    sentinel_tokens: Iterable[str] = DEFAULT_SENTINEL_TOKENS,
    *,
    collapse_sentinels: bool = False,
) -> pd.Series:
    """Normalise categories while keeping semantic sentinels visible by default."""

    normalised = normalise_categories(series)
    if collapse_sentinels:
        normalised = normalised.mask(sentinel_mask(series, sentinel_tokens))
    return normalised.fillna(MISSING_CATEGORY)


def categorical_summary(
    training: pd.DataFrame,
    test: pd.DataFrame,
    columns: Iterable[str],
    *,
    rare_threshold: int = 20,
    sentinel_tokens_by_column: Mapping[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    """Summarise quality, coverage and marginal drift for categorical columns."""

    token_overrides = sentinel_tokens_by_column or {}
    records: list[dict[str, object]] = []

    for column in columns:
        tokens = token_overrides.get(column, DEFAULT_SENTINEL_TOKENS)
        training_values = analysis_categories(training[column], tokens)
        test_values = analysis_categories(test[column], tokens)
        training_counts = training_values.value_counts(dropna=False)
        test_counts = test_values.value_counts(dropna=False)
        training_levels = set(training_counts.index) - {MISSING_CATEGORY}
        test_levels = set(test_counts.index) - {MISSING_CATEGORY}
        rare_levels = set(
            training_counts.loc[
                (training_counts < rare_threshold)
                & training_counts.index.to_series().ne(MISSING_CATEGORY).to_numpy()
            ].index
        )
        test_only_levels = test_levels - training_levels
        training_only_levels = training_levels - test_levels

        all_levels = sorted(set(training_counts.index) | set(test_counts.index))
        training_share = training_counts.reindex(all_levels, fill_value=0) / len(training)
        test_share = test_counts.reindex(all_levels, fill_value=0) / len(test)
        total_variation = 0.5 * float((training_share - test_share).abs().sum())

        records.append(
            {
                "feature": column,
                "training explicit missing": int(training[column].isna().sum()),
                "training source blank rows": int(source_blank_mask(training[column]).sum()),
                "training sentinel rows": int(sentinel_mask(training[column], tokens).sum()),
                "test explicit missing": int(test[column].isna().sum()),
                "test source blank rows": int(source_blank_mask(test[column]).sum()),
                "test sentinel rows": int(sentinel_mask(test[column], tokens).sum()),
                "training levels": len(training_levels),
                "test levels": len(test_levels),
                f"training levels with <{rare_threshold} rows": len(rare_levels),
                "training rows in rare levels (%)": round(
                    float(training_values.isin(rare_levels).mean() * 100), 2
                ),
                "test-only levels": len(test_only_levels),
                "test rows in unseen levels (%)": round(
                    float(test_values.isin(test_only_levels).mean() * 100), 2
                ),
                "training-only levels": len(training_only_levels),
                "marginal total-variation distance": round(total_variation, 4),
            }
        )

    return pd.DataFrame.from_records(records).set_index("feature")


def numeric_summary(
    training: pd.DataFrame,
    test: pd.DataFrame,
    columns: Iterable[str],
    *,
    sentinel_values_by_column: Mapping[str, Iterable[float]] | None = None,
) -> pd.DataFrame:
    """Return aligned distribution summaries for numeric predictors."""

    sentinel_overrides = sentinel_values_by_column or {}
    records: list[dict[str, object]] = []

    for column in columns:
        sentinels = set(sentinel_overrides.get(column, ()))
        for frame_name, frame in (("training", training), ("test", test)):
            values = pd.to_numeric(frame[column], errors="coerce")
            non_missing = values.dropna()
            records.append(
                {
                    "feature": column,
                    "frame": frame_name,
                    "rows": len(values),
                    "missing": int(values.isna().sum()),
                    "sentinel rows": int(values.isin(sentinels).sum()),
                    "zero rows": int(values.eq(0).sum()),
                    "negative rows": int(values.lt(0).sum()),
                    "unique": int(non_missing.nunique()),
                    "minimum": non_missing.min(),
                    "p01": non_missing.quantile(0.01),
                    "median": non_missing.median(),
                    "mean": non_missing.mean(),
                    "p99": non_missing.quantile(0.99),
                    "maximum": non_missing.max(),
                    "skewness": non_missing.skew(),
                }
            )

    result = pd.DataFrame.from_records(records).set_index(["feature", "frame"])
    numeric_columns = result.select_dtypes(include="number").columns
    result[numeric_columns] = result[numeric_columns].round(3)
    return result


def numeric_target_summary(
    training_data: pd.DataFrame,
    column: str,
    *,
    target: str = "status_group",
    sentinel_values: Iterable[float] = (),
) -> pd.DataFrame:
    """Describe a numeric feature by target class without imposing class order."""

    values = pd.to_numeric(training_data[column], errors="coerce")
    sentinels = set(sentinel_values)
    working = pd.DataFrame({column: values, target: training_data[target]})
    groups = working.groupby(target, observed=True)[column]
    result = groups.agg(rows="size", missing="count", median="median", mean="mean")
    result["missing"] = result["rows"] - result["missing"]
    result["sentinel rows"] = groups.apply(lambda group: int(group.isin(sentinels).sum()))
    result["zero rows (%)"] = groups.apply(lambda group: float(group.eq(0).mean() * 100))
    result["p90"] = groups.quantile(0.90)
    return result.round(3)


def categorical_target_profile(
    training_data: pd.DataFrame,
    column: str,
    *,
    target: str = "status_group",
    minimum_support: int = 100,
    sentinel_tokens: Iterable[str] = DEFAULT_SENTINEL_TOKENS,
) -> pd.DataFrame:
    """Return support-aware class percentages for each auditable category."""

    categories = analysis_categories(training_data[column], sentinel_tokens)
    counts = pd.crosstab(categories, training_data[target], dropna=False)
    result = counts.copy()
    result.insert(0, "rows", counts.sum(axis=1))
    for status in counts.columns:
        result[f"{status} (%)"] = counts[status].div(result["rows"]).mul(100)
        result = result.drop(columns=status)
    result.insert(1, "meets support threshold", result["rows"].ge(minimum_support))
    result = result.sort_values(["meets support threshold", "rows"], ascending=False)
    percentage_columns = [column for column in result if column.endswith("(%)")]
    result[percentage_columns] = result[percentage_columns].round(2)
    return result


def category_frequency_table(
    training: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    *,
    top_n: int = 12,
    sentinel_tokens: Iterable[str] = DEFAULT_SENTINEL_TOKENS,
) -> pd.DataFrame:
    """Compare the most common normalised categories in training and test."""

    training_values = analysis_categories(training[column], sentinel_tokens)
    test_values = analysis_categories(test[column], sentinel_tokens)
    training_counts = training_values.value_counts(dropna=False)
    test_counts = test_values.value_counts(dropna=False)
    levels = list(training_counts.head(top_n).index)
    for level in test_counts.head(top_n).index:
        if level not in levels:
            levels.append(level)
    result = pd.DataFrame(
        {
            "training rows": training_counts.reindex(levels, fill_value=0),
            "training (%)": training_counts.reindex(levels, fill_value=0)
            .div(len(training))
            .mul(100),
            "test rows": test_counts.reindex(levels, fill_value=0),
            "test (%)": test_counts.reindex(levels, fill_value=0).div(len(test)).mul(100),
        }
    )
    return result.round(2)


def hierarchy_summary(
    training: pd.DataFrame,
    test: pd.DataFrame,
    relationships: Iterable[tuple[str, str]],
    *,
    sentinel_tokens: Iterable[str] = DEFAULT_SENTINEL_TOKENS,
) -> pd.DataFrame:
    """Check whether each child category maps deterministically to its parent."""

    records: list[dict[str, object]] = []
    for child, parent in relationships:
        for frame_name, frame in (("training", training), ("test", test)):
            pairs = pd.DataFrame(
                {
                    "child": analysis_categories(frame[child], sentinel_tokens),
                    "parent": analysis_categories(frame[parent], sentinel_tokens),
                }
            )
            pairs = pairs.loc[
                pairs["child"].ne(MISSING_CATEGORY)
                & pairs["parent"].ne(MISSING_CATEGORY)
            ]
            parents_per_child = pairs.groupby("child", observed=True)["parent"].nunique()
            ambiguous_children = set(parents_per_child.loc[parents_per_child.gt(1)].index)
            children_per_parent = pairs.groupby("parent", observed=True)["child"].nunique()
            records.append(
                {
                    "relationship": f"{child} -> {parent}",
                    "frame": frame_name,
                    "complete rows": len(pairs),
                    "child levels": int(parents_per_child.size),
                    "parent levels": int(children_per_parent.size),
                    "ambiguous child levels": len(ambiguous_children),
                    "rows in ambiguous child levels": int(
                        pairs["child"].isin(ambiguous_children).sum()
                    ),
                    "deterministic child-to-parent": len(ambiguous_children) == 0,
                    "one-to-one level mapping": bool(
                        len(ambiguous_children) == 0 and children_per_parent.le(1).all()
                    ),
                }
            )

    return pd.DataFrame.from_records(records).set_index(["relationship", "frame"])


def hierarchy_conflicts(
    frame: pd.DataFrame,
    child: str,
    parent: str,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """List supported child levels that map to more than one parent level."""

    pairs = pd.DataFrame(
        {
            "child": analysis_categories(frame[child]),
            "parent": analysis_categories(frame[parent]),
        }
    )
    parent_counts = pairs.groupby("child", observed=True)["parent"].nunique()
    ambiguous = parent_counts.loc[parent_counts.gt(1)].index
    if len(ambiguous) == 0:
        return pd.DataFrame(columns=["rows", "parent levels", "parents"])
    details = pairs.loc[pairs["child"].isin(ambiguous)].groupby("child", observed=True)
    result = details.agg(
        rows=("parent", "size"),
        **{
            "parent levels": ("parent", "nunique"),
            "parents": ("parent", lambda values: ", ".join(sorted(set(values)))),
        },
    )
    return result.sort_values(["rows", "parent levels"], ascending=False).head(top_n)


def text_normalisation_summary(
    training: pd.DataFrame,
    test: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Quantify cardinality removed by safe whitespace/case normalisation."""

    records: list[dict[str, object]] = []
    for column in columns:
        for frame_name, frame in (("training", training), ("test", test)):
            raw = frame[column].astype("string").str.strip().replace("", pd.NA)
            normalised = normalise_categories(raw)
            raw_unique = int(raw.dropna().nunique())
            normalised_unique = int(normalised.dropna().nunique())
            counts = normalised.value_counts()
            records.append(
                {
                    "feature": column,
                    "frame": frame_name,
                    "raw levels": raw_unique,
                    "normalised levels": normalised_unique,
                    "levels collapsed": raw_unique - normalised_unique,
                    "singleton levels": int(counts.eq(1).sum()),
                    "rows in singleton levels (%)": round(
                        float(normalised.isin(counts.loc[counts.eq(1)].index).mean() * 100),
                        2,
                    ),
                }
            )
    return pd.DataFrame.from_records(records).set_index(["feature", "frame"])


def cramer_v(table: pd.DataFrame) -> float:
    """Return bias-corrected Cramer's V for a contingency table."""

    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total == 0 or min(observed.shape) < 2:
        return float("nan")
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / total
    valid = expected > 0
    chi_squared = float((((observed - expected) ** 2 / expected)[valid]).sum())
    phi_squared = chi_squared / total
    rows, columns = observed.shape
    correction = ((columns - 1) * (rows - 1)) / max(total - 1, 1)
    corrected_phi = max(0.0, phi_squared - correction)
    corrected_rows = rows - ((rows - 1) ** 2) / max(total - 1, 1)
    corrected_columns = columns - ((columns - 1) ** 2) / max(total - 1, 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return float(np.sqrt(corrected_phi / denominator)) if denominator > 0 else float("nan")


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Return eta for a categorical/numeric relationship."""

    working = pd.DataFrame(
        {
            "category": analysis_categories(categories),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna(subset=["value"])
    if working.empty or working["category"].nunique() < 2:
        return float("nan")
    grand_mean = float(working["value"].mean())
    groups = working.groupby("category", observed=True)["value"]
    between = float(
        sum(len(group) * (float(group.mean()) - grand_mean) ** 2 for _, group in groups)
    )
    total = float(((working["value"] - grand_mean) ** 2).sum())
    return float(np.sqrt(between / total)) if total > 0 else float("nan")


def _as_relationship_numeric(series: pd.Series, audit_type: str) -> pd.Series:
    if audit_type == "date":
        dates = pd.to_datetime(series, errors="coerce")
        return dates.map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
    return pd.to_numeric(series, errors="coerce")


def _is_numeric_relationship_type(audit_type: str) -> bool:
    return audit_type in {"numeric", "coordinate", "year", "date"}


def pairwise_relationship_summary(
    frame: pd.DataFrame,
    primary: str,
    related: str,
    *,
    primary_type: str,
    related_type: str,
) -> dict[str, object]:
    """Describe a predictor pair using a measure appropriate to their audit types."""

    primary_is_numeric = _is_numeric_relationship_type(primary_type)
    related_is_numeric = _is_numeric_relationship_type(related_type)
    common = frame[[primary, related]].copy()

    if primary_is_numeric and related_is_numeric:
        primary_values = _as_relationship_numeric(common[primary], primary_type)
        related_values = _as_relationship_numeric(common[related], related_type)
        valid = primary_values.notna() & related_values.notna()
        coefficient = primary_values.loc[valid].corr(
            related_values.loc[valid],
            method="spearman",
        )
        return {
            "primary": primary,
            "related": related,
            "measure": "Spearman correlation",
            "association": round(float(coefficient), 4) if pd.notna(coefficient) else np.nan,
            "complete rows": int(valid.sum()),
            "primary levels": int(primary_values.loc[valid].nunique()),
            "related levels": int(related_values.loc[valid].nunique()),
            "forward modal purity (%)": np.nan,
            "reverse modal purity (%)": np.nan,
        }

    if not primary_is_numeric and not related_is_numeric:
        primary_values = analysis_categories(common[primary])
        related_values = analysis_categories(common[related])
        table = pd.crosstab(primary_values, related_values, dropna=False)
        forward_purity = table.max(axis=1).sum() / table.to_numpy().sum() * 100
        reverse_purity = table.max(axis=0).sum() / table.to_numpy().sum() * 100
        return {
            "primary": primary,
            "related": related,
            "measure": "bias-corrected Cramer's V",
            "association": round(cramer_v(table), 4),
            "complete rows": len(common),
            "primary levels": int(table.shape[0]),
            "related levels": int(table.shape[1]),
            "forward modal purity (%)": round(float(forward_purity), 2),
            "reverse modal purity (%)": round(float(reverse_purity), 2),
        }

    if primary_is_numeric:
        numeric_values = _as_relationship_numeric(common[primary], primary_type)
        category_values = common[related]
    else:
        numeric_values = _as_relationship_numeric(common[related], related_type)
        category_values = common[primary]
    valid = numeric_values.notna()
    return {
        "primary": primary,
        "related": related,
        "measure": "correlation ratio (eta)",
        "association": round(
            correlation_ratio(category_values.loc[valid], numeric_values.loc[valid]),
            4,
        ),
        "complete rows": int(valid.sum()),
        "primary levels": int(common.loc[valid, primary].nunique(dropna=False)),
        "related levels": int(common.loc[valid, related].nunique(dropna=False)),
        "forward modal purity (%)": np.nan,
        "reverse modal purity (%)": np.nan,
    }


def related_feature_summary(
    frame: pd.DataFrame,
    primary: str,
    primary_type: str,
    related_features: Iterable[Mapping[str, str]],
    feature_types: Mapping[str, str],
) -> pd.DataFrame:
    """Return aligned pairwise diagnostics for a catalogue-defined relationship set."""

    records = []
    for relationship in related_features:
        related = relationship["feature"]
        record = pairwise_relationship_summary(
            frame,
            primary,
            related,
            primary_type=primary_type,
            related_type=feature_types[related],
        )
        record["relationship rationale"] = relationship["reason"]
        records.append(record)
    return pd.DataFrame.from_records(records)
