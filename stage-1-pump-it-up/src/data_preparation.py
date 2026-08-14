"""Structural data preparation for the Pump It Up competition data.

This module applies only fixed, evidence-backed structural rules. It does not
learn statistics, impute values, encode categories or otherwise fit transforms.
"""

# C#-brain note to future me: this postpones evaluation of type annotations.
# It is roughly the difference between retaining type metadata as text and
# resolving every referenced type while the module is being imported. It also
# lets annotations refer to types that are declared later in the file.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd


# `Final` is guidance for the type checker, not a runtime `const`. The tuple is
# the important second half of the intent: unlike a list, it has no append or
# item-assignment operations. I am using it as an ordered, readonly-ish schema.
SOURCE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "amount_tsh",
    "date_recorded",
    "funder",
    "gps_height",
    "installer",
    "longitude",
    "latitude",
    "wpt_name",
    "num_private",
    "basin",
    "subvillage",
    "region",
    "region_code",
    "district_code",
    "lga",
    "ward",
    "population",
    "public_meeting",
    "recorded_by",
    "scheme_management",
    "scheme_name",
    "permit",
    "construction_year",
    "extraction_type",
    "extraction_type_group",
    "extraction_type_class",
    "management",
    "management_group",
    "payment",
    "payment_type",
    "water_quality",
    "quality_group",
    "quantity",
    "quantity_group",
    "source",
    "source_type",
    "source_class",
    "waterpoint_type",
    "waterpoint_type_group",
)

ID_COLUMN: Final = "id"
TARGET_COLUMN: Final = "status_group"
RECORDED_BY_VALUE: Final = "GeoData Consultants Ltd"

PAYMENT_TO_PAYMENT_TYPE: Final[dict[str, str]] = {
    "pay annually": "annually",
    "pay monthly": "monthly",
    "pay per bucket": "per bucket",
    "pay when scheme fails": "on failure",
    "never pay": "never pay",
    "other": "other",
    "unknown": "unknown",
}

DROPPED_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "recorded_by",
    "payment",
    "quantity_group",
)

# Read this like a small LINQ query followed by `ToArray()`:
# SOURCE_FEATURE_COLUMNS.Where(column => !excluded.Contains(column)).
# Python calls the inner form a generator expression; `tuple(...)` consumes it.
PREDICTOR_COLUMNS: Final[tuple[str, ...]] = tuple(
    column
    for column in SOURCE_FEATURE_COLUMNS
    if column not in {ID_COLUMN, *DROPPED_FEATURE_COLUMNS}
)

DATA_FILES: Final[tuple[str, ...]] = (
    "TrainingSetValues.csv",
    "TrainingSetLabels.csv",
    "TestSetValues.csv",
    "SubmissionFormat.csv",
)


class DataPreparationError(ValueError):
    """Raised when source data breaks an agreed structural contract."""


# A frozen dataclass is the nearest thing here to a small immutable C# record:
# Python generates the constructor, equality and representation for me. The
# freeze is shallow, so it prevents rebinding fields but would not freeze a
# mutable object stored inside one.
@dataclass(frozen=True)
class ColumnAction:
    """A structural treatment applied to a source column."""

    column: str
    action: str
    reason: str


COLUMN_ACTIONS: Final[tuple[ColumnAction, ...]] = (
    ColumnAction(
        ID_COLUMN,
        "metadata",
        "Preserved for joins and submissions but excluded from predictors.",
    ),
    ColumnAction(
        "quantity_group",
        "dropped",
        "Duplicates quantity row for row.",
    ),
    ColumnAction(
        "payment",
        "dropped",
        "Is a one-to-one relabelling of payment_type.",
    ),
    ColumnAction(
        "recorded_by",
        "dropped",
        f"Contains only the constant value {RECORDED_BY_VALUE!r}.",
    ),
)


@dataclass(frozen=True)
class PreparationReport:
    """Summary of a structural preparation run."""

    dataset_name: str
    row_count: int
    source_column_count: int
    predictor_column_count: int
    column_actions: tuple[ColumnAction, ...] = COLUMN_ACTIONS

    def to_frame(self) -> pd.DataFrame:
        """Return the structural actions as a display-friendly data frame."""

        # The parenthesised expression is a lazy sequence, much like
        # `IEnumerable<Dictionary<string, object>>`. DataFrame enumerates it and
        # turns each dictionary into one row, using the keys as column names.
        return pd.DataFrame(
            (
                {
                    "column": item.column,
                    "action": item.action,
                    "reason": item.reason,
                }
                for item in self.column_actions
            )
        )


@dataclass(frozen=True)
class CompetitionFrames:
    """The four unmodified CSV data frames supplied for the competition."""

    training_values: pd.DataFrame
    training_labels: pd.DataFrame
    test_values: pd.DataFrame
    submission_format: pd.DataFrame


@dataclass(frozen=True)
class PreparedFeatures:
    """Predictors and identifiers prepared from one feature data set."""

    predictors: pd.DataFrame
    ids: pd.Series
    report: PreparationReport


@dataclass(frozen=True)
class PreparedTrainingData:
    """Prepared training predictors, identifiers and aligned target labels."""

    predictors: pd.DataFrame
    ids: pd.Series
    target: pd.Series
    report: PreparationReport


@dataclass(frozen=True)
class PreparedCompetitionData:
    """Prepared training and test data with a shared predictor schema."""

    training: PreparedTrainingData
    test: PreparedFeatures


def load_competition_data(data_directory: str | Path) -> CompetitionFrames:
    """Load the four canonical competition CSVs from a local data directory."""

    # `Path` is my `FileInfo`/`DirectoryInfo` mental model. The `/` operator used
    # below is deliberately overloaded for path joining, i.e. `Path.Combine`.
    directory = Path(data_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Competition data directory not found: {directory}")

    # This dictionary comprehension is the compact Python equivalent of
    # `DATA_FILES.ToDictionary(name => name, name => directory / name)`.
    paths = {filename: directory / filename for filename in DATA_FILES}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Competition data directory is missing: {missing_names}."
        )

    # `read_csv` materialises each file as a mutable DataFrame. A DataFrame is
    # best thought of as an in-memory table whose columns are Series objects;
    # unlike a `DataTable`, its row labels (the index) participate in alignment.
    frames = {
        filename: pd.read_csv(path)
        for filename, path in paths.items()
    }

    return CompetitionFrames(
        training_values=frames["TrainingSetValues.csv"],
        training_labels=frames["TrainingSetLabels.csv"],
        test_values=frames["TestSetValues.csv"],
        submission_format=frames["SubmissionFormat.csv"],
    )


def prepare_features(
    source: pd.DataFrame,
    *,
    dataset_name: str = "features",
) -> PreparedFeatures:
    """Validate and structurally prepare a training or test feature frame."""

    # The bare `*` makes everything after it keyword-only. It is the Python way
    # of making the call site say `dataset_name=...`, which saves me having to
    # remember what an otherwise anonymous second string argument meant.
    _require_data_frame(source, dataset_name)
    _validate_exact_columns(source, SOURCE_FEATURE_COLUMNS, dataset_name)
    _validate_ids(source[ID_COLUMN], dataset_name)
    _validate_quantity_duplicate(source, dataset_name)
    _validate_payment_mapping(source, dataset_name)
    _validate_recorded_by(source, dataset_name)

    # `.loc[rows, columns]` means all rows (`:`), then these named columns in
    # exactly this order. Pandas can return views in some selections, so the
    # explicit deep copy is the reference-semantics firewall: later notebook
    # edits cannot mutate the caller-owned frame through this result.
    predictors = source.loc[:, PREDICTOR_COLUMNS].copy(deep=True)
    ids = source[ID_COLUMN].copy(deep=True)

    report = PreparationReport(
        dataset_name=dataset_name,
        row_count=len(source),
        # `shape` is `(row_count, column_count)`, hence index 1 here. This tuple
        # convention is NumPy/pandas muscle memory rather than a named property.
        source_column_count=source.shape[1],
        predictor_column_count=predictors.shape[1],
    )

    return PreparedFeatures(predictors=predictors, ids=ids, report=report)


def prepare_training_data(
    training_values: pd.DataFrame,
    training_labels: pd.DataFrame,
    *,
    dataset_name: str = "training",
    target_column: str = TARGET_COLUMN,
) -> PreparedTrainingData:
    """Prepare training features and align labels to feature row order by ID."""

    prepared = prepare_features(training_values, dataset_name=dataset_name)
    labels_name = f"{dataset_name} labels"

    _require_data_frame(training_labels, labels_name)
    _validate_exact_columns(
        training_labels,
        (ID_COLUMN, target_column),
        labels_name,
    )
    _validate_ids(training_labels[ID_COLUMN], labels_name)

    if training_labels[target_column].isna().any():
        missing_count = int(training_labels[target_column].isna().sum())
        raise DataPreparationError(
            f"{labels_name} contains {missing_count} missing "
            f"{target_column!r} value(s)."
        )

    # A pandas Index is not just a row number; it is a set-like collection of
    # labels used for alignment. These `difference` calls are therefore my
    # keyed integrity check before I attempt to line the target up with features.
    values_ids = pd.Index(prepared.ids)
    label_ids = pd.Index(training_labels[ID_COLUMN])
    missing_labels = values_ids.difference(label_ids)
    unexpected_labels = label_ids.difference(values_ids)
    if not missing_labels.empty or not unexpected_labels.empty:
        raise DataPreparationError(
            f"{labels_name} IDs do not match {dataset_name} feature IDs; "
            f"missing labels={len(missing_labels)}, "
            f"unexpected labels={len(unexpected_labels)}."
        )

    # Think `ToDictionary(label => label.Id)` followed by projecting values in
    # the feature-ID order. `set_index` makes ID the lookup key; `reindex` asks
    # for that exact order. Missing keys would become NaN, which is why I prove
    # the two ID sets match immediately above rather than silently accepting it.
    labels_by_id = training_labels.set_index(ID_COLUMN)[target_column]
    aligned_target = labels_by_id.reindex(prepared.ids.to_numpy()).copy(deep=True)
    # Reindexing used IDs as temporary row labels. Restore the feature frame's
    # original index so pandas will align target and predictors row for row later.
    aligned_target.index = training_values.index.copy()
    aligned_target.name = target_column

    return PreparedTrainingData(
        predictors=prepared.predictors,
        ids=prepared.ids,
        target=aligned_target,
        report=prepared.report,
    )


def prepare_competition_data(
    frames: CompetitionFrames,
) -> PreparedCompetitionData:
    """Prepare both feature sets and confirm their ordered schemas match."""

    training = prepare_training_data(
        frames.training_values,
        frames.training_labels,
    )
    test = prepare_features(frames.test_values, dataset_name="test")

    # Tuple equality checks both content and order, unlike comparing two sets.
    # That matters because downstream transformers will treat column position as
    # part of the contract even when the same names exist in a different order.
    training_schema = tuple(training.predictors.columns)
    test_schema = tuple(test.predictors.columns)
    if training_schema != test_schema:
        raise DataPreparationError(
            "Prepared training and test predictors have different ordered schemas."
        )

    _validate_submission_format(frames.submission_format, test.ids)

    return PreparedCompetitionData(training=training, test=test)


def load_and_prepare_competition_data(
    data_directory: str | Path,
) -> PreparedCompetitionData:
    """Load and structurally prepare the complete competition data set."""

    return prepare_competition_data(load_competition_data(data_directory))


def _require_data_frame(value: object, dataset_name: str) -> None:
    # Leading underscores mean "private implementation detail" by convention.
    # Python does not enforce this like C# access modifiers do.
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{dataset_name} must be a pandas DataFrame.")


def _validate_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    dataset_name: str,
) -> None:
    # `duplicated()` returns one bool per column label. Feeding that boolean
    # Series back into `frame.columns[...]` is vectorised filtering, comparable
    # to a LINQ `Where`, but executed by pandas rather than a Python loop.
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise DataPreparationError(
            f"{dataset_name} contains duplicate column labels: {duplicated!r}."
        )

    # Converting the Index to a tuple gives ordinary value-and-order equality.
    actual = tuple(frame.columns)
    if actual == expected:
        return

    missing = [column for column in expected if column not in actual]
    unexpected = [column for column in actual if column not in expected]
    details = []
    if missing:
        details.append(f"missing={missing!r}")
    if unexpected:
        details.append(f"unexpected={unexpected!r}")
    if not missing and not unexpected:
        details.append("column order has changed")

    raise DataPreparationError(
        f"{dataset_name} does not match the expected ordered schema: "
        + "; ".join(details)
        + "."
    )


def _validate_ids(ids: pd.Series, dataset_name: str) -> None:
    # These methods operate on the whole column and return Series/scalars. In C#
    # terms, read the first line as roughly `ids.Count(x => x is null)`—without
    # writing a Python loop. `int(...)` converts NumPy's integer scalar to a
    # plain Python integer for predictable messages and dataclass values.
    missing_count = int(ids.isna().sum())
    duplicate_count = int(ids.duplicated().sum())
    if missing_count or duplicate_count:
        raise DataPreparationError(
            f"{dataset_name} IDs must be non-null and unique; "
            f"missing={missing_count}, duplicates={duplicate_count}."
        )


def _validate_quantity_duplicate(frame: pd.DataFrame, dataset_name: str) -> None:
    quantity = frame["quantity"]
    quantity_group = frame["quantity_group"]
    # `eq`, `isna`, `|` and `&` are element-by-element operations. Python's
    # scalar `or`/`and` cannot combine whole Series. The explicit missing-value
    # clause also matters because, like SQL NULL, NaN does not equal itself.
    matches = quantity.eq(quantity_group) | (
        quantity.isna() & quantity_group.isna()
    )

    if not bool(matches.all()):
        mismatch_count = int((~matches).sum())
        raise DataPreparationError(
            f"{dataset_name} quantity_group no longer duplicates quantity; "
            f"{mismatch_count} row(s) differ."
        )


def _validate_payment_mapping(frame: pd.DataFrame, dataset_name: str) -> None:
    payment = frame["payment"]
    # `~` is element-wise boolean negation here—the Series equivalent of putting
    # `!` in front of every result. `isin` is the vectorised `Contains` check.
    unknown_payment = ~payment.isin(PAYMENT_TO_PAYMENT_TYPE)
    if bool(unknown_payment.any()):
        unexpected = payment.loc[unknown_payment].drop_duplicates().tolist()
        raise DataPreparationError(
            f"{dataset_name} contains payment values outside the agreed mapping: "
            f"{unexpected!r}."
        )

    # `map(dictionary)` replaces every Series value via that dictionary. My C#
    # translation is `payment.Select(value => mapping[value])`, except pandas
    # preserves the original row index so the comparison remains aligned.
    expected_payment_type = payment.map(PAYMENT_TO_PAYMENT_TYPE)
    matches = expected_payment_type.eq(frame["payment_type"])
    if not bool(matches.all()):
        mismatch_count = int((~matches).sum())
        raise DataPreparationError(
            f"{dataset_name} payment no longer maps one-to-one to payment_type; "
            f"{mismatch_count} row(s) differ."
        )


def _validate_recorded_by(frame: pd.DataFrame, dataset_name: str) -> None:
    matches = frame["recorded_by"].eq(RECORDED_BY_VALUE)
    if not bool(matches.all()):
        mismatch_count = int((~matches).sum())
        raise DataPreparationError(
            f"{dataset_name} recorded_by must contain only "
            f"{RECORDED_BY_VALUE!r}; {mismatch_count} row(s) differ."
        )


def _validate_submission_format(
    submission_format: pd.DataFrame,
    test_ids: pd.Series,
) -> None:
    dataset_name = "submission format"
    _require_data_frame(submission_format, dataset_name)
    _validate_exact_columns(
        submission_format,
        (ID_COLUMN, TARGET_COLUMN),
        dataset_name,
    )
    _validate_ids(submission_format[ID_COLUMN], dataset_name)

    # Series equality includes index labels. Resetting with `drop=True` discards
    # those labels and creates 0..n-1 indexes, so this final comparison asks the
    # intended question: are the ID values identical and in the same row order?
    if not submission_format[ID_COLUMN].reset_index(drop=True).equals(
        test_ids.reset_index(drop=True)
    ):
        raise DataPreparationError(
            "Submission format IDs do not match test feature IDs in row order."
        )
