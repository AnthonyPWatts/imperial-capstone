"""Cross-fitted local spatial class-rate features."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
from sklearn.neighbors import BallTree as _BallTree
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.preprocessing import StandardScaler as _StandardScaler
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from feature_engineering import engineer_initial_features
from feature_engineering import valid_tanzania_coordinates


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
CLASS_TO_INTEGER = {
    label: position for position, label in enumerate(CLASS_LABELS)
}
SPATIAL_INNER_FOLDS = 5
SPATIAL_CROSS_FIT_SEED = 20260822
EARTH_RADIUS_KM = 6371.0088
BASE_RARE_CATEGORY_MINIMUM = 20
SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_WORST_FOLD = -0.0025
SELECTION_GATE_REPAIR_RECALL = -0.02

SPATIAL_RATE_FEATURES = (
    "spatial_functional_rate",
    "spatial_repair_rate",
    "spatial_non_functional_rate",
)
SPATIAL_RADIUS_FEATURE = "spatial_neighbour_radius_km"
SPATIAL_NUMERIC_FEATURES = (*SPATIAL_RATE_FEATURES, SPATIAL_RADIUS_FEATURE)


@_dataclass(frozen=True)
class SpatialOutcomePolicy:
    """One fixed local-neighbour class-rate representation."""

    key: str
    label: str
    neighbours: int
    smoothing: float


@_dataclass(frozen=True)
class SpatialOutcomeModel:
    """Training coordinates, encoded labels and global fallback prior."""

    coordinates_radians: _np.ndarray
    encoded_target: _np.ndarray
    class_prior: _np.ndarray
    neighbours: int
    smoothing: float


SPATIAL_OUTCOME_POLICY = SpatialOutcomePolicy(
    key="spatial_k10_s20",
    label="Ten-neighbour spatial class rates",
    neighbours=10,
    smoothing=20.0,
)


class SpatialOutcomeFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Add spatial rates with cross-fitting for training transformations."""

    def __init__(self, policy: SpatialOutcomePolicy = SPATIAL_OUTCOME_POLICY):
        self.policy = policy

    def fit(
        self,
        X: _pd.DataFrame,
        y: _pd.Series | _np.ndarray,
    ) -> "SpatialOutcomeFeatureEngineer":
        _validate_policy(self.policy)
        encoded_target = _encode_target(y, expected_rows=len(X))
        self.model_ = fit_spatial_outcome_model(
            X,
            encoded_target,
            self.policy,
        )
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(
            spatial_model_features(),
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def fit_transform(
        self,
        X: _pd.DataFrame,
        y: _pd.Series | _np.ndarray,
        **fit_params,
    ) -> _pd.DataFrame:
        """Return inner-OOF rates, then retain a full model for transformation."""

        _validate_policy(self.policy)
        encoded_target = _encode_target(y, expected_rows=len(X))
        cross_validation = _StratifiedKFold(
            n_splits=SPATIAL_INNER_FOLDS,
            shuffle=True,
            random_state=SPATIAL_CROSS_FIT_SEED,
        )
        spatial_values = _np.full(
            (len(X), len(SPATIAL_NUMERIC_FEATURES)),
            fill_value=_np.nan,
        )
        positions = _np.arange(len(X))
        for training_positions, validation_positions in cross_validation.split(
            positions,
            encoded_target,
        ):
            model = fit_spatial_outcome_model(
                X.iloc[training_positions],
                encoded_target[training_positions],
                self.policy,
            )
            spatial_values[validation_positions] = spatial_outcome_values(
                X.iloc[validation_positions],
                model,
            )
        if _np.isnan(spatial_values[:, : len(SPATIAL_RATE_FEATURES)]).any():
            raise ValueError("Cross-fitted spatial class rates are incomplete.")
        self.fit(X, encoded_target)
        return _append_spatial_values(X, spatial_values)

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, ("model_", "feature_names_out_"))
        return _append_spatial_values(
            X,
            spatial_outcome_values(X, self.model_),
        )

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def fit_spatial_outcome_model(
    X_training: _pd.DataFrame,
    encoded_target: _pd.Series | _np.ndarray,
    policy: SpatialOutcomePolicy = SPATIAL_OUTCOME_POLICY,
) -> SpatialOutcomeModel:
    """Fit the local-neighbour reference set from training rows only."""

    _validate_policy(policy)
    target = _encode_target(encoded_target, expected_rows=len(X_training))
    valid = valid_tanzania_coordinates(
        X_training["longitude"],
        X_training["latitude"],
    ).to_numpy()
    if int(valid.sum()) < policy.neighbours:
        raise ValueError("Too few valid training coordinates for spatial rates.")
    coordinates = _coordinate_radians(X_training.loc[valid])
    class_prior = _np.bincount(
        target,
        minlength=len(CLASS_LABELS),
    ).astype("float64")
    class_prior /= len(target)
    return SpatialOutcomeModel(
        coordinates_radians=coordinates,
        encoded_target=target[valid],
        class_prior=class_prior,
        neighbours=policy.neighbours,
        smoothing=policy.smoothing,
    )


def spatial_outcome_values(
    X_prediction: _pd.DataFrame,
    model: SpatialOutcomeModel,
) -> _np.ndarray:
    """Return three smoothed class rates and the kth-neighbour radius."""

    values = _np.empty(
        (len(X_prediction), len(SPATIAL_NUMERIC_FEATURES)),
        dtype="float64",
    )
    values[:, : len(CLASS_LABELS)] = model.class_prior
    values[:, -1] = _np.nan
    valid = valid_tanzania_coordinates(
        X_prediction["longitude"],
        X_prediction["latitude"],
    ).to_numpy()
    if valid.any():
        tree = _BallTree(model.coordinates_radians, metric="haversine")
        distances, indices = tree.query(
            _coordinate_radians(X_prediction.loc[valid]),
            k=min(model.neighbours, len(model.encoded_target)),
        )
        neighbour_targets = model.encoded_target[indices]
        counts = _np.column_stack(
            [
                (neighbour_targets == class_position).sum(axis=1)
                for class_position in range(len(CLASS_LABELS))
            ]
        )
        denominator = counts.sum(axis=1, keepdims=True) + model.smoothing
        rates = (
            counts + model.smoothing * model.class_prior.reshape(1, -1)
        ) / denominator
        values[valid, : len(CLASS_LABELS)] = rates
        values[valid, -1] = distances[:, -1] * EARTH_RADIUS_KM
    return values


def spatial_model_features() -> tuple[str, ...]:
    """Return the accepted model features plus spatial outcome values."""

    return (*MODEL_FEATURES, *SPATIAL_NUMERIC_FEATURES)


def make_spatial_outcome_preprocessor(
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build accepted preprocessing with cross-fitted spatial outcome values."""

    numeric_steps: list[tuple[str, object]] = [
        (
            "median_imputation",
            _SimpleImputer(strategy="median", add_indicator=True),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("standardisation", _StandardScaler()))
    column_preprocessing = _ColumnTransformer(
        transformers=[
            (
                "numeric",
                _Pipeline(steps=numeric_steps),
                [*NUMERIC_FEATURES, *SPATIAL_NUMERIC_FEATURES],
            ),
            (
                "categorical",
                _OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=BASE_RARE_CATEGORY_MINIMUM,
                    sparse_output=sparse_output,
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", SpatialOutcomeFeatureEngineer()),
            ("column_preprocessing", column_preprocessing),
        ]
    )


def _append_spatial_values(
    X: _pd.DataFrame,
    values: _np.ndarray,
) -> _pd.DataFrame:
    engineered = engineer_initial_features(X)
    for position, feature in enumerate(SPATIAL_NUMERIC_FEATURES):
        engineered[feature] = values[:, position]
    if tuple(engineered.columns) != spatial_model_features():
        raise ValueError("Spatial outcome feature order changed.")
    return engineered


def _coordinate_radians(X: _pd.DataFrame) -> _np.ndarray:
    return _np.radians(
        X.loc[:, ["latitude", "longitude"]].to_numpy(dtype="float64")
    )


def _encode_target(
    target: _pd.Series | _np.ndarray,
    *,
    expected_rows: int,
) -> _np.ndarray:
    values = _np.asarray(target)
    if values.shape != (expected_rows,):
        raise ValueError("Spatial target must align one-dimensionally with rows.")
    if _np.issubdtype(values.dtype, _np.integer):
        encoded = values.astype("int64", copy=True)
    else:
        encoded = _pd.Series(values).map(CLASS_TO_INTEGER).to_numpy()
    if _pd.isna(encoded).any() or not _np.isin(
        encoded,
        _np.arange(len(CLASS_LABELS)),
    ).all():
        raise ValueError("Spatial target contains unexpected classes.")
    return encoded.astype("int64", copy=False)


def _validate_policy(policy: SpatialOutcomePolicy) -> None:
    if policy != SPATIAL_OUTCOME_POLICY:
        raise ValueError("Spatial outcome policy differs from the fixed register.")
    if policy.neighbours <= 0:
        raise ValueError("Spatial neighbours must be positive.")
    if policy.smoothing <= 0:
        raise ValueError("Spatial smoothing must be positive.")
