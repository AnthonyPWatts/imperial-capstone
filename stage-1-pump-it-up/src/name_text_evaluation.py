"""Evaluate a fold-safe character n-gram model over deferred name fields."""

from __future__ import annotations

import time as _time

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.pipeline import FeatureUnion as _FeatureUnion
from sklearn.pipeline import Pipeline as _Pipeline

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_preprocessing import make_initial_preprocessor
from target_encoding_features import normalise_identity


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
NAME_TEXT_FIELDS = (
    "funder",
    "installer",
    "wpt_name",
    "subvillage",
    "ward",
    "scheme_name",
)
TEXT_NGRAM_RANGE = (3, 5)
TEXT_MINIMUM_DOCUMENTS = 3
TEXT_MAXIMUM_FEATURES = 60_000
TEXT_LOGISTIC_C = 1.0
TEXT_LOGISTIC_MAX_ITERATIONS = 300
TEXT_SEED = 20260822


class NameTextExtractor(_BaseEstimator, _TransformerMixin):
    """Turn each row's named identity fields into one field-tagged document."""

    def fit(self, X: _pd.DataFrame, y: object = None) -> "NameTextExtractor":
        missing = [field for field in NAME_TEXT_FIELDS if field not in X]
        if missing:
            raise ValueError(f"Name-text source fields are missing: {missing!r}.")
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _np.ndarray:
        missing = [field for field in NAME_TEXT_FIELDS if field not in X]
        if missing:
            raise ValueError(f"Name-text source fields are missing: {missing!r}.")
        tokens = []
        for field in NAME_TEXT_FIELDS:
            values = normalise_identity(X[field]).str.replace(" ", "_", regex=False)
            tokens.append(field + "_" + values)
        documents = tokens[0]
        for values in tokens[1:]:
            documents = documents.str.cat(values, sep=" ")
        return documents.to_numpy(dtype=object)


def make_name_text_pipeline() -> _Pipeline:
    """Return the one fixed accepted-plus-character-text linear model."""

    text = _Pipeline(
        steps=[
            ("extract_names", NameTextExtractor()),
            (
                "tfidf",
                _TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=TEXT_NGRAM_RANGE,
                    min_df=TEXT_MINIMUM_DOCUMENTS,
                    max_features=TEXT_MAXIMUM_FEATURES,
                    sublinear_tf=True,
                    dtype=_np.float32,
                ),
            ),
        ]
    )
    features = _FeatureUnion(
        transformer_list=[
            (
                "accepted",
                make_initial_preprocessor(
                    sparse_output=True,
                    scale_numeric=True,
                ),
            ),
            ("name_text", text),
        ]
    )
    classifier = _LogisticRegression(
        C=TEXT_LOGISTIC_C,
        solver="saga",
        max_iter=TEXT_LOGISTIC_MAX_ITERATIONS,
        tol=1e-3,
        class_weight=None,
        random_state=TEXT_SEED,
    )
    return _Pipeline(
        steps=[
            ("features", features),
            ("classifier", classifier),
        ]
    )


def evaluate_name_text_model(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> CandidateEvaluation:
    """Fit the fixed text model separately inside every outer fold."""

    probability_values = _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostics = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        started = _time.perf_counter()
        pipeline = make_name_text_pipeline()
        pipeline.fit(X_training, y_training)
        probabilities = _pd.DataFrame(
            pipeline.predict_proba(X_validation),
            columns=pipeline.named_steps["classifier"].classes_,
        ).loc[:, list(CLASS_LABELS)]
        probability_values[validation_positions] = probabilities.to_numpy()
        text_pipeline = dict(
            pipeline.named_steps["features"].transformer_list
        )["name_text"]
        vectorizer = text_pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["classifier"]
        diagnostics.append(
            {
                "validation_fold": fold_number,
                "text_features": len(vectorizer.vocabulary_),
                "classifier_iterations": int(classifier.n_iter_.max()),
                "total_seconds": _time.perf_counter() - started,
            }
        )
        print(
            f"Completed name-text fold {fold_number}/{CROSS_VALIDATION_FOLDS} "
            f"with {len(vectorizer.vocabulary_):,} n-grams in "
            f"{diagnostics[-1]['total_seconds']:.1f} seconds.",
            flush=True,
        )
    return build_candidate_evaluation(
        model_name="accepted features plus deferred-name character n-grams",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )
