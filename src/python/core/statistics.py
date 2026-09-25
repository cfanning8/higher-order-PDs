from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


LabelArray: TypeAlias = NDArray[np.int64]
IndexArray: TypeAlias = NDArray[np.int64]
FloatArray: TypeAlias = NDArray[np.float64]

_UINT64_MAX = (1 << 64) - 1


def _require_integer(
    value: int,
    *,
    description: str,
) -> int:
    if (
        isinstance(
            value,
            (bool, np.bool_),
        )
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise TypeError(
            f"{description.capitalize()} must be an integer."
        )

    return int(
        value
    )


def _require_positive_integer(
    value: int,
    *,
    description: str,
) -> int:
    parsed = _require_integer(
        value,
        description=description,
    )

    if parsed <= 0:
        raise ValueError(
            f"{description.capitalize()} must be positive."
        )

    return parsed


def _require_seed(
    value: int,
    *,
    description: str,
) -> int:
    parsed = _require_integer(
        value,
        description=description,
    )

    if (
        parsed < 0
        or parsed > _UINT64_MAX
    ):
        raise ValueError(
            f"{description.capitalize()} must lie in the unsigned "
            "64-bit range."
        )

    return parsed


def _require_confidence_level(
    value: float,
) -> float:
    if (
        isinstance(
            value,
            (bool, np.bool_),
        )
        or not isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    ):
        raise TypeError(
            "Bootstrap confidence level must be numeric."
        )

    parsed = float(
        value
    )

    if (
        not math.isfinite(
            parsed
        )
        or parsed <= 0.0
        or parsed >= 1.0
    ):
        raise ValueError(
            "Bootstrap confidence level must lie strictly between "
            "0 and 1."
        )

    return parsed


def _require_resample_count(
    value: int,
) -> int:
    parsed = _require_positive_integer(
        value,
        description="bootstrap resample count",
    )

    if parsed < 2:
        raise ValueError(
            "Bootstrap resample count must be at least 2."
        )

    return parsed


def _as_integer_labels(
    values: ArrayLike,
    *,
    description: str,
    require_two_classes: bool,
) -> LabelArray:
    raw = np.asarray(
        values
    )

    if raw.ndim != 1:
        raise ValueError(
            f"{description.capitalize()} must be one-dimensional."
        )

    if raw.size == 0:
        raise ValueError(
            f"{description.capitalize()} must not be empty."
        )

    if np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError(
            f"{description.capitalize()} must contain integers, not "
            "booleans."
        )

    if not np.issubdtype(
        raw.dtype,
        np.integer,
    ):
        raise TypeError(
            f"{description.capitalize()} must have an integer dtype."
        )

    try:
        converted = np.asarray(
            raw,
            dtype=np.int64,
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{description.capitalize()} must be representable as "
            "signed 64-bit integers."
        ) from error

    if not np.array_equal(
        raw,
        converted,
    ):
        raise ValueError(
            f"{description.capitalize()} cannot be represented "
            "exactly as signed 64-bit integers."
        )

    result = np.array(
        converted,
        dtype=np.int64,
        copy=True,
    )

    if (
        require_two_classes
        and np.unique(
            result
        ).size < 2
    ):
        raise ValueError(
            f"{description.capitalize()} must contain at least two "
            "distinct classes."
        )

    result.setflags(
        write=False
    )

    return result


def _as_fold_ids(
    values: ArrayLike,
    *,
    expected_size: int,
) -> IndexArray:
    raw = np.asarray(
        values
    )

    if raw.ndim != 1:
        raise ValueError(
            "Fold IDs must be one-dimensional."
        )

    if raw.size != expected_size:
        raise ValueError(
            f"Fold ID count {raw.size} does not match sample count "
            f"{expected_size}."
        )

    if np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError(
            "Fold IDs must contain integers, not booleans."
        )

    if not np.issubdtype(
        raw.dtype,
        np.integer,
    ):
        raise TypeError(
            "Fold IDs must have an integer dtype."
        )

    converted = np.asarray(
        raw,
        dtype=np.int64,
    )

    if not np.array_equal(
        raw,
        converted,
    ):
        raise ValueError(
            "Fold IDs cannot be represented exactly as signed "
            "64-bit integers."
        )

    result = np.array(
        converted,
        dtype=np.int64,
        copy=True,
    )

    if bool(
        np.any(
            result < 0
        )
    ):
        raise ValueError(
            "Fold IDs must be nonnegative."
        )

    result.setflags(
        write=False
    )

    return result


def _validated_classification_inputs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[
    LabelArray,
    LabelArray,
    LabelArray,
]:
    true_labels = _as_integer_labels(
        y_true,
        description="true labels",
        require_two_classes=True,
    )

    predicted_labels = _as_integer_labels(
        y_pred,
        description="predicted labels",
        require_two_classes=False,
    )

    if true_labels.shape != predicted_labels.shape:
        raise ValueError(
            f"Prediction count {predicted_labels.size} does not "
            f"match label count {true_labels.size}."
        )

    classes = np.array(
        np.unique(
            true_labels
        ),
        dtype=np.int64,
        copy=True,
    )

    classes.setflags(
        write=False
    )

    return (
        true_labels,
        predicted_labels,
        classes,
    )


def _encode_labels(
    labels: LabelArray,
    classes: LabelArray,
) -> IndexArray:
    encoded = np.asarray(
        np.searchsorted(
            classes,
            labels,
        ),
        dtype=np.int64,
    )

    return encoded


def _metric_values_encoded(
    true_codes: IndexArray,
    predicted_codes: IndexArray,
    class_count: int,
) -> tuple[
    float,
    float,
]:
    confusion = np.bincount(
        (
            true_codes
            *
            class_count
            +
            predicted_codes
        ),
        minlength=(
            class_count
            *
            class_count
        ),
    ).reshape(
        (
            class_count,
            class_count,
        )
    )

    true_positive = np.diag(
        confusion
    ).astype(
        np.float64,
        copy=False,
    )

    true_count = np.sum(
        confusion,
        axis=1,
        dtype=np.int64,
    ).astype(
        np.float64,
        copy=False,
    )

    predicted_count = np.sum(
        confusion,
        axis=0,
        dtype=np.int64,
    ).astype(
        np.float64,
        copy=False,
    )

    accuracy_value = (
        float(
            np.sum(
                true_positive
            )
        )
        /
        int(
            true_codes.size
        )
    )

    class_f1 = (
        2.0
        *
        true_positive
        /
        (
            true_count
            +
            predicted_count
        )
    )

    macro_f1_value = float(
        np.mean(
            class_f1
        )
    )

    return (
        accuracy_value,
        macro_f1_value,
    )


def _metric_values_validated(
    y_true: LabelArray,
    y_pred: LabelArray,
    classes: LabelArray,
) -> tuple[
    float,
    float,
]:
    class_count = int(
        classes.size
    )

    return _metric_values_encoded(
        _encode_labels(
            y_true,
            classes,
        ),
        _encode_labels(
            y_pred,
            classes,
        ),
        class_count,
    )


@dataclass(
    frozen=True,
    slots=True,
)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float


@dataclass(
    frozen=True,
    slots=True,
)
class MetricDifference:
    accuracy: float
    macro_f1: float


@dataclass(
    frozen=True,
    slots=True,
)
class FoldMetrics:
    fold_index: int
    sample_count: int
    metrics: ClassificationMetrics


@dataclass(
    frozen=True,
    slots=True,
)
class MetricSummary:
    count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


@dataclass(
    frozen=True,
    slots=True,
)
class ClassificationMetricSummary:
    fold_count: int
    accuracy: MetricSummary
    macro_f1: MetricSummary


@dataclass(
    frozen=True,
    slots=True,
)
class OutOfFoldEvaluation:
    sample_count: int
    pooled_metrics: ClassificationMetrics
    fold_metrics: tuple[
        FoldMetrics,
        ...,
    ]
    fold_summary: ClassificationMetricSummary


@dataclass(
    frozen=True,
    slots=True,
)
class ConfidenceInterval:
    confidence_level: float
    lower: float
    upper: float


@dataclass(
    frozen=True,
    slots=True,
)
class BootstrapScalarSummary:
    observed: float
    bootstrap_mean: float
    bootstrap_standard_deviation: float
    interval: ConfidenceInterval


@dataclass(
    frozen=True,
    slots=True,
)
class AbsoluteBootstrapMetrics:
    accuracy: BootstrapScalarSummary
    macro_f1: BootstrapScalarSummary


@dataclass(
    frozen=True,
    slots=True,
)
class DifferenceBootstrapMetrics:
    accuracy: BootstrapScalarSummary
    macro_f1: BootstrapScalarSummary


@dataclass(
    frozen=True,
    slots=True,
)
class RealizationAveragedBootstrapMetrics:
    accuracy: BootstrapScalarSummary
    macro_f1: BootstrapScalarSummary


@dataclass(
    frozen=True,
    slots=True,
)
class ClassificationBootstrapResult:
    resample_count: int
    seed: int
    confidence_level: float
    metrics: AbsoluteBootstrapMetrics


@dataclass(
    frozen=True,
    slots=True,
)
class PairedBootstrapComparison:
    resample_count: int
    seed: int
    confidence_level: float
    first: AbsoluteBootstrapMetrics
    second: AbsoluteBootstrapMetrics
    difference: DifferenceBootstrapMetrics


@dataclass(
    frozen=True,
    slots=True,
)
class HierarchicalPairedBootstrapComparison:
    resample_count: int
    realization_count: int
    seed: int
    confidence_level: float
    first_mean: RealizationAveragedBootstrapMetrics
    second: AbsoluteBootstrapMetrics
    difference: DifferenceBootstrapMetrics


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> ClassificationMetrics:
    (
        true_labels,
        predicted_labels,
        classes,
    ) = _validated_classification_inputs(
        y_true,
        y_pred,
    )

    accuracy_value, macro_f1_value = (
        _metric_values_validated(
            true_labels,
            predicted_labels,
            classes,
        )
    )

    return ClassificationMetrics(
        accuracy=accuracy_value,
        macro_f1=macro_f1_value,
    )


def accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    return classification_metrics(
        y_true,
        y_pred,
    ).accuracy


def macro_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    return classification_metrics(
        y_true,
        y_pred,
    ).macro_f1


def _metric_summary(
    values: Sequence[float],
) -> MetricSummary:
    normalized = np.asarray(
        tuple(
            values
        ),
        dtype=np.float64,
    )

    if normalized.size == 0:
        raise ValueError(
            "Metric summary requires at least one value."
        )

    count = int(
        normalized.size
    )

    return MetricSummary(
        count=count,
        mean=float(
            np.mean(
                normalized
            )
        ),
        standard_deviation=(
            0.0
            if count == 1
            else float(
                np.std(
                    normalized,
                    ddof=1,
                )
            )
        ),
        minimum=float(
            np.min(
                normalized
            )
        ),
        maximum=float(
            np.max(
                normalized
            )
        ),
    )


def _metrics_by_fold_validated(
    true_labels: LabelArray,
    predicted_labels: LabelArray,
    classes: LabelArray,
    folds: IndexArray,
) -> tuple[
    FoldMetrics,
    ...,
]:
    fold_indices = np.unique(
        folds
    )

    results: list[
        FoldMetrics
    ] = []

    for fold_index_value in fold_indices:
        fold_index = int(
            fold_index_value
        )

        mask = (
            folds
            ==
            fold_index
        )

        fold_true = true_labels[
            mask
        ]

        fold_predicted = predicted_labels[
            mask
        ]

        accuracy_value, macro_f1_value = (
            _metric_values_validated(
                fold_true,
                fold_predicted,
                classes,
            )
        )

        results.append(
            FoldMetrics(
                fold_index=fold_index,
                sample_count=int(
                    fold_true.size
                ),
                metrics=ClassificationMetrics(
                    accuracy=accuracy_value,
                    macro_f1=macro_f1_value,
                ),
            )
        )

    return tuple(
        results
    )


def metrics_by_fold(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    fold_ids: ArrayLike,
) -> tuple[
    FoldMetrics,
    ...,
]:
    (
        true_labels,
        predicted_labels,
        classes,
    ) = _validated_classification_inputs(
        y_true,
        y_pred,
    )

    folds = _as_fold_ids(
        fold_ids,
        expected_size=int(
            true_labels.size
        ),
    )

    return _metrics_by_fold_validated(
        true_labels,
        predicted_labels,
        classes,
        folds,
    )


def summarize_fold_metrics(
    folds: Sequence[FoldMetrics],
) -> ClassificationMetricSummary:
    normalized = tuple(
        folds
    )

    if not normalized:
        raise ValueError(
            "Fold metric summary requires at least one fold."
        )

    return ClassificationMetricSummary(
        fold_count=len(
            normalized
        ),
        accuracy=_metric_summary(
            tuple(
                fold.metrics.accuracy
                for fold in normalized
            )
        ),
        macro_f1=_metric_summary(
            tuple(
                fold.metrics.macro_f1
                for fold in normalized
            )
        ),
    )


def evaluate_out_of_fold_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    fold_ids: ArrayLike,
) -> OutOfFoldEvaluation:
    (
        true_labels,
        predicted_labels,
        classes,
    ) = _validated_classification_inputs(
        y_true,
        y_pred,
    )

    folds = _as_fold_ids(
        fold_ids,
        expected_size=int(
            true_labels.size
        ),
    )

    pooled_accuracy, pooled_macro_f1 = (
        _metric_values_validated(
            true_labels,
            predicted_labels,
            classes,
        )
    )

    per_fold = _metrics_by_fold_validated(
        true_labels,
        predicted_labels,
        classes,
        folds,
    )

    return OutOfFoldEvaluation(
        sample_count=int(
            true_labels.size
        ),
        pooled_metrics=ClassificationMetrics(
            accuracy=pooled_accuracy,
            macro_f1=pooled_macro_f1,
        ),
        fold_metrics=per_fold,
        fold_summary=summarize_fold_metrics(
            per_fold
        ),
    )


def paired_metric_difference(
    y_true: ArrayLike,
    first_predictions: ArrayLike,
    second_predictions: ArrayLike,
) -> MetricDifference:
    (
        true_labels,
        first_labels,
        classes,
    ) = _validated_classification_inputs(
        y_true,
        first_predictions,
    )

    second_labels = _as_integer_labels(
        second_predictions,
        description="second predictions",
        require_two_classes=False,
    )

    if second_labels.shape != true_labels.shape:
        raise ValueError(
            f"Prediction count {second_labels.size} does not match "
            f"label count {true_labels.size}."
        )

    first_accuracy, first_macro_f1 = (
        _metric_values_validated(
            true_labels,
            first_labels,
            classes,
        )
    )

    second_accuracy, second_macro_f1 = (
        _metric_values_validated(
            true_labels,
            second_labels,
            classes,
        )
    )

    return MetricDifference(
        accuracy=(
            first_accuracy
            -
            second_accuracy
        ),
        macro_f1=(
            first_macro_f1
            -
            second_macro_f1
        ),
    )


def _class_index_groups(
    y_true: LabelArray,
    classes: LabelArray,
) -> tuple[
    IndexArray,
    ...,
]:
    return tuple(
        np.asarray(
            np.flatnonzero(
                y_true
                ==
                class_label
            ),
            dtype=np.int64,
        )
        for class_label in classes
    )


def _sample_stratified_indices(
    rng: np.random.Generator,
    class_indices: tuple[
        IndexArray,
        ...,
    ],
) -> IndexArray:
    return np.concatenate(
        tuple(
            rng.choice(
                indices,
                size=indices.size,
                replace=True,
            )
            for indices in class_indices
        )
    ).astype(
        np.int64,
        copy=False,
    )


def _bootstrap_distribution_summary(
    observed: float,
    values: FloatArray,
    *,
    confidence_level: float,
) -> BootstrapScalarSummary:
    alpha = (
        1.0
        -
        confidence_level
    )

    lower, upper = np.quantile(
        values,
        (
            alpha
            /
            2.0,
            1.0
            -
            alpha
            /
            2.0,
        ),
        method="linear",
    )

    return BootstrapScalarSummary(
        observed=float(
            observed
        ),
        bootstrap_mean=float(
            np.mean(
                values
            )
        ),
        bootstrap_standard_deviation=float(
            np.std(
                values,
                ddof=1,
            )
        ),
        interval=ConfidenceInterval(
            confidence_level=confidence_level,
            lower=float(
                lower
            ),
            upper=float(
                upper
            ),
        ),
    )


def _absolute_bootstrap_metrics(
    observed_accuracy: float,
    observed_macro_f1: float,
    accuracy_values: FloatArray,
    macro_f1_values: FloatArray,
    *,
    confidence_level: float,
) -> AbsoluteBootstrapMetrics:
    return AbsoluteBootstrapMetrics(
        accuracy=_bootstrap_distribution_summary(
            observed_accuracy,
            accuracy_values,
            confidence_level=confidence_level,
        ),
        macro_f1=_bootstrap_distribution_summary(
            observed_macro_f1,
            macro_f1_values,
            confidence_level=confidence_level,
        ),
    )


def _difference_bootstrap_metrics(
    observed_accuracy: float,
    observed_macro_f1: float,
    accuracy_values: FloatArray,
    macro_f1_values: FloatArray,
    *,
    confidence_level: float,
) -> DifferenceBootstrapMetrics:
    return DifferenceBootstrapMetrics(
        accuracy=_bootstrap_distribution_summary(
            observed_accuracy,
            accuracy_values,
            confidence_level=confidence_level,
        ),
        macro_f1=_bootstrap_distribution_summary(
            observed_macro_f1,
            macro_f1_values,
            confidence_level=confidence_level,
        ),
    )


def _realization_averaged_bootstrap_metrics(
    observed_accuracy: float,
    observed_macro_f1: float,
    accuracy_values: FloatArray,
    macro_f1_values: FloatArray,
    *,
    confidence_level: float,
) -> RealizationAveragedBootstrapMetrics:
    return RealizationAveragedBootstrapMetrics(
        accuracy=_bootstrap_distribution_summary(
            observed_accuracy,
            accuracy_values,
            confidence_level=confidence_level,
        ),
        macro_f1=_bootstrap_distribution_summary(
            observed_macro_f1,
            macro_f1_values,
            confidence_level=confidence_level,
        ),
    )


def stratified_bootstrap(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int,
) -> ClassificationBootstrapResult:
    """
    Bootstrap pooled classification performance by resampling
    observations within true classes.

    The bootstrap conditions on the fitted out-of-fold prediction
    pipeline. It does not refit representations, preprocessing,
    classifiers, model-selection procedures, or hyperparameters.
    """

    (
        true_labels,
        predicted_labels,
        classes,
    ) = _validated_classification_inputs(
        y_true,
        y_pred,
    )

    parsed_resamples = _require_resample_count(
        resamples
    )

    parsed_confidence_level = _require_confidence_level(
        confidence_level
    )

    parsed_seed = _require_seed(
        seed,
        description="bootstrap seed",
    )

    class_count = int(
        classes.size
    )

    true_codes = _encode_labels(
        true_labels,
        classes,
    )

    predicted_codes = _encode_labels(
        predicted_labels,
        classes,
    )

    observed_accuracy, observed_macro_f1 = (
        _metric_values_encoded(
            true_codes,
            predicted_codes,
            class_count,
        )
    )

    class_indices = _class_index_groups(
        true_labels,
        classes,
    )

    rng = np.random.default_rng(
        parsed_seed
    )

    accuracy_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    macro_f1_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    for replicate_index in range(
        parsed_resamples
    ):
        sampled_indices = _sample_stratified_indices(
            rng,
            class_indices,
        )

        (
            accuracy_values[
                replicate_index
            ],
            macro_f1_values[
                replicate_index
            ],
        ) = _metric_values_encoded(
            true_codes[
                sampled_indices
            ],
            predicted_codes[
                sampled_indices
            ],
            class_count,
        )

    return ClassificationBootstrapResult(
        resample_count=parsed_resamples,
        seed=parsed_seed,
        confidence_level=parsed_confidence_level,
        metrics=_absolute_bootstrap_metrics(
            observed_accuracy,
            observed_macro_f1,
            accuracy_values,
            macro_f1_values,
            confidence_level=parsed_confidence_level,
        ),
    )


def paired_stratified_bootstrap(
    y_true: ArrayLike,
    first_predictions: ArrayLike,
    second_predictions: ArrayLike,
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int,
) -> PairedBootstrapComparison:
    """
    Compare two fixed out-of-fold prediction vectors with a paired
    class-stratified percentile bootstrap.

    Differences use the convention first - second. Every bootstrap
    replicate uses the same sampled observations for both methods.

    The bootstrap conditions on the fitted out-of-fold prediction
    pipelines and quantifies sample-level resampling uncertainty. It
    does not refit representations, preprocessing, classifiers,
    model-selection procedures, or hyperparameters.
    """

    (
        true_labels,
        first_labels,
        classes,
    ) = _validated_classification_inputs(
        y_true,
        first_predictions,
    )

    second_labels = _as_integer_labels(
        second_predictions,
        description="second predictions",
        require_two_classes=False,
    )

    if second_labels.shape != true_labels.shape:
        raise ValueError(
            f"Prediction count {second_labels.size} does not match "
            f"label count {true_labels.size}."
        )

    parsed_resamples = _require_resample_count(
        resamples
    )

    parsed_confidence_level = _require_confidence_level(
        confidence_level
    )

    parsed_seed = _require_seed(
        seed,
        description="paired bootstrap seed",
    )

    class_count = int(
        classes.size
    )

    true_codes = _encode_labels(
        true_labels,
        classes,
    )

    first_codes = _encode_labels(
        first_labels,
        classes,
    )

    second_codes = _encode_labels(
        second_labels,
        classes,
    )

    (
        observed_first_accuracy,
        observed_first_macro_f1,
    ) = _metric_values_encoded(
        true_codes,
        first_codes,
        class_count,
    )

    (
        observed_second_accuracy,
        observed_second_macro_f1,
    ) = _metric_values_encoded(
        true_codes,
        second_codes,
        class_count,
    )

    observed_accuracy_difference = (
        observed_first_accuracy
        -
        observed_second_accuracy
    )

    observed_macro_f1_difference = (
        observed_first_macro_f1
        -
        observed_second_macro_f1
    )

    class_indices = _class_index_groups(
        true_labels,
        classes,
    )

    rng = np.random.default_rng(
        parsed_seed
    )

    first_accuracy_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    first_macro_f1_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    second_accuracy_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    second_macro_f1_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    accuracy_difference_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    macro_f1_difference_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    for replicate_index in range(
        parsed_resamples
    ):
        sampled_indices = _sample_stratified_indices(
            rng,
            class_indices,
        )

        sampled_true_codes = true_codes[
            sampled_indices
        ]

        (
            first_accuracy,
            first_macro_f1,
        ) = _metric_values_encoded(
            sampled_true_codes,
            first_codes[
                sampled_indices
            ],
            class_count,
        )

        (
            second_accuracy,
            second_macro_f1,
        ) = _metric_values_encoded(
            sampled_true_codes,
            second_codes[
                sampled_indices
            ],
            class_count,
        )

        first_accuracy_values[
            replicate_index
        ] = first_accuracy

        first_macro_f1_values[
            replicate_index
        ] = first_macro_f1

        second_accuracy_values[
            replicate_index
        ] = second_accuracy

        second_macro_f1_values[
            replicate_index
        ] = second_macro_f1

        accuracy_difference_values[
            replicate_index
        ] = (
            first_accuracy
            -
            second_accuracy
        )

        macro_f1_difference_values[
            replicate_index
        ] = (
            first_macro_f1
            -
            second_macro_f1
        )

    return PairedBootstrapComparison(
        resample_count=parsed_resamples,
        seed=parsed_seed,
        confidence_level=parsed_confidence_level,
        first=_absolute_bootstrap_metrics(
            observed_first_accuracy,
            observed_first_macro_f1,
            first_accuracy_values,
            first_macro_f1_values,
            confidence_level=parsed_confidence_level,
        ),
        second=_absolute_bootstrap_metrics(
            observed_second_accuracy,
            observed_second_macro_f1,
            second_accuracy_values,
            second_macro_f1_values,
            confidence_level=parsed_confidence_level,
        ),
        difference=_difference_bootstrap_metrics(
            observed_accuracy_difference,
            observed_macro_f1_difference,
            accuracy_difference_values,
            macro_f1_difference_values,
            confidence_level=parsed_confidence_level,
        ),
    )


def hierarchical_paired_stratified_bootstrap(
    y_true: ArrayLike,
    first_predictions: Sequence[ArrayLike],
    second_predictions: ArrayLike,
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int,
) -> HierarchicalPairedBootstrapComparison:
    """
    Compare realization-averaged performance with one fixed paired
    reference using a hierarchical percentile bootstrap.

    Differences use the convention mean(first realizations) - second.

    Each bootstrap replicate first resamples observations within true
    classes. It then independently resamples realization indices with
    replacement. Every selected realization and the fixed reference
    use the same sampled observations.

    The bootstrap conditions on the fitted nested-cross-validation
    pipelines for all realizations. It captures sample-level
    resampling uncertainty and finite-realization Monte Carlo
    uncertainty. It does not refit representations, preprocessing,
    classifiers, model-selection procedures, or hyperparameters.
    """

    true_labels = _as_integer_labels(
        y_true,
        description="true labels",
        require_two_classes=True,
    )

    first_prediction_values = tuple(
        first_predictions
    )

    if not first_prediction_values:
        raise ValueError(
            "Hierarchical paired bootstrap requires at least one "
            "first prediction vector."
        )

    classes = np.array(
        np.unique(
            true_labels
        ),
        dtype=np.int64,
        copy=True,
    )

    classes.setflags(
        write=False
    )

    first_label_arrays: list[
        LabelArray
    ] = []

    for realization_index, values in enumerate(
        first_prediction_values
    ):
        labels = _as_integer_labels(
            values,
            description=(
                f"first prediction vector {realization_index}"
            ),
            require_two_classes=False,
        )

        if labels.shape != true_labels.shape:
            raise ValueError(
                f"First prediction vector {realization_index} has "
                f"{labels.size} predictions, but {true_labels.size} "
                "labels were supplied."
            )

        first_label_arrays.append(
            labels
        )

    second_labels = _as_integer_labels(
        second_predictions,
        description="second predictions",
        require_two_classes=False,
    )

    if second_labels.shape != true_labels.shape:
        raise ValueError(
            f"Prediction count {second_labels.size} does not match "
            f"label count {true_labels.size}."
        )

    parsed_resamples = _require_resample_count(
        resamples
    )

    parsed_confidence_level = _require_confidence_level(
        confidence_level
    )

    parsed_seed = _require_seed(
        seed,
        description="hierarchical paired bootstrap seed",
    )

    realization_count = len(
        first_label_arrays
    )

    class_count = int(
        classes.size
    )

    true_codes = _encode_labels(
        true_labels,
        classes,
    )

    first_codes = tuple(
        _encode_labels(
            labels,
            classes,
        )
        for labels in first_label_arrays
    )

    second_codes = _encode_labels(
        second_labels,
        classes,
    )

    observed_first_accuracy_values = np.empty(
        realization_count,
        dtype=np.float64,
    )

    observed_first_macro_f1_values = np.empty(
        realization_count,
        dtype=np.float64,
    )

    for realization_index, realization_codes in enumerate(
        first_codes
    ):
        (
            observed_first_accuracy_values[
                realization_index
            ],
            observed_first_macro_f1_values[
                realization_index
            ],
        ) = _metric_values_encoded(
            true_codes,
            realization_codes,
            class_count,
        )

    observed_first_accuracy = float(
        np.mean(
            observed_first_accuracy_values
        )
    )

    observed_first_macro_f1 = float(
        np.mean(
            observed_first_macro_f1_values
        )
    )

    (
        observed_second_accuracy,
        observed_second_macro_f1,
    ) = _metric_values_encoded(
        true_codes,
        second_codes,
        class_count,
    )

    observed_accuracy_difference = (
        observed_first_accuracy
        -
        observed_second_accuracy
    )

    observed_macro_f1_difference = (
        observed_first_macro_f1
        -
        observed_second_macro_f1
    )

    class_indices = _class_index_groups(
        true_labels,
        classes,
    )

    rng = np.random.default_rng(
        parsed_seed
    )

    first_accuracy_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    first_macro_f1_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    second_accuracy_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    second_macro_f1_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    accuracy_difference_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    macro_f1_difference_values = np.empty(
        parsed_resamples,
        dtype=np.float64,
    )

    realization_accuracy_buffer = np.empty(
        realization_count,
        dtype=np.float64,
    )

    realization_macro_f1_buffer = np.empty(
        realization_count,
        dtype=np.float64,
    )

    for replicate_index in range(
        parsed_resamples
    ):
        sampled_indices = _sample_stratified_indices(
            rng,
            class_indices,
        )

        sampled_realization_indices = rng.integers(
            0,
            realization_count,
            size=realization_count,
        )

        sampled_true_codes = true_codes[
            sampled_indices
        ]

        for sampled_position, realization_index in enumerate(
            sampled_realization_indices
        ):
            (
                realization_accuracy_buffer[
                    sampled_position
                ],
                realization_macro_f1_buffer[
                    sampled_position
                ],
            ) = _metric_values_encoded(
                sampled_true_codes,
                first_codes[
                    int(
                        realization_index
                    )
                ][
                    sampled_indices
                ],
                class_count,
            )

        first_accuracy = float(
            np.mean(
                realization_accuracy_buffer
            )
        )

        first_macro_f1 = float(
            np.mean(
                realization_macro_f1_buffer
            )
        )

        (
            second_accuracy,
            second_macro_f1,
        ) = _metric_values_encoded(
            sampled_true_codes,
            second_codes[
                sampled_indices
            ],
            class_count,
        )

        first_accuracy_values[
            replicate_index
        ] = first_accuracy

        first_macro_f1_values[
            replicate_index
        ] = first_macro_f1

        second_accuracy_values[
            replicate_index
        ] = second_accuracy

        second_macro_f1_values[
            replicate_index
        ] = second_macro_f1

        accuracy_difference_values[
            replicate_index
        ] = (
            first_accuracy
            -
            second_accuracy
        )

        macro_f1_difference_values[
            replicate_index
        ] = (
            first_macro_f1
            -
            second_macro_f1
        )

    return HierarchicalPairedBootstrapComparison(
        resample_count=parsed_resamples,
        realization_count=realization_count,
        seed=parsed_seed,
        confidence_level=parsed_confidence_level,
        first_mean=_realization_averaged_bootstrap_metrics(
            observed_first_accuracy,
            observed_first_macro_f1,
            first_accuracy_values,
            first_macro_f1_values,
            confidence_level=parsed_confidence_level,
        ),
        second=_absolute_bootstrap_metrics(
            observed_second_accuracy,
            observed_second_macro_f1,
            second_accuracy_values,
            second_macro_f1_values,
            confidence_level=parsed_confidence_level,
        ),
        difference=_difference_bootstrap_metrics(
            observed_accuracy_difference,
            observed_macro_f1_difference,
            accuracy_difference_values,
            macro_f1_difference_values,
            confidence_level=parsed_confidence_level,
        ),
    )


__all__ = [
    "AbsoluteBootstrapMetrics",
    "BootstrapScalarSummary",
    "ClassificationBootstrapResult",
    "ClassificationMetricSummary",
    "ClassificationMetrics",
    "ConfidenceInterval",
    "DifferenceBootstrapMetrics",
    "FoldMetrics",
    "HierarchicalPairedBootstrapComparison",
    "MetricDifference",
    "MetricSummary",
    "OutOfFoldEvaluation",
    "PairedBootstrapComparison",
    "RealizationAveragedBootstrapMetrics",
    "accuracy",
    "classification_metrics",
    "evaluate_out_of_fold_predictions",
    "hierarchical_paired_stratified_bootstrap",
    "macro_f1",
    "metrics_by_fold",
    "paired_metric_difference",
    "paired_stratified_bootstrap",
    "stratified_bootstrap",
    "summarize_fold_metrics",
]