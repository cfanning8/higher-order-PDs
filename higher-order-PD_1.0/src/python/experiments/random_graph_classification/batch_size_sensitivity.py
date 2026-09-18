from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from core.cross_validation import (
    NestedCrossValidationPlan,
    outer_test_fold_by_sample,
    select_candidate,
)
from core.statistics import (
    ClassificationBootstrapResult,
    OutOfFoldEvaluation,
    PairedBootstrapComparison,
    accuracy,
    classification_metrics,
    evaluate_out_of_fold_predictions,
    paired_stratified_bootstrap,
    stratified_bootstrap,
)

from .benchmark import (
    BenchmarkResult,
    _fit_predict_svm,
)
from .configuration import (
    ClassifierSettings,
    PreprocessingSettings,
    RandomGraphClassificationConfig,
)
from .data import (
    BatchSizeCatalog,
    BatchSizeKey,
    BenchmarkSampleKey,
    CommonMetadata,
)
from .output import (
    AnalysisOutputLayout,
    write_batch_size_outer_folds,
    write_batch_size_summary,
)


FloatMatrix: TypeAlias = NDArray[np.float64]
LabelArray: TypeAlias = NDArray[np.int64]
PredictionArray: TypeAlias = NDArray[np.int64]
IndexArray: TypeAlias = NDArray[np.int64]


@dataclass(
    frozen=True,
    slots=True,
)
class _SvmCandidate:
    c: float


@dataclass(
    frozen=True,
    slots=True,
)
class BatchSizeOuterFoldResult:
    graphs_per_batch: int
    outer_fold: int
    train_sample_count: int
    test_sample_count: int
    selected_c: float
    accuracy: float
    macro_f1: float


@dataclass(
    frozen=True,
    slots=True,
)
class BatchSizeConditionResult:
    graphs_per_batch: int
    outer_folds: tuple[
        BatchSizeOuterFoldResult,
        ...,
    ]
    predictions: PredictionArray
    evaluation: OutOfFoldEvaluation
    bootstrap: ClassificationBootstrapResult
    reference_comparison: PairedBootstrapComparison | None


@dataclass(
    frozen=True,
    slots=True,
)
class BatchSizeSensitivityResult:
    sample_keys: tuple[
        BenchmarkSampleKey,
        ...,
    ]
    labels: LabelArray
    cross_validation_plan: NestedCrossValidationPlan
    reference_graphs_per_batch: int
    conditions: Mapping[
        int,
        BatchSizeConditionResult,
    ]


def _batch_size_feature_matrix(
    catalog: BatchSizeCatalog,
    sample_keys: Sequence[
        BenchmarkSampleKey
    ],
    graphs_per_batch: int,
) -> FloatMatrix:
    rows = tuple(
        catalog.harmonic_features[
            (
                model_index,
                parameter_condition_index,
                batch_index,
                graphs_per_batch,
            )
        ].values
        for (
            model_index,
            parameter_condition_index,
            batch_index,
        ) in sample_keys
    )

    result = np.vstack(
        rows
    ).astype(
        np.float64,
        copy=False,
    )

    result.setflags(
        write=False
    )

    return result


def _candidate_score(
    candidate: _SvmCandidate,
    train_indices: IndexArray,
    validation_indices: IndexArray,
    *,
    features: FloatMatrix,
    labels: LabelArray,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
) -> float:
    predictions = _fit_predict_svm(
        features[
            train_indices
        ],
        labels[
            train_indices
        ],
        features[
            validation_indices
        ],
        c=candidate.c,
        preprocessing=preprocessing,
        classifier=classifier,
    )

    return accuracy(
        labels[
            validation_indices
        ],
        predictions,
    )


def _run_batch_size_condition(
    graphs_per_batch: int,
    features: FloatMatrix,
    labels: LabelArray,
    plan: NestedCrossValidationPlan,
    *,
    c_values: Sequence[float],
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
) -> BatchSizeConditionResult:
    candidates = tuple(
        _SvmCandidate(
            c=float(
                c
            )
        )
        for c in c_values
    )

    predictions = np.empty(
        labels.shape,
        dtype=np.int64,
    )

    outer_results: list[
        BatchSizeOuterFoldResult
    ] = []

    for outer_fold in plan.folds:
        selection = select_candidate(
            candidates,
            outer_fold,
            lambda candidate, train_indices, validation_indices: (
                _candidate_score(
                    candidate,
                    train_indices,
                    validation_indices,
                    features=features,
                    labels=labels,
                    preprocessing=preprocessing,
                    classifier=classifier,
                )
            ),
        )

        selected = selection.selected_candidate

        fold_predictions = _fit_predict_svm(
            features[
                outer_fold.train_indices
            ],
            labels[
                outer_fold.train_indices
            ],
            features[
                outer_fold.test_indices
            ],
            c=selected.c,
            preprocessing=preprocessing,
            classifier=classifier,
        )

        predictions[
            outer_fold.test_indices
        ] = fold_predictions

        fold_metrics = classification_metrics(
            labels[
                outer_fold.test_indices
            ],
            fold_predictions,
        )

        outer_results.append(
            BatchSizeOuterFoldResult(
                graphs_per_batch=graphs_per_batch,
                outer_fold=outer_fold.fold_index,
                train_sample_count=int(
                    outer_fold.train_indices.size
                ),
                test_sample_count=int(
                    outer_fold.test_indices.size
                ),
                selected_c=selected.c,
                accuracy=fold_metrics.accuracy,
                macro_f1=fold_metrics.macro_f1,
            )
        )

    fold_ids = outer_test_fold_by_sample(
        plan
    )

    evaluation = evaluate_out_of_fold_predictions(
        labels,
        predictions,
        fold_ids,
    )

    bootstrap = stratified_bootstrap(
        labels,
        predictions,
        resamples=bootstrap_resamples,
        confidence_level=bootstrap_confidence_level,
        seed=bootstrap_seed,
    )

    frozen_predictions = np.array(
        predictions,
        dtype=np.int64,
        copy=True,
    )

    frozen_predictions.setflags(
        write=False
    )

    return BatchSizeConditionResult(
        graphs_per_batch=graphs_per_batch,
        outer_folds=tuple(
            outer_results
        ),
        predictions=frozen_predictions,
        evaluation=evaluation,
        bootstrap=bootstrap,
        reference_comparison=None,
    )


def run_batch_size_sensitivity(
    common: CommonMetadata,
    catalog: BatchSizeCatalog,
    benchmark_result: BenchmarkResult,
    config: RandomGraphClassificationConfig,
) -> BatchSizeSensitivityResult:
    sample_keys = benchmark_result.sample_keys
    labels = benchmark_result.labels
    plan = benchmark_result.cross_validation_plan

    reference_graphs_per_batch = (
        common.generation.benchmark.graphs_per_batch
    )

    condition_results: dict[
        int,
        BatchSizeConditionResult,
    ] = {}

    for graphs_per_batch in catalog.graph_counts:
        features = _batch_size_feature_matrix(
            catalog,
            sample_keys,
            graphs_per_batch,
        )

        condition_results[
            graphs_per_batch
        ] = _run_batch_size_condition(
            graphs_per_batch,
            features,
            labels,
            plan,
            c_values=config.classifier.c_values,
            preprocessing=config.preprocessing,
            classifier=config.classifier,
            bootstrap_resamples=(
                config.bootstrap.resample_count
            ),
            bootstrap_confidence_level=(
                config.bootstrap.confidence_level
            ),
            bootstrap_seed=(
                config.bootstrap.batch_size_seed
            ),
        )

    reference = condition_results[
        reference_graphs_per_batch
    ]

    finalized_conditions: dict[
        int,
        BatchSizeConditionResult,
    ] = {}

    for graphs_per_batch in catalog.graph_counts:
        condition = condition_results[
            graphs_per_batch
        ]

        comparison = (
            None
            if graphs_per_batch
            == reference_graphs_per_batch
            else paired_stratified_bootstrap(
                labels,
                condition.predictions,
                reference.predictions,
                resamples=(
                    config.bootstrap.resample_count
                ),
                confidence_level=(
                    config.bootstrap.confidence_level
                ),
                seed=(
                    config.bootstrap.batch_size_seed
                ),
            )
        )

        finalized_conditions[
            graphs_per_batch
        ] = BatchSizeConditionResult(
            graphs_per_batch=(
                condition.graphs_per_batch
            ),
            outer_folds=condition.outer_folds,
            predictions=condition.predictions,
            evaluation=condition.evaluation,
            bootstrap=condition.bootstrap,
            reference_comparison=comparison,
        )

    return BatchSizeSensitivityResult(
        sample_keys=sample_keys,
        labels=labels,
        cross_validation_plan=plan,
        reference_graphs_per_batch=(
            reference_graphs_per_batch
        ),
        conditions=MappingProxyType(
            {
                graphs_per_batch: (
                    finalized_conditions[
                        graphs_per_batch
                    ]
                )
                for graphs_per_batch in (
                    catalog.graph_counts
                )
            }
        ),
    )


def batch_size_outer_fold_rows(
    result: BatchSizeSensitivityResult,
) -> list[
    dict[
        str,
        object,
    ]
]:
    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for condition_result in result.conditions.values():
        for fold in condition_result.outer_folds:
            rows.append(
                {
                    "graphs_per_batch": (
                        fold.graphs_per_batch
                    ),
                    "outer_fold": (
                        fold.outer_fold
                    ),
                    "train_sample_count": (
                        fold.train_sample_count
                    ),
                    "test_sample_count": (
                        fold.test_sample_count
                    ),
                    "selected_c": (
                        fold.selected_c
                    ),
                    "accuracy": (
                        fold.accuracy
                    ),
                    "macro_f1": (
                        fold.macro_f1
                    ),
                }
            )

    return rows


def batch_size_summary_rows(
    result: BatchSizeSensitivityResult,
) -> list[
    dict[
        str,
        object,
    ]
]:
    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for (
        graphs_per_batch,
        condition_result,
    ) in result.conditions.items():
        evaluation = (
            condition_result.evaluation
        )

        fold_summary = (
            evaluation.fold_summary
        )

        accuracy_bootstrap = (
            condition_result.bootstrap
            .metrics
            .accuracy
        )

        macro_f1_bootstrap = (
            condition_result.bootstrap
            .metrics
            .macro_f1
        )

        comparison = (
            condition_result.reference_comparison
        )

        if comparison is None:
            accuracy_difference = None
            accuracy_difference_ci_low = None
            accuracy_difference_ci_high = None
            macro_f1_difference = None
            macro_f1_difference_ci_low = None
            macro_f1_difference_ci_high = None
        else:
            accuracy_difference_summary = (
                comparison.difference.accuracy
            )

            macro_f1_difference_summary = (
                comparison.difference.macro_f1
            )

            accuracy_difference = (
                accuracy_difference_summary.observed
            )

            accuracy_difference_ci_low = (
                accuracy_difference_summary
                .interval
                .lower
            )

            accuracy_difference_ci_high = (
                accuracy_difference_summary
                .interval
                .upper
            )

            macro_f1_difference = (
                macro_f1_difference_summary.observed
            )

            macro_f1_difference_ci_low = (
                macro_f1_difference_summary
                .interval
                .lower
            )

            macro_f1_difference_ci_high = (
                macro_f1_difference_summary
                .interval
                .upper
            )

        rows.append(
            {
                "graphs_per_batch": (
                    graphs_per_batch
                ),
                "is_reference": (
                    graphs_per_batch
                    == result.reference_graphs_per_batch
                ),
                "outer_fold_count": (
                    fold_summary.fold_count
                ),
                "pooled_accuracy": (
                    evaluation
                    .pooled_metrics
                    .accuracy
                ),
                "pooled_macro_f1": (
                    evaluation
                    .pooled_metrics
                    .macro_f1
                ),
                "confidence_level": (
                    accuracy_bootstrap
                    .interval
                    .confidence_level
                ),
                "accuracy_ci_low": (
                    accuracy_bootstrap
                    .interval
                    .lower
                ),
                "accuracy_ci_high": (
                    accuracy_bootstrap
                    .interval
                    .upper
                ),
                "macro_f1_ci_low": (
                    macro_f1_bootstrap
                    .interval
                    .lower
                ),
                "macro_f1_ci_high": (
                    macro_f1_bootstrap
                    .interval
                    .upper
                ),
                "accuracy_difference_from_reference": (
                    accuracy_difference
                ),
                "accuracy_difference_ci_low": (
                    accuracy_difference_ci_low
                ),
                "accuracy_difference_ci_high": (
                    accuracy_difference_ci_high
                ),
                "macro_f1_difference_from_reference": (
                    macro_f1_difference
                ),
                "macro_f1_difference_ci_low": (
                    macro_f1_difference_ci_low
                ),
                "macro_f1_difference_ci_high": (
                    macro_f1_difference_ci_high
                ),
                "fold_mean_accuracy": (
                    fold_summary
                    .accuracy
                    .mean
                ),
                "fold_sd_accuracy": (
                    fold_summary
                    .accuracy
                    .standard_deviation
                ),
                "fold_mean_macro_f1": (
                    fold_summary
                    .macro_f1
                    .mean
                ),
                "fold_sd_macro_f1": (
                    fold_summary
                    .macro_f1
                    .standard_deviation
                ),
            }
        )

    return rows


def write_batch_size_sensitivity_result(
    layout: AnalysisOutputLayout,
    result: BatchSizeSensitivityResult,
) -> None:
    write_batch_size_outer_folds(
        layout,
        batch_size_outer_fold_rows(
            result
        ),
    )

    write_batch_size_summary(
        layout,
        batch_size_summary_rows(
            result
        ),
    )


__all__ = [
    "BatchSizeConditionResult",
    "BatchSizeOuterFoldResult",
    "BatchSizeSensitivityResult",
    "batch_size_outer_fold_rows",
    "batch_size_summary_rows",
    "run_batch_size_sensitivity",
    "write_batch_size_sensitivity_result",
]