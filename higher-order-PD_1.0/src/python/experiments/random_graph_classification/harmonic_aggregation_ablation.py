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
    HierarchicalPairedBootstrapComparison,
    OutOfFoldEvaluation,
    PairedBootstrapComparison,
    accuracy,
    classification_metrics,
    evaluate_out_of_fold_predictions,
    hierarchical_paired_stratified_bootstrap,
    paired_stratified_bootstrap,
)

from .benchmark import (
    BenchmarkMethodResult,
    BenchmarkOuterFoldResult,
    BenchmarkResult,
    _fit_predict_svm,
)
from .configuration import (
    CROSS_OBSERVATION_ABLATION,
    HARMONIC_AGGREGATION_ABLATION_KINDS,
    HARMONIC_AGGREGATION_METHOD,
    LINEAR_ABLATION,
    NO_PREORDER_ABLATION,
    ClassifierSettings,
    PreprocessingSettings,
    RandomGraphClassificationConfig,
)
from .data import (
    AblationCatalog,
    BenchmarkSampleKey,
)
from .output import (
    AnalysisOutputLayout,
    write_ablation_kind_summary,
    write_ablation_outer_folds,
    write_ablation_summary,
)


FloatMatrix: TypeAlias = NDArray[np.float64]
LabelArray: TypeAlias = NDArray[np.int64]
PredictionArray: TypeAlias = NDArray[np.int64]
IndexArray: TypeAlias = NDArray[np.int64]

AblationConditionKey: TypeAlias = tuple[str, int]


_ABLATION_KIND_RANK = {
    ablation_kind: rank
    for rank, ablation_kind in enumerate(
        HARMONIC_AGGREGATION_ABLATION_KINDS
    )
}


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
class AblationOuterFoldResult:
    ablation_kind: str
    realization_index: int
    outer_fold: int
    train_sample_count: int
    test_sample_count: int
    selected_c: float
    accuracy: float
    macro_f1: float
    baseline_accuracy: float
    baseline_macro_f1: float
    accuracy_difference: float
    macro_f1_difference: float


@dataclass(
    frozen=True,
    slots=True,
)
class AblationConditionResult:
    ablation_kind: str
    realization_index: int
    outer_folds: tuple[
        AblationOuterFoldResult,
        ...,
    ]
    predictions: PredictionArray
    evaluation: OutOfFoldEvaluation
    comparison: PairedBootstrapComparison


@dataclass(
    frozen=True,
    slots=True,
)
class HarmonicAggregationAblationResult:
    sample_keys: tuple[
        BenchmarkSampleKey,
        ...,
    ]
    labels: LabelArray
    cross_validation_plan: NestedCrossValidationPlan
    baseline: BenchmarkMethodResult
    conditions: Mapping[
        AblationConditionKey,
        AblationConditionResult,
    ]
    cross_observation_comparison: HierarchicalPairedBootstrapComparison


def _report(
    message: str,
) -> None:
    print(
        message,
        flush=True,
    )


def _ablation_condition_sort_key(
    condition: AblationConditionKey,
) -> tuple[int, int]:
    ablation_kind, realization_index = condition

    return (
        _ABLATION_KIND_RANK[
            ablation_kind
        ],
        realization_index,
    )


def _ablation_condition_keys(
    catalog: AblationCatalog,
) -> tuple[
    AblationConditionKey,
    ...,
]:
    return tuple(
        sorted(
            {
                (
                    condition.ablation_kind,
                    condition.realization_index,
                )
                for condition in catalog.conditions.values()
            },
            key=_ablation_condition_sort_key,
        )
    )


def _ablation_feature_matrix(
    catalog: AblationCatalog,
    sample_keys: Sequence[
        BenchmarkSampleKey
    ],
    ablation_kind: str,
    realization_index: int,
) -> FloatMatrix:
    rows = tuple(
        catalog.harmonic_features[
            (
                model_index,
                parameter_condition_index,
                batch_index,
                ablation_kind,
                realization_index,
            )
        ].values
        for (
            model_index,
            parameter_condition_index,
            batch_index,
        ) in sample_keys
    )

    features = np.vstack(
        rows
    ).astype(
        np.float64,
        copy=False,
    )

    features.setflags(
        write=False
    )

    return features


def _baseline_outer_fold_lookup(
    baseline: BenchmarkMethodResult,
) -> Mapping[
    int,
    BenchmarkOuterFoldResult,
]:
    return MappingProxyType(
        {
            fold.outer_fold: fold
            for fold in baseline.outer_folds
        }
    )


def _benchmark_dependencies(
    benchmark_result: BenchmarkResult,
) -> tuple[
    tuple[
        BenchmarkSampleKey,
        ...,
    ],
    LabelArray,
    NestedCrossValidationPlan,
    BenchmarkMethodResult,
    Mapping[
        int,
        BenchmarkOuterFoldResult,
    ],
]:
    baseline = benchmark_result.methods[
        HARMONIC_AGGREGATION_METHOD
    ]

    return (
        benchmark_result.sample_keys,
        benchmark_result.labels,
        benchmark_result.cross_validation_plan,
        baseline,
        _baseline_outer_fold_lookup(
            baseline
        ),
    )


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


def _run_ablation_condition(
    ablation_kind: str,
    realization_index: int,
    features: FloatMatrix,
    labels: LabelArray,
    plan: NestedCrossValidationPlan,
    baseline: BenchmarkMethodResult,
    baseline_outer_folds: Mapping[
        int,
        BenchmarkOuterFoldResult,
    ],
    *,
    c_values: Sequence[float],
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
) -> AblationConditionResult:
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
        AblationOuterFoldResult
    ] = []

    outer_fold_count = len(
        plan.folds
    )

    for outer_position, outer_fold in enumerate(
        plan.folds,
        start=1,
    ):
        _report(
            "[Ablation] "
            f"{ablation_kind}, realization {realization_index}: "
            f"outer {outer_position}/{outer_fold_count}"
        )

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

        baseline_fold = baseline_outer_folds[
            outer_fold.fold_index
        ]

        outer_results.append(
            AblationOuterFoldResult(
                ablation_kind=ablation_kind,
                realization_index=realization_index,
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
                baseline_accuracy=(
                    baseline_fold.accuracy
                ),
                baseline_macro_f1=(
                    baseline_fold.macro_f1
                ),
                accuracy_difference=(
                    fold_metrics.accuracy
                    - baseline_fold.accuracy
                ),
                macro_f1_difference=(
                    fold_metrics.macro_f1
                    - baseline_fold.macro_f1
                ),
            )
        )

    evaluation = evaluate_out_of_fold_predictions(
        labels,
        predictions,
        outer_test_fold_by_sample(
            plan
        ),
    )

    comparison = paired_stratified_bootstrap(
        labels,
        predictions,
        baseline.predictions,
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

    return AblationConditionResult(
        ablation_kind=ablation_kind,
        realization_index=realization_index,
        outer_folds=tuple(
            outer_results
        ),
        predictions=frozen_predictions,
        evaluation=evaluation,
        comparison=comparison,
    )


def run_harmonic_aggregation_ablation(
    catalog: AblationCatalog,
    benchmark_result: BenchmarkResult,
    config: RandomGraphClassificationConfig,
) -> HarmonicAggregationAblationResult:
    (
        sample_keys,
        labels,
        plan,
        baseline,
        baseline_outer_folds,
    ) = _benchmark_dependencies(
        benchmark_result
    )

    condition_keys = _ablation_condition_keys(
        catalog
    )

    condition_results: dict[
        AblationConditionKey,
        AblationConditionResult,
    ] = {}

    condition_count = len(
        condition_keys
    )

    for condition_position, (
        ablation_kind,
        realization_index,
    ) in enumerate(
        condition_keys,
        start=1,
    ):
        _report(
            "[Ablation] Running condition "
            f"{condition_position}/{condition_count}: "
            f"{ablation_kind}, realization "
            f"{realization_index}"
        )

        features = _ablation_feature_matrix(
            catalog,
            sample_keys,
            ablation_kind,
            realization_index,
        )

        condition_key: AblationConditionKey = (
            ablation_kind,
            realization_index,
        )

        condition_results[
            condition_key
        ] = _run_ablation_condition(
            ablation_kind,
            realization_index,
            features,
            labels,
            plan,
            baseline,
            baseline_outer_folds,
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
                config.bootstrap.ablation_seed
            ),
        )

    cross_observation_conditions = tuple(
        condition_key
        for condition_key in condition_keys
        if (
            condition_key[0]
            == CROSS_OBSERVATION_ABLATION
        )
    )

    cross_observation_predictions = tuple(
        condition_results[
            condition_key
        ].predictions
        for condition_key in cross_observation_conditions
    )

    cross_observation_comparison = (
        hierarchical_paired_stratified_bootstrap(
            labels,
            cross_observation_predictions,
            baseline.predictions,
            resamples=(
                config.bootstrap.resample_count
            ),
            confidence_level=(
                config.bootstrap.confidence_level
            ),
            seed=(
                config.bootstrap.ablation_seed
            ),
        )
    )

    _report(
        "[Ablation] Complete"
    )

    return HarmonicAggregationAblationResult(
        sample_keys=sample_keys,
        labels=labels,
        cross_validation_plan=plan,
        baseline=baseline,
        conditions=MappingProxyType(
            {
                condition_key: (
                    condition_results[
                        condition_key
                    ]
                )
                for condition_key in condition_keys
            }
        ),
        cross_observation_comparison=(
            cross_observation_comparison
        ),
    )


def ablation_outer_fold_rows(
    result: HarmonicAggregationAblationResult,
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

    for condition_key in sorted(
        result.conditions,
        key=_ablation_condition_sort_key,
    ):
        condition_result = (
            result.conditions[
                condition_key
            ]
        )

        for fold in condition_result.outer_folds:
            rows.append(
                {
                    "ablation_kind": (
                        fold.ablation_kind
                    ),
                    "realization_index": (
                        fold.realization_index
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
                    "baseline_accuracy": (
                        fold.baseline_accuracy
                    ),
                    "baseline_macro_f1": (
                        fold.baseline_macro_f1
                    ),
                    "accuracy_difference": (
                        fold.accuracy_difference
                    ),
                    "macro_f1_difference": (
                        fold.macro_f1_difference
                    ),
                }
            )

    return rows


def ablation_summary_rows(
    result: HarmonicAggregationAblationResult,
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

    for condition_key in sorted(
        result.conditions,
        key=_ablation_condition_sort_key,
    ):
        condition_result = (
            result.conditions[
                condition_key
            ]
        )

        comparison = (
            condition_result.comparison
        )

        accuracy_difference = (
            comparison.difference.accuracy
        )

        macro_f1_difference = (
            comparison.difference.macro_f1
        )

        fold_summary = (
            condition_result.evaluation
            .fold_summary
        )

        rows.append(
            {
                "ablation_kind": (
                    condition_result.ablation_kind
                ),
                "realization_index": (
                    condition_result.realization_index
                ),
                "outer_fold_count": (
                    fold_summary.fold_count
                ),
                "pooled_accuracy": (
                    condition_result.evaluation
                    .pooled_metrics
                    .accuracy
                ),
                "pooled_macro_f1": (
                    condition_result.evaluation
                    .pooled_metrics
                    .macro_f1
                ),
                "accuracy_difference": (
                    accuracy_difference.observed
                ),
                "macro_f1_difference": (
                    macro_f1_difference.observed
                ),
                "confidence_level": (
                    accuracy_difference
                    .interval
                    .confidence_level
                ),
                "accuracy_difference_ci_low": (
                    accuracy_difference
                    .interval
                    .lower
                ),
                "accuracy_difference_ci_high": (
                    accuracy_difference
                    .interval
                    .upper
                ),
                "macro_f1_difference_ci_low": (
                    macro_f1_difference
                    .interval
                    .lower
                ),
                "macro_f1_difference_ci_high": (
                    macro_f1_difference
                    .interval
                    .upper
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


def _deterministic_condition(
    result: HarmonicAggregationAblationResult,
    ablation_kind: str,
) -> AblationConditionResult:
    return result.conditions[
        (
            ablation_kind,
            0,
        )
    ]


def ablation_kind_summary_rows(
    result: HarmonicAggregationAblationResult,
) -> list[
    dict[
        str,
        object,
    ]
]:
    cross_comparison = (
        result.cross_observation_comparison
    )

    cross_accuracy = (
        cross_comparison.first_mean.accuracy
    )

    cross_macro_f1 = (
        cross_comparison.first_mean.macro_f1
    )

    cross_accuracy_difference = (
        cross_comparison.difference.accuracy
    )

    cross_macro_f1_difference = (
        cross_comparison.difference.macro_f1
    )

    cross_realization_count = sum(
        1
        for (
            ablation_kind,
            _,
        ) in result.conditions
        if (
            ablation_kind
            == CROSS_OBSERVATION_ABLATION
        )
    )

    rows: list[
        dict[
            str,
            object,
        ]
    ] = [
        {
            "ablation_kind": (
                CROSS_OBSERVATION_ABLATION
            ),
            "realization_count": (
                cross_realization_count
            ),
            "mean_accuracy": (
                cross_accuracy.observed
            ),
            "mean_macro_f1": (
                cross_macro_f1.observed
            ),
            "accuracy_difference": (
                cross_accuracy_difference.observed
            ),
            "macro_f1_difference": (
                cross_macro_f1_difference.observed
            ),
            "confidence_level": (
                cross_accuracy_difference
                .interval
                .confidence_level
            ),
            "accuracy_difference_ci_low": (
                cross_accuracy_difference
                .interval
                .lower
            ),
            "accuracy_difference_ci_high": (
                cross_accuracy_difference
                .interval
                .upper
            ),
            "macro_f1_difference_ci_low": (
                cross_macro_f1_difference
                .interval
                .lower
            ),
            "macro_f1_difference_ci_high": (
                cross_macro_f1_difference
                .interval
                .upper
            ),
            "uncertainty_method": (
                "hierarchical_paired_bootstrap"
            ),
        }
    ]

    for ablation_kind in (
        LINEAR_ABLATION,
        NO_PREORDER_ABLATION,
    ):
        condition = _deterministic_condition(
            result,
            ablation_kind,
        )

        comparison = (
            condition.comparison
        )

        absolute_accuracy = (
            comparison.first.accuracy
        )

        absolute_macro_f1 = (
            comparison.first.macro_f1
        )

        accuracy_difference = (
            comparison.difference.accuracy
        )

        macro_f1_difference = (
            comparison.difference.macro_f1
        )

        rows.append(
            {
                "ablation_kind": (
                    ablation_kind
                ),
                "realization_count": 1,
                "mean_accuracy": (
                    absolute_accuracy.observed
                ),
                "mean_macro_f1": (
                    absolute_macro_f1.observed
                ),
                "accuracy_difference": (
                    accuracy_difference.observed
                ),
                "macro_f1_difference": (
                    macro_f1_difference.observed
                ),
                "confidence_level": (
                    accuracy_difference
                    .interval
                    .confidence_level
                ),
                "accuracy_difference_ci_low": (
                    accuracy_difference
                    .interval
                    .lower
                ),
                "accuracy_difference_ci_high": (
                    accuracy_difference
                    .interval
                    .upper
                ),
                "macro_f1_difference_ci_low": (
                    macro_f1_difference
                    .interval
                    .lower
                ),
                "macro_f1_difference_ci_high": (
                    macro_f1_difference
                    .interval
                    .upper
                ),
                "uncertainty_method": (
                    "paired_bootstrap"
                ),
            }
        )

    return rows


def write_harmonic_aggregation_ablation_result(
    layout: AnalysisOutputLayout,
    result: HarmonicAggregationAblationResult,
) -> None:
    write_ablation_outer_folds(
        layout,
        ablation_outer_fold_rows(
            result
        ),
    )

    write_ablation_summary(
        layout,
        ablation_summary_rows(
            result
        ),
    )

    write_ablation_kind_summary(
        layout,
        ablation_kind_summary_rows(
            result
        ),
    )


__all__ = [
    "AblationConditionResult",
    "AblationOuterFoldResult",
    "HarmonicAggregationAblationResult",
    "ablation_kind_summary_rows",
    "ablation_outer_fold_rows",
    "ablation_summary_rows",
    "run_harmonic_aggregation_ablation",
    "write_harmonic_aggregation_ablation_result",
]