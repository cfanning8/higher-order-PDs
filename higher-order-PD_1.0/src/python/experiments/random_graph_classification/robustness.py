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
    accuracy,
    classification_metrics,
    evaluate_out_of_fold_predictions,
    hierarchical_paired_stratified_bootstrap,
)

from .benchmark import (
    BenchmarkMethodResult,
    BenchmarkOuterFoldResult,
    BenchmarkResult,
    _fit_predict_svm,
)
from .configuration import (
    HARMONIC_AGGREGATION_METHOD,
    ClassifierSettings,
    PreprocessingSettings,
    RandomGraphClassificationConfig,
)
from .data import (
    BenchmarkSampleKey,
    RobustnessCatalog,
    RobustnessFeatureKey,
)
from .output import (
    AnalysisOutputLayout,
    write_robustness_outer_folds,
    write_robustness_summary,
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
class RobustnessOuterFoldResult:
    condition_index: int
    perturbation: str
    severity: float
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
class RobustnessRealizationResult:
    condition_index: int
    perturbation: str
    severity: float
    realization_index: int
    outer_folds: tuple[
        RobustnessOuterFoldResult,
        ...,
    ]
    predictions: PredictionArray
    evaluation: OutOfFoldEvaluation


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessConditionResult:
    condition_index: int
    perturbation: str
    severity: float
    realizations: Mapping[
        int,
        RobustnessRealizationResult,
    ]
    comparison: HierarchicalPairedBootstrapComparison


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessResult:
    sample_keys: tuple[
        BenchmarkSampleKey,
        ...,
    ]
    labels: LabelArray
    cross_validation_plan: NestedCrossValidationPlan
    baseline: BenchmarkMethodResult
    conditions: Mapping[
        int,
        RobustnessConditionResult,
    ]


def _condition_metadata(
    catalog: RobustnessCatalog,
) -> Mapping[
    int,
    tuple[
        str,
        float,
        tuple[int, ...],
    ],
]:
    perturbation_and_severity: dict[
        int,
        tuple[str, float],
    ] = {}

    realizations: dict[
        int,
        set[int],
    ] = {}

    for condition in catalog.conditions.values():
        perturbation_and_severity[
            condition.condition_index
        ] = (
            condition.perturbation,
            condition.severity,
        )

        realizations.setdefault(
            condition.condition_index,
            set(),
        ).add(
            condition.realization_index
        )

    return MappingProxyType(
        {
            condition_index: (
                perturbation_and_severity[
                    condition_index
                ][0],
                perturbation_and_severity[
                    condition_index
                ][1],
                tuple(
                    sorted(
                        realizations[
                            condition_index
                        ]
                    )
                ),
            )
            for condition_index in sorted(
                perturbation_and_severity
            )
        }
    )


def _robustness_feature_matrix(
    catalog: RobustnessCatalog,
    sample_keys: Sequence[
        BenchmarkSampleKey
    ],
    condition_index: int,
    realization_index: int,
) -> FloatMatrix:
    rows = tuple(
        catalog.harmonic_features[
            (
                model_index,
                parameter_condition_index,
                batch_index,
                condition_index,
                realization_index,
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


def _run_robustness_realization(
    condition_index: int,
    perturbation: str,
    severity: float,
    realization_index: int,
    features: FloatMatrix,
    labels: LabelArray,
    plan: NestedCrossValidationPlan,
    baseline_outer_folds: Mapping[
        int,
        BenchmarkOuterFoldResult,
    ],
    *,
    c_values: Sequence[float],
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
) -> RobustnessRealizationResult:
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
        RobustnessOuterFoldResult
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

        baseline_fold = baseline_outer_folds[
            outer_fold.fold_index
        ]

        outer_results.append(
            RobustnessOuterFoldResult(
                condition_index=condition_index,
                perturbation=perturbation,
                severity=severity,
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

    frozen_predictions = np.array(
        predictions,
        dtype=np.int64,
        copy=True,
    )

    frozen_predictions.setflags(
        write=False
    )

    return RobustnessRealizationResult(
        condition_index=condition_index,
        perturbation=perturbation,
        severity=severity,
        realization_index=realization_index,
        outer_folds=tuple(
            outer_results
        ),
        predictions=frozen_predictions,
        evaluation=evaluation,
    )


def run_robustness(
    catalog: RobustnessCatalog,
    benchmark_result: BenchmarkResult,
    config: RandomGraphClassificationConfig,
) -> RobustnessResult:
    sample_keys = benchmark_result.sample_keys
    labels = benchmark_result.labels
    plan = benchmark_result.cross_validation_plan

    baseline = benchmark_result.methods[
        HARMONIC_AGGREGATION_METHOD
    ]

    baseline_outer_folds = (
        _baseline_outer_fold_lookup(
            baseline
        )
    )

    metadata = _condition_metadata(
        catalog
    )

    condition_results: dict[
        int,
        RobustnessConditionResult,
    ] = {}

    for (
        condition_index,
        (
            perturbation,
            severity,
            realization_indices,
        ),
    ) in metadata.items():
        realization_results: dict[
            int,
            RobustnessRealizationResult,
        ] = {}

        for realization_index in realization_indices:
            features = _robustness_feature_matrix(
                catalog,
                sample_keys,
                condition_index,
                realization_index,
            )

            realization_results[
                realization_index
            ] = _run_robustness_realization(
                condition_index,
                perturbation,
                severity,
                realization_index,
                features,
                labels,
                plan,
                baseline_outer_folds,
                c_values=config.classifier.c_values,
                preprocessing=config.preprocessing,
                classifier=config.classifier,
            )

        comparison = hierarchical_paired_stratified_bootstrap(
            labels,
            tuple(
                realization_results[
                    realization_index
                ].predictions
                for realization_index in (
                    realization_indices
                )
            ),
            baseline.predictions,
            resamples=(
                config.bootstrap.resample_count
            ),
            confidence_level=(
                config.bootstrap.confidence_level
            ),
            seed=(
                config.bootstrap.robustness_seed
            ),
        )

        condition_results[
            condition_index
        ] = RobustnessConditionResult(
            condition_index=condition_index,
            perturbation=perturbation,
            severity=severity,
            realizations=MappingProxyType(
                {
                    realization_index: (
                        realization_results[
                            realization_index
                        ]
                    )
                    for realization_index in (
                        realization_indices
                    )
                }
            ),
            comparison=comparison,
        )

    return RobustnessResult(
        sample_keys=sample_keys,
        labels=labels,
        cross_validation_plan=plan,
        baseline=baseline,
        conditions=MappingProxyType(
            {
                condition_index: (
                    condition_results[
                        condition_index
                    ]
                )
                for condition_index in metadata
            }
        ),
    )


def robustness_outer_fold_rows(
    result: RobustnessResult,
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
        for realization_result in (
            condition_result.realizations.values()
        ):
            for fold in realization_result.outer_folds:
                rows.append(
                    {
                        "condition_index": (
                            fold.condition_index
                        ),
                        "perturbation": (
                            fold.perturbation
                        ),
                        "severity": (
                            fold.severity
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


def robustness_summary_rows(
    result: RobustnessResult,
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
        comparison = (
            condition_result.comparison
        )

        absolute_accuracy = (
            comparison.first_mean.accuracy
        )

        absolute_macro_f1 = (
            comparison.first_mean.macro_f1
        )

        accuracy_difference = (
            comparison.difference.accuracy
        )

        macro_f1_difference = (
            comparison.difference.macro_f1
        )

        rows.append(
            {
                "condition_index": (
                    condition_result.condition_index
                ),
                "perturbation": (
                    condition_result.perturbation
                ),
                "severity": (
                    condition_result.severity
                ),
                "realization_count": (
                    len(
                        condition_result.realizations
                    )
                ),
                "mean_accuracy": (
                    absolute_accuracy.observed
                ),
                "mean_macro_f1": (
                    absolute_macro_f1.observed
                ),
                "confidence_level": (
                    absolute_accuracy
                    .interval
                    .confidence_level
                ),
                "accuracy_ci_low": (
                    absolute_accuracy
                    .interval
                    .lower
                ),
                "accuracy_ci_high": (
                    absolute_accuracy
                    .interval
                    .upper
                ),
                "macro_f1_ci_low": (
                    absolute_macro_f1
                    .interval
                    .lower
                ),
                "macro_f1_ci_high": (
                    absolute_macro_f1
                    .interval
                    .upper
                ),
                "accuracy_difference": (
                    accuracy_difference.observed
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
                "macro_f1_difference": (
                    macro_f1_difference.observed
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
                    "hierarchical_paired_bootstrap"
                ),
            }
        )

    return rows


def write_robustness_result(
    layout: AnalysisOutputLayout,
    result: RobustnessResult,
) -> None:
    write_robustness_outer_folds(
        layout,
        robustness_outer_fold_rows(
            result
        ),
    )

    write_robustness_summary(
        layout,
        robustness_summary_rows(
            result
        ),
    )


__all__ = [
    "RobustnessConditionResult",
    "RobustnessOuterFoldResult",
    "RobustnessRealizationResult",
    "RobustnessResult",
    "robustness_outer_fold_rows",
    "robustness_summary_rows",
    "run_robustness",
    "write_robustness_result",
]