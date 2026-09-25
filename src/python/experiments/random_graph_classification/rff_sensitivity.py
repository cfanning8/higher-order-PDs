# experiments/random_graph_classification/rff_sensitivity.py

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
    RffCatalog,
)
from .output import (
    AnalysisOutputLayout,
    write_rff_outer_folds,
    write_rff_replicate_summary,
    write_rff_summary,
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
class RffOuterFoldResult:
    character_count: int
    replicate_index: int
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
class RffReplicateResult:
    character_count: int
    replicate_index: int
    outer_folds: tuple[
        RffOuterFoldResult,
        ...,
    ]
    predictions: PredictionArray
    evaluation: OutOfFoldEvaluation


@dataclass(
    frozen=True,
    slots=True,
)
class RffDimensionResult:
    character_count: int
    replicates: Mapping[
        int,
        RffReplicateResult,
    ]
    comparison: HierarchicalPairedBootstrapComparison


@dataclass(
    frozen=True,
    slots=True,
)
class RffSensitivityResult:
    sample_keys: tuple[
        BenchmarkSampleKey,
        ...,
    ]
    labels: LabelArray
    cross_validation_plan: NestedCrossValidationPlan
    baseline: BenchmarkMethodResult
    dimensions: Mapping[
        int,
        RffDimensionResult,
    ]


def _rff_feature_matrix(
    catalog: RffCatalog,
    sample_keys: Sequence[
        BenchmarkSampleKey
    ],
    character_count: int,
    replicate_index: int,
) -> FloatMatrix:
    rows = tuple(
        catalog.harmonic_features[
            (
                model_index,
                parameter_condition_index,
                batch_index,
                replicate_index,
                character_count,
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


def _run_rff_replicate(
    character_count: int,
    replicate_index: int,
    features: FloatMatrix,
    labels: LabelArray,
    plan: NestedCrossValidationPlan,
    *,
    c_values: Sequence[float],
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
) -> RffReplicateResult:
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
        RffOuterFoldResult
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
            RffOuterFoldResult(
                character_count=character_count,
                replicate_index=replicate_index,
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

    return RffReplicateResult(
        character_count=character_count,
        replicate_index=replicate_index,
        outer_folds=tuple(
            outer_results
        ),
        predictions=frozen_predictions,
        evaluation=evaluation,
    )


def run_rff_sensitivity(
    catalog: RffCatalog,
    benchmark_result: BenchmarkResult,
    config: RandomGraphClassificationConfig,
) -> RffSensitivityResult:
    sample_keys = benchmark_result.sample_keys
    labels = benchmark_result.labels
    plan = benchmark_result.cross_validation_plan

    baseline = benchmark_result.methods[
        HARMONIC_AGGREGATION_METHOD
    ]

    dimension_results: dict[
        int,
        RffDimensionResult,
    ] = {}

    for character_count in catalog.character_counts:
        replicate_results: dict[
            int,
            RffReplicateResult,
        ] = {}

        for replicate_index in catalog.replicate_indices:
            features = _rff_feature_matrix(
                catalog,
                sample_keys,
                character_count,
                replicate_index,
            )

            replicate_results[
                replicate_index
            ] = _run_rff_replicate(
                character_count,
                replicate_index,
                features,
                labels,
                plan,
                c_values=config.classifier.c_values,
                preprocessing=config.preprocessing,
                classifier=config.classifier,
            )

        comparison = hierarchical_paired_stratified_bootstrap(
            labels,
            tuple(
                replicate_results[
                    replicate_index
                ].predictions
                for replicate_index in (
                    catalog.replicate_indices
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
                config.bootstrap.rff_seed
            ),
        )

        dimension_results[
            character_count
        ] = RffDimensionResult(
            character_count=character_count,
            replicates=MappingProxyType(
                {
                    replicate_index: (
                        replicate_results[
                            replicate_index
                        ]
                    )
                    for replicate_index in (
                        catalog.replicate_indices
                    )
                }
            ),
            comparison=comparison,
        )

    return RffSensitivityResult(
        sample_keys=sample_keys,
        labels=labels,
        cross_validation_plan=plan,
        baseline=baseline,
        dimensions=MappingProxyType(
            {
                character_count: (
                    dimension_results[
                        character_count
                    ]
                )
                for character_count in (
                    catalog.character_counts
                )
            }
        ),
    )


def rff_outer_fold_rows(
    result: RffSensitivityResult,
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

    for dimension_result in result.dimensions.values():
        for replicate_result in (
            dimension_result.replicates.values()
        ):
            for fold in replicate_result.outer_folds:
                rows.append(
                    {
                        "character_count": (
                            fold.character_count
                        ),
                        "replicate_index": (
                            fold.replicate_index
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


def rff_replicate_summary_rows(
    result: RffSensitivityResult,
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
        character_count,
        dimension_result,
    ) in result.dimensions.items():
        for (
            replicate_index,
            replicate_result,
        ) in dimension_result.replicates.items():
            evaluation = (
                replicate_result.evaluation
            )

            fold_summary = (
                evaluation.fold_summary
            )

            rows.append(
                {
                    "character_count": (
                        character_count
                    ),
                    "replicate_index": (
                        replicate_index
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


def rff_summary_rows(
    result: RffSensitivityResult,
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
        character_count,
        dimension_result,
    ) in result.dimensions.items():
        comparison = (
            dimension_result.comparison
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
                "character_count": (
                    character_count
                ),
                "replicate_count": (
                    len(
                        dimension_result.replicates
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
                "accuracy_difference_from_baseline": (
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
                "macro_f1_difference_from_baseline": (
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


def write_rff_sensitivity_result(
    layout: AnalysisOutputLayout,
    result: RffSensitivityResult,
) -> None:
    write_rff_outer_folds(
        layout,
        rff_outer_fold_rows(
            result
        ),
    )

    write_rff_replicate_summary(
        layout,
        rff_replicate_summary_rows(
            result
        ),
    )

    write_rff_summary(
        layout,
        rff_summary_rows(
            result
        ),
    )


__all__ = [
    "RffDimensionResult",
    "RffOuterFoldResult",
    "RffReplicateResult",
    "RffSensitivityResult",
    "rff_outer_fold_rows",
    "rff_replicate_summary_rows",
    "rff_summary_rows",
    "run_rff_sensitivity",
    "write_rff_sensitivity_result",
]