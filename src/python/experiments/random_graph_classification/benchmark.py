from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from core.cross_validation import (
    NestedCrossValidationPlan,
    OuterFold,
    make_nested_stratified_plan,
    outer_test_fold_by_sample,
    select_candidate,
)
from core.statistics import (
    ClassificationBootstrapResult,
    OutOfFoldEvaluation,
    accuracy,
    classification_metrics,
    evaluate_out_of_fold_predictions,
    stratified_bootstrap,
)
from methods.TDA import (
    FittedPersistenceImage,
    FittedPersistenceLandscape,
    PersistenceImageConfig,
    PersistenceLandscapeConfig,
    aggregate_persistence_images,
    aggregate_persistence_landscapes,
    fit_persistence_image,
    fit_persistence_landscape,
    transform_persistence_images,
    transform_persistence_landscapes,
)
from methods.graph_statistics.features import (
    aggregate_graph_statistics,
    compute_graph_statistics,
)

from .configuration import (
    BENCHMARK_METHODS,
    GRAPH_STATISTICS_METHOD,
    HARMONIC_AGGREGATION_METHOD,
    PERSISTENCE_IMAGE_METHOD,
    PERSISTENCE_LANDSCAPE_METHOD,
    PERSLAY_METHOD,
    BenchmarkSettings,
    ClassifierSettings,
    PersLaySettings,
    PreprocessingSettings,
    RandomGraphClassificationConfig,
)
from .data import (
    BenchmarkCatalog,
    BenchmarkSampleKey,
    CommonMetadata,
    PersistenceDiagram,
)
from .output import (
    AnalysisOutputLayout,
    write_benchmark_outer_folds,
    write_benchmark_summary,
)
from .perslay import PerslayModel


IndexKey: TypeAlias = tuple[int, ...]

FloatMatrix: TypeAlias = NDArray[np.float64]
DiagramArray: TypeAlias = NDArray[np.float64]
LabelArray: TypeAlias = NDArray[np.int64]
PredictionArray: TypeAlias = NDArray[np.int64]
IndexArray: TypeAlias = NDArray[np.int64]
StratificationArray: TypeAlias = NDArray[np.int64]

_UINT64_MASK = (1 << 64) - 1


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
class _LandscapeCandidate:
    num_landscapes: int
    resolution: int
    c: float


@dataclass(
    frozen=True,
    slots=True,
)
class _LandscapeRepresentation:
    num_landscapes: int
    resolution: int


@dataclass(
    frozen=True,
    slots=True,
)
class _PersistenceImageCandidate:
    resolution: int
    bandwidth: float
    c: float


@dataclass(
    frozen=True,
    slots=True,
)
class _PersistenceImageRepresentation:
    resolution: int
    bandwidth: float


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkOuterFoldResult:
    method: str
    outer_fold: int
    train_sample_count: int
    test_sample_count: int
    selected_c: float
    selected_num_landscapes: int | None
    selected_landscape_resolution: int | None
    selected_image_resolution: int | None
    selected_image_bandwidth: float | None
    accuracy: float
    macro_f1: float


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkMethodResult:
    method: str
    outer_folds: tuple[
        BenchmarkOuterFoldResult,
        ...,
    ]
    predictions: PredictionArray
    evaluation: OutOfFoldEvaluation
    bootstrap: ClassificationBootstrapResult


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkResult:
    sample_keys: tuple[
        BenchmarkSampleKey,
        ...,
    ]
    labels: LabelArray
    cross_validation_plan: NestedCrossValidationPlan
    methods: Mapping[
        str,
        BenchmarkMethodResult,
    ]


def _report(
    message: str,
) -> None:
    print(
        message,
        flush=True,
    )


def _derived_seed(
    root_seed: int,
    *components: object,
) -> int:
    digest = hashlib.blake2b(
        digest_size=8,
        person=b"rg_perslay_v1",
    )

    digest.update(
        int(
            root_seed
        ).to_bytes(
            8,
            byteorder="little",
            signed=False,
        )
    )

    for component in components:
        encoded = str(
            component
        ).encode(
            "utf-8"
        )

        digest.update(
            len(
                encoded
            ).to_bytes(
                4,
                byteorder="little",
                signed=False,
            )
        )

        digest.update(
            encoded
        )

    return (
        int.from_bytes(
            digest.digest(),
            byteorder="little",
            signed=False,
        )
        &
        _UINT64_MASK
    )


def _torch_seed(
    seed: int,
) -> int:
    return int(
        seed
        %
        ((1 << 63) - 1)
    )


def _benchmark_sample_domain(
    benchmark: BenchmarkCatalog,
) -> tuple[
    tuple[
        BenchmarkSampleKey,
        ...,
    ],
    LabelArray,
    StratificationArray,
]:
    sample_keys = tuple(
        batch.key
        for batch in benchmark.batches
    )

    labels = np.asarray(
        tuple(
            model_index
            for (
                model_index,
                _,
                _,
            ) in sample_keys
        ),
        dtype=np.int64,
    )

    stratum_keys = tuple(
        (
            model_index,
            parameter_condition_index,
        )
        for (
            model_index,
            parameter_condition_index,
            _,
        ) in sample_keys
    )

    stratum_index = {
        stratum: index
        for index, stratum in enumerate(
            dict.fromkeys(
                stratum_keys
            )
        )
    }

    stratification_domain = np.asarray(
        tuple(
            stratum_index[
                stratum
            ]
            for stratum in stratum_keys
        ),
        dtype=np.int64,
    )

    labels.setflags(
        write=False
    )

    stratification_domain.setflags(
        write=False
    )

    return (
        sample_keys,
        labels,
        stratification_domain,
    )


def _svm_pipeline(
    *,
    c: float,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
) -> Pipeline:
    return Pipeline(
        steps=(
            (
                "variance",
                VarianceThreshold(
                    threshold=preprocessing.variance_tolerance,
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    C=c,
                    kernel=classifier.kernel,
                    gamma=classifier.gamma,
                    class_weight=classifier.class_weight,
                    probability=classifier.probability,
                ),
            ),
        )
    )


def _fit_predict_svm(
    train_features: FloatMatrix,
    train_labels: LabelArray,
    test_features: FloatMatrix,
    *,
    c: float,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
) -> PredictionArray:
    pipeline = _svm_pipeline(
        c=c,
        preprocessing=preprocessing,
        classifier=classifier,
    )

    pipeline.fit(
        train_features,
        train_labels,
    )

    return np.asarray(
        pipeline.predict(
            test_features
        ),
        dtype=np.int64,
    )


def _fixed_feature_candidate_score(
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


def _finalize_method_result(
    method: str,
    outer_folds: Sequence[
        BenchmarkOuterFoldResult
    ],
    labels: LabelArray,
    predictions: PredictionArray,
    plan: NestedCrossValidationPlan,
    *,
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
) -> BenchmarkMethodResult:
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

    return BenchmarkMethodResult(
        method=method,
        outer_folds=tuple(
            outer_folds
        ),
        predictions=frozen_predictions,
        evaluation=evaluation,
        bootstrap=bootstrap,
    )


def _run_fixed_feature_benchmark(
    method: str,
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
) -> BenchmarkMethodResult:
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
        BenchmarkOuterFoldResult
    ] = []

    outer_fold_count = len(
        plan.folds
    )

    for outer_position, outer_fold in enumerate(
        plan.folds,
        start=1,
    ):
        _report(
            f"[Benchmark] {method}: outer "
            f"{outer_position}/{outer_fold_count}"
        )

        selection = select_candidate(
            candidates,
            outer_fold,
            lambda candidate, train_indices, validation_indices: (
                _fixed_feature_candidate_score(
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
            BenchmarkOuterFoldResult(
                method=method,
                outer_fold=outer_fold.fold_index,
                train_sample_count=int(
                    outer_fold.train_indices.size
                ),
                test_sample_count=int(
                    outer_fold.test_indices.size
                ),
                selected_c=selected.c,
                selected_num_landscapes=None,
                selected_landscape_resolution=None,
                selected_image_resolution=None,
                selected_image_bandwidth=None,
                accuracy=fold_metrics.accuracy,
                macro_f1=fold_metrics.macro_f1,
            )
        )

    return _finalize_method_result(
        method,
        outer_results,
        labels,
        predictions,
        plan,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_seed=bootstrap_seed,
    )


def _graph_statistics_feature_matrix(
    common: CommonMetadata,
    benchmark: BenchmarkCatalog,
    sample_keys: Sequence[
        BenchmarkSampleKey
    ],
) -> FloatMatrix:
    graphs_per_batch = (
        common.generation
        .benchmark
        .graphs_per_batch
    )

    graph_features_by_sample: dict[
        BenchmarkSampleKey,
        list[
            NDArray[np.float64] | None
        ],
    ] = {
        key: [
            None
            for _ in range(
                graphs_per_batch
            )
        ]
        for key in sample_keys
    }

    total_graph_count = (
        len(
            sample_keys
        )
        *
        graphs_per_batch
    )

    report_interval = max(
        1,
        total_graph_count
        //
        30,
    )

    processed_graph_count = 0

    for graph in benchmark.graphs:
        sample_key: BenchmarkSampleKey = (
            graph.model_index,
            graph.parameter_condition_index,
            graph.batch_index,
        )

        graph_index = graph.graph_index

        edges = np.asarray(
            tuple(
                (
                    edge.u,
                    edge.v,
                )
                for edge in graph.edges
            ),
            dtype=np.int64,
        ).reshape(
            -1,
            2,
        )

        graph_features_by_sample[
            sample_key
        ][
            graph_index
        ] = compute_graph_statistics(
            graph.vertex_count,
            edges,
        ).values

        processed_graph_count += 1

        if (
            processed_graph_count == total_graph_count
            or processed_graph_count
            %
            report_interval
            ==
            0
        ):
            _report(
                "[Benchmark] Graph statistics: "
                f"{processed_graph_count}/{total_graph_count} graphs"
            )

    batch_features: list[
        NDArray[np.float64]
    ] = []

    for sample_key in sample_keys:
        sample_graph_features = (
            graph_features_by_sample[
                sample_key
            ]
        )

        ordered_graph_features: list[
            NDArray[np.float64]
        ] = []

        for graph_index in range(
            graphs_per_batch
        ):
            graph_features = (
                sample_graph_features[
                    graph_index
                ]
            )

            if graph_features is None:
                raise ValueError(
                    "Benchmark sample "
                    f"{sample_key!r} is missing graph "
                    f"{graph_index}."
                )

            ordered_graph_features.append(
                graph_features
            )

        batch_features.append(
            aggregate_graph_statistics(
                np.vstack(
                    ordered_graph_features
                )
            ).values
        )

    result = np.asarray(
        batch_features,
        dtype=np.float64,
    )

    result.setflags(
        write=False
    )

    return result


def _diagram_array(
    diagram: PersistenceDiagram,
) -> DiagramArray:
    rows: list[
        tuple[
            float,
            float,
        ]
    ] = []

    for atom in diagram.atoms:
        multiplicity = int(
            atom.coefficient
        )

        if (
            multiplicity <= 0
            or float(
                multiplicity
            )
            != atom.coefficient
        ):
            raise ValueError(
                "Benchmark persistence coefficient must be a "
                "positive integral multiplicity."
            )

        rows.extend(
            (
                atom.birth,
                atom.death,
            )
            for _ in range(
                multiplicity
            )
        )

    result = np.asarray(
        rows,
        dtype=np.float64,
    ).reshape(
        -1,
        2,
    )

    result.setflags(
        write=False
    )

    return result


def _benchmark_diagrams_by_sample(
    benchmark: BenchmarkCatalog,
    sample_keys: Sequence[
        BenchmarkSampleKey
    ],
    *,
    graphs_per_batch: int,
) -> tuple[
    tuple[
        DiagramArray,
        ...,
    ],
    ...,
]:
    sample_position = {
        key: position
        for position, key in enumerate(
            sample_keys
        )
    }

    diagrams: list[
        list[
            DiagramArray | None
        ]
    ] = [
        [
            None
            for _ in range(
                graphs_per_batch
            )
        ]
        for _ in sample_keys
    ]

    total_graph_count = (
        len(
            sample_keys
        )
        *
        graphs_per_batch
    )

    report_interval = max(
        1,
        total_graph_count
        //
        30,
    )

    converted_graph_count = 0

    for graph_key, diagram in benchmark.diagrams:
        (
            model_index,
            parameter_condition_index,
            batch_index,
            graph_index,
        ) = graph_key

        sample_key: BenchmarkSampleKey = (
            model_index,
            parameter_condition_index,
            batch_index,
        )

        position = sample_position[
            sample_key
        ]

        diagrams[
            position
        ][
            graph_index
        ] = _diagram_array(
            diagram
        )

        converted_graph_count += 1

        if (
            converted_graph_count == total_graph_count
            or converted_graph_count
            %
            report_interval
            ==
            0
        ):
            _report(
                "[Benchmark] Persistence diagrams: "
                f"{converted_graph_count}/{total_graph_count} graphs"
            )

    complete_diagrams: list[
        tuple[
            DiagramArray,
            ...,
        ]
    ] = []

    for sample_key, sample_diagrams in zip(
        sample_keys,
        diagrams,
        strict=True,
    ):
        ordered_diagrams: list[
            DiagramArray
        ] = []

        for graph_index in range(
            graphs_per_batch
        ):
            diagram = sample_diagrams[
                graph_index
            ]

            if diagram is None:
                raise ValueError(
                    "Benchmark sample "
                    f"{sample_key!r} is missing persistence diagram "
                    f"{graph_index}."
                )

            ordered_diagrams.append(
                diagram
            )

        complete_diagrams.append(
            tuple(
                ordered_diagrams
            )
        )

    return tuple(
        complete_diagrams
    )


def _graph_diagrams_for_samples(
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    sample_indices: IndexArray,
) -> tuple[
    DiagramArray,
    ...,
]:
    return tuple(
        diagram
        for sample_index in sample_indices
        for diagram in sample_diagrams[
            int(
                sample_index
            )
        ]
    )


def _landscape_batch_features(
    fitted: FittedPersistenceLandscape,
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    sample_indices: IndexArray,
) -> FloatMatrix:
    rows: list[
        NDArray[np.float64]
    ] = []

    for sample_index in sample_indices:
        diagrams = sample_diagrams[
            int(
                sample_index
            )
        ]

        graph_features = (
            transform_persistence_landscapes(
                fitted,
                diagrams,
            )
        )

        rows.append(
            aggregate_persistence_landscapes(
                graph_features,
                num_landscapes=fitted.num_landscapes,
                resolution=fitted.resolution,
            ).values
        )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


def _image_batch_features(
    fitted: FittedPersistenceImage,
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    sample_indices: IndexArray,
) -> FloatMatrix:
    rows: list[
        NDArray[np.float64]
    ] = []

    for sample_index in sample_indices:
        diagrams = sample_diagrams[
            int(
                sample_index
            )
        ]

        graph_features = transform_persistence_images(
            fitted,
            diagrams,
        )

        rows.append(
            aggregate_persistence_images(
                graph_features,
                bandwidth=fitted.bandwidth,
                resolution_x=fitted.resolution_x,
                resolution_y=fitted.resolution_y,
            ).values
        )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


def _landscape_candidates(
    benchmark: BenchmarkSettings,
    classifier: ClassifierSettings,
) -> tuple[
    _LandscapeCandidate,
    ...,
]:
    return tuple(
        _LandscapeCandidate(
            num_landscapes=num_landscapes,
            resolution=resolution,
            c=float(
                c
            ),
        )
        for num_landscapes in (
            benchmark.persistence_landscape_num_landscapes
        )
        for resolution in (
            benchmark.persistence_landscape_resolutions
        )
        for c in classifier.c_values
    )


def _landscape_representations(
    benchmark: BenchmarkSettings,
) -> tuple[
    _LandscapeRepresentation,
    ...,
]:
    return tuple(
        _LandscapeRepresentation(
            num_landscapes=num_landscapes,
            resolution=resolution,
        )
        for num_landscapes in (
            benchmark.persistence_landscape_num_landscapes
        )
        for resolution in (
            benchmark.persistence_landscape_resolutions
        )
    )


def _persistence_image_candidates(
    benchmark: BenchmarkSettings,
    classifier: ClassifierSettings,
) -> tuple[
    _PersistenceImageCandidate,
    ...,
]:
    return tuple(
        _PersistenceImageCandidate(
            resolution=resolution,
            bandwidth=bandwidth,
            c=float(
                c
            ),
        )
        for resolution in (
            benchmark.persistence_image_square_resolutions
        )
        for bandwidth in (
            benchmark.persistence_image_bandwidths
        )
        for c in classifier.c_values
    )


def _persistence_image_representations(
    benchmark: BenchmarkSettings,
) -> tuple[
    _PersistenceImageRepresentation,
    ...,
]:
    return tuple(
        _PersistenceImageRepresentation(
            resolution=resolution,
            bandwidth=bandwidth,
        )
        for resolution in (
            benchmark.persistence_image_square_resolutions
        )
        for bandwidth in (
            benchmark.persistence_image_bandwidths
        )
    )


def _index_key(
    indices: IndexArray,
) -> IndexKey:
    return tuple(
        int(
            index
        )
        for index in indices
    )


def _landscape_inner_feature_cache(
    outer_fold: OuterFold,
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    *,
    benchmark: BenchmarkSettings,
    outer_position: int,
    outer_fold_count: int,
) -> Mapping[
    tuple[
        IndexKey,
        IndexKey,
        _LandscapeRepresentation,
    ],
    tuple[
        FloatMatrix,
        FloatMatrix,
    ],
]:
    representations = _landscape_representations(
        benchmark
    )

    representation_count = len(
        representations
    )

    inner_fold_count = len(
        outer_fold.inner_folds
    )

    cache: dict[
        tuple[
            IndexKey,
            IndexKey,
            _LandscapeRepresentation,
        ],
        tuple[
            FloatMatrix,
            FloatMatrix,
        ],
    ] = {}

    for inner_position, inner_fold in enumerate(
        outer_fold.inner_folds,
        start=1,
    ):
        training_diagrams = (
            _graph_diagrams_for_samples(
                sample_diagrams,
                inner_fold.train_indices,
            )
        )

        train_key = _index_key(
            inner_fold.train_indices
        )

        validation_key = _index_key(
            inner_fold.validation_indices
        )

        for representation_position, representation in enumerate(
            representations,
            start=1,
        ):
            _report(
                "[Benchmark] Persistence landscape: "
                f"outer {outer_position}/{outer_fold_count}, "
                f"inner {inner_position}/{inner_fold_count}, "
                f"representation "
                f"{representation_position}/{representation_count}"
            )

            fitted = fit_persistence_landscape(
                training_diagrams,
                PersistenceLandscapeConfig(
                    num_landscapes=(
                        representation.num_landscapes
                    ),
                    resolution=representation.resolution,
                ),
            )

            cache[
                (
                    train_key,
                    validation_key,
                    representation,
                )
            ] = (
                _landscape_batch_features(
                    fitted,
                    sample_diagrams,
                    inner_fold.train_indices,
                ),
                _landscape_batch_features(
                    fitted,
                    sample_diagrams,
                    inner_fold.validation_indices,
                ),
            )

    return MappingProxyType(
        cache
    )


def _landscape_candidate_score(
    candidate: _LandscapeCandidate,
    train_indices: IndexArray,
    validation_indices: IndexArray,
    *,
    cache: Mapping[
        tuple[
            IndexKey,
            IndexKey,
            _LandscapeRepresentation,
        ],
        tuple[
            FloatMatrix,
            FloatMatrix,
        ],
    ],
    labels: LabelArray,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
) -> float:
    representation = _LandscapeRepresentation(
        num_landscapes=candidate.num_landscapes,
        resolution=candidate.resolution,
    )

    train_features, validation_features = cache[
        (
            _index_key(
                train_indices
            ),
            _index_key(
                validation_indices
            ),
            representation,
        )
    ]

    predictions = _fit_predict_svm(
        train_features,
        labels[
            train_indices
        ],
        validation_features,
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


def _select_landscape_candidate(
    outer_fold: OuterFold,
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    labels: LabelArray,
    *,
    benchmark: BenchmarkSettings,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
    outer_position: int,
    outer_fold_count: int,
) -> _LandscapeCandidate:
    candidates = _landscape_candidates(
        benchmark,
        classifier,
    )

    cache = _landscape_inner_feature_cache(
        outer_fold,
        sample_diagrams,
        benchmark=benchmark,
        outer_position=outer_position,
        outer_fold_count=outer_fold_count,
    )

    selection = select_candidate(
        candidates,
        outer_fold,
        lambda candidate, train_indices, validation_indices: (
            _landscape_candidate_score(
                candidate,
                train_indices,
                validation_indices,
                cache=cache,
                labels=labels,
                preprocessing=preprocessing,
                classifier=classifier,
            )
        ),
    )

    return selection.selected_candidate


def _run_persistence_landscape_benchmark(
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    labels: LabelArray,
    plan: NestedCrossValidationPlan,
    *,
    benchmark: BenchmarkSettings,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
) -> BenchmarkMethodResult:
    predictions = np.empty(
        labels.shape,
        dtype=np.int64,
    )

    outer_results: list[
        BenchmarkOuterFoldResult
    ] = []

    outer_fold_count = len(
        plan.folds
    )

    for outer_position, outer_fold in enumerate(
        plan.folds,
        start=1,
    ):
        selected = _select_landscape_candidate(
            outer_fold,
            sample_diagrams,
            labels,
            benchmark=benchmark,
            preprocessing=preprocessing,
            classifier=classifier,
            outer_position=outer_position,
            outer_fold_count=outer_fold_count,
        )

        _report(
            "[Benchmark] Persistence landscape: "
            f"outer {outer_position}/{outer_fold_count}, final fit"
        )

        fitted = fit_persistence_landscape(
            _graph_diagrams_for_samples(
                sample_diagrams,
                outer_fold.train_indices,
            ),
            PersistenceLandscapeConfig(
                num_landscapes=selected.num_landscapes,
                resolution=selected.resolution,
            ),
        )

        fold_predictions = _fit_predict_svm(
            _landscape_batch_features(
                fitted,
                sample_diagrams,
                outer_fold.train_indices,
            ),
            labels[
                outer_fold.train_indices
            ],
            _landscape_batch_features(
                fitted,
                sample_diagrams,
                outer_fold.test_indices,
            ),
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
            BenchmarkOuterFoldResult(
                method=PERSISTENCE_LANDSCAPE_METHOD,
                outer_fold=outer_fold.fold_index,
                train_sample_count=int(
                    outer_fold.train_indices.size
                ),
                test_sample_count=int(
                    outer_fold.test_indices.size
                ),
                selected_c=selected.c,
                selected_num_landscapes=selected.num_landscapes,
                selected_landscape_resolution=selected.resolution,
                selected_image_resolution=None,
                selected_image_bandwidth=None,
                accuracy=fold_metrics.accuracy,
                macro_f1=fold_metrics.macro_f1,
            )
        )

    return _finalize_method_result(
        PERSISTENCE_LANDSCAPE_METHOD,
        outer_results,
        labels,
        predictions,
        plan,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_seed=bootstrap_seed,
    )


def _persistence_image_inner_feature_cache(
    outer_fold: OuterFold,
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    *,
    benchmark: BenchmarkSettings,
    outer_position: int,
    outer_fold_count: int,
) -> Mapping[
    tuple[
        IndexKey,
        IndexKey,
        _PersistenceImageRepresentation,
    ],
    tuple[
        FloatMatrix,
        FloatMatrix,
    ],
]:
    representations = _persistence_image_representations(
        benchmark
    )

    representation_count = len(
        representations
    )

    inner_fold_count = len(
        outer_fold.inner_folds
    )

    cache: dict[
        tuple[
            IndexKey,
            IndexKey,
            _PersistenceImageRepresentation,
        ],
        tuple[
            FloatMatrix,
            FloatMatrix,
        ],
    ] = {}

    for inner_position, inner_fold in enumerate(
        outer_fold.inner_folds,
        start=1,
    ):
        training_diagrams = _graph_diagrams_for_samples(
            sample_diagrams,
            inner_fold.train_indices,
        )

        train_key = _index_key(
            inner_fold.train_indices
        )

        validation_key = _index_key(
            inner_fold.validation_indices
        )

        for representation_position, representation in enumerate(
            representations,
            start=1,
        ):
            _report(
                "[Benchmark] Persistence image: "
                f"outer {outer_position}/{outer_fold_count}, "
                f"inner {inner_position}/{inner_fold_count}, "
                f"representation "
                f"{representation_position}/{representation_count}"
            )

            fitted = fit_persistence_image(
                training_diagrams,
                PersistenceImageConfig(
                    bandwidth=representation.bandwidth,
                    resolution_x=representation.resolution,
                    resolution_y=representation.resolution,
                ),
            )

            cache[
                (
                    train_key,
                    validation_key,
                    representation,
                )
            ] = (
                _image_batch_features(
                    fitted,
                    sample_diagrams,
                    inner_fold.train_indices,
                ),
                _image_batch_features(
                    fitted,
                    sample_diagrams,
                    inner_fold.validation_indices,
                ),
            )

    return MappingProxyType(
        cache
    )


def _persistence_image_candidate_score(
    candidate: _PersistenceImageCandidate,
    train_indices: IndexArray,
    validation_indices: IndexArray,
    *,
    cache: Mapping[
        tuple[
            IndexKey,
            IndexKey,
            _PersistenceImageRepresentation,
        ],
        tuple[
            FloatMatrix,
            FloatMatrix,
        ],
    ],
    labels: LabelArray,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
) -> float:
    representation = _PersistenceImageRepresentation(
        resolution=candidate.resolution,
        bandwidth=candidate.bandwidth,
    )

    train_features, validation_features = cache[
        (
            _index_key(
                train_indices
            ),
            _index_key(
                validation_indices
            ),
            representation,
        )
    ]

    predictions = _fit_predict_svm(
        train_features,
        labels[
            train_indices
        ],
        validation_features,
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


def _select_persistence_image_candidate(
    outer_fold: OuterFold,
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    labels: LabelArray,
    *,
    benchmark: BenchmarkSettings,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
    outer_position: int,
    outer_fold_count: int,
) -> _PersistenceImageCandidate:
    candidates = _persistence_image_candidates(
        benchmark,
        classifier,
    )

    cache = _persistence_image_inner_feature_cache(
        outer_fold,
        sample_diagrams,
        benchmark=benchmark,
        outer_position=outer_position,
        outer_fold_count=outer_fold_count,
    )

    selection = select_candidate(
        candidates,
        outer_fold,
        lambda candidate, train_indices, validation_indices: (
            _persistence_image_candidate_score(
                candidate,
                train_indices,
                validation_indices,
                cache=cache,
                labels=labels,
                preprocessing=preprocessing,
                classifier=classifier,
            )
        ),
    )

    return selection.selected_candidate


def _run_persistence_image_benchmark(
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    labels: LabelArray,
    plan: NestedCrossValidationPlan,
    *,
    benchmark: BenchmarkSettings,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
) -> BenchmarkMethodResult:
    predictions = np.empty(
        labels.shape,
        dtype=np.int64,
    )

    outer_results: list[
        BenchmarkOuterFoldResult
    ] = []

    outer_fold_count = len(
        plan.folds
    )

    for outer_position, outer_fold in enumerate(
        plan.folds,
        start=1,
    ):
        selected = _select_persistence_image_candidate(
            outer_fold,
            sample_diagrams,
            labels,
            benchmark=benchmark,
            preprocessing=preprocessing,
            classifier=classifier,
            outer_position=outer_position,
            outer_fold_count=outer_fold_count,
        )

        _report(
            "[Benchmark] Persistence image: "
            f"outer {outer_position}/{outer_fold_count}, final fit"
        )

        fitted = fit_persistence_image(
            _graph_diagrams_for_samples(
                sample_diagrams,
                outer_fold.train_indices,
            ),
            PersistenceImageConfig(
                bandwidth=selected.bandwidth,
                resolution_x=selected.resolution,
                resolution_y=selected.resolution,
            ),
        )

        fold_predictions = _fit_predict_svm(
            _image_batch_features(
                fitted,
                sample_diagrams,
                outer_fold.train_indices,
            ),
            labels[
                outer_fold.train_indices
            ],
            _image_batch_features(
                fitted,
                sample_diagrams,
                outer_fold.test_indices,
            ),
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
            BenchmarkOuterFoldResult(
                method=PERSISTENCE_IMAGE_METHOD,
                outer_fold=outer_fold.fold_index,
                train_sample_count=int(
                    outer_fold.train_indices.size
                ),
                test_sample_count=int(
                    outer_fold.test_indices.size
                ),
                selected_c=selected.c,
                selected_num_landscapes=None,
                selected_landscape_resolution=None,
                selected_image_resolution=selected.resolution,
                selected_image_bandwidth=selected.bandwidth,
                accuracy=fold_metrics.accuracy,
                macro_f1=fold_metrics.macro_f1,
            )
        )

    return _finalize_method_result(
        PERSISTENCE_IMAGE_METHOD,
        outer_results,
        labels,
        predictions,
        plan,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_seed=bootstrap_seed,
    )


def _padded_perslay_diagrams(
    diagrams: Sequence[
        DiagramArray
    ],
    *,
    device: torch.device,
) -> torch.Tensor:
    maximum_point_count = max(
        (
            int(
                diagram.shape[
                    0
                ]
            )
            for diagram in diagrams
        ),
        default=0,
    )

    padded = np.zeros(
        (
            len(
                diagrams
            ),
            max(
                1,
                maximum_point_count,
            ),
            3,
        ),
        dtype=np.float32,
    )

    for diagram_index, diagram in enumerate(
        diagrams
    ):
        point_count = int(
            diagram.shape[
                0
            ]
        )

        if point_count == 0:
            continue

        padded[
            diagram_index,
            :point_count,
            :2,
        ] = np.asarray(
            diagram,
            dtype=np.float32,
        )

        padded[
            diagram_index,
            :point_count,
            2,
        ] = 1.0

    return torch.as_tensor(
        padded,
        dtype=torch.float32,
        device=device,
    )


def _perslay_sample_bounds(
    diagrams: Sequence[
        DiagramArray
    ],
) -> tuple[
    float,
    float,
]:
    nonempty = tuple(
        diagram
        for diagram in diagrams
        if diagram.shape[
            0
        ]
        !=
        0
    )

    if not nonempty:
        return (
            0.0,
            1.0,
        )

    minimum_birth = min(
        float(
            np.min(
                diagram[
                    :,
                    0,
                ]
            )
        )
        for diagram in nonempty
    )

    maximum_death = max(
        float(
            np.max(
                diagram[
                    :,
                    1,
                ]
            )
        )
        for diagram in nonempty
    )

    if maximum_death <= minimum_birth:
        return (
            minimum_birth,
            minimum_birth
            +
            1.0,
        )

    return (
        minimum_birth,
        maximum_death,
    )


def _graphs_per_sample(
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    sample_indices: IndexArray,
) -> int:
    if sample_indices.size == 0:
        raise ValueError(
            "PersLay sample indices must not be empty."
        )

    first_sample_index = int(
        sample_indices[
            0
        ]
    )

    graph_count = len(
        sample_diagrams[
            first_sample_index
        ]
    )

    if graph_count <= 0:
        raise ValueError(
            "PersLay benchmark samples must contain at least one graph."
        )

    for sample_index in sample_indices:
        current_graph_count = len(
            sample_diagrams[
                int(
                    sample_index
                )
            ]
        )

        if current_graph_count != graph_count:
            raise ValueError(
                "PersLay benchmark samples must contain the same "
                "number of graphs."
            )

    return graph_count


def _make_perslay_model(
    training_diagrams: Sequence[
        DiagramArray
    ],
    *,
    class_count: int,
    perslay: PersLaySettings,
    seed: int,
    device: torch.device,
) -> PerslayModel:
    torch_seed = _torch_seed(
        seed
    )

    torch.manual_seed(
        torch_seed
    )

    torch.cuda.manual_seed_all(
        torch_seed
    )

    lower, upper = _perslay_sample_bounds(
        training_diagrams
    )

    samples = np.linspace(
        lower,
        upper,
        perslay.feature_count,
        dtype=np.float32,
    )

    perslay_parameters = [
        {
            "layer": "Landscape",
            "layer_train": True,
            "lsample_num": perslay.feature_count,
            "lsample_init": samples,
            "perm_op": "sum",
            "keep": 1,
            "pweight": "power",
            "pweight_init": np.ones(
                (
                    1,
                ),
                dtype=np.float32,
            ),
            "pweight_train": True,
            "pweight_power": 1,
            "final_model": "identity",
        }
    ]

    rho = nn.Linear(
        perslay.feature_count,
        class_count,
    )

    nn.init.xavier_uniform_(
        rho.weight
    )

    nn.init.zeros_(
        rho.bias
    )

    return PerslayModel(
        name="random_graph_classification_perslay",
        diagdim=2,
        perslay_parameters=perslay_parameters,
        rho=rho,
    ).to(
        device
    )


def _fit_perslay(
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    labels: LabelArray,
    train_indices: IndexArray,
    *,
    perslay: PersLaySettings,
    seed: int,
) -> PerslayModel:
    device = torch.device(
        "cuda"
    )

    graph_count = _graphs_per_sample(
        sample_diagrams,
        train_indices,
    )

    training_diagrams = _graph_diagrams_for_samples(
        sample_diagrams,
        train_indices,
    )

    diagrams = _padded_perslay_diagrams(
        training_diagrams,
        device=device,
    )

    train_labels = labels[
        train_indices
    ]

    classes = np.unique(
        train_labels
    )

    class_positions = {
        int(
            class_label
        ): position
        for position, class_label in enumerate(
            classes
        )
    }

    encoded_labels = torch.as_tensor(
        np.asarray(
            tuple(
                class_positions[
                    int(
                        label
                    )
                ]
                for label in train_labels
            ),
            dtype=np.int64,
        ),
        dtype=torch.long,
        device=device,
    )

    model = _make_perslay_model(
        training_diagrams,
        class_count=int(
            classes.size
        ),
        perslay=perslay,
        seed=seed,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=perslay.learning_rate,
    )

    loss_function = nn.CrossEntropyLoss()

    for _ in range(
        perslay.epochs
    ):
        model.train()

        optimizer.zero_grad()

        graph_representations = (
            model.compute_representations(
                [
                    diagrams
                ],
                training=True,
            )
        )

        if (
            graph_representations.shape[
                0
            ]
            !=
            int(
                train_indices.size
            )
            *
            graph_count
        ):
            raise ValueError(
                "PersLay graph representation count does not match "
                "the benchmark training samples."
            )

        if (
            graph_representations.shape[
                1
            ]
            !=
            perslay.feature_count
        ):
            raise ValueError(
                "PersLay graph representation dimension "
                f"{graph_representations.shape[1]} does not match "
                f"the configured dimension {perslay.feature_count}."
            )

        batch_representations = (
            graph_representations
            .reshape(
                int(
                    train_indices.size
                ),
                graph_count,
                perslay.feature_count,
            )
            .mean(
                dim=1
            )
        )

        logits = model.rho(
            batch_representations
        )

        loss = loss_function(
            logits,
            encoded_labels,
        )

        loss.backward()

        optimizer.step()

    return model


def _perslay_graph_features(
    model: PerslayModel,
    diagrams: Sequence[
        DiagramArray
    ],
    *,
    feature_count: int,
) -> FloatMatrix:
    device = next(
        model.parameters()
    ).device

    padded = _padded_perslay_diagrams(
        diagrams,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        representations = (
            model.compute_representations(
                [
                    padded
                ],
                training=False,
            )
        )

    result = (
        representations
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    if (
        result.ndim != 2
        or result.shape[
            1
        ]
        != feature_count
    ):
        raise ValueError(
            "PersLay graph feature dimension does not match "
            f"the configured dimension {feature_count}."
        )

    result.setflags(
        write=False
    )

    return result


def _perslay_batch_features(
    model: PerslayModel,
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    sample_indices: IndexArray,
    *,
    feature_count: int,
) -> FloatMatrix:
    graph_count = _graphs_per_sample(
        sample_diagrams,
        sample_indices,
    )

    diagrams = _graph_diagrams_for_samples(
        sample_diagrams,
        sample_indices,
    )

    graph_features = _perslay_graph_features(
        model,
        diagrams,
        feature_count=feature_count,
    )

    expected_graph_count = (
        int(
            sample_indices.size
        )
        *
        graph_count
    )

    if (
        graph_features.shape[
            0
        ]
        != expected_graph_count
    ):
        raise ValueError(
            "PersLay graph feature count does not match "
            "the selected benchmark samples."
        )

    result = (
        graph_features
        .reshape(
            int(
                sample_indices.size
            ),
            graph_count,
            feature_count,
        )
        .mean(
            axis=1,
            dtype=np.float64,
        )
    )

    result = np.asarray(
        result,
        dtype=np.float64,
    )

    result.setflags(
        write=False
    )

    return result


def _perslay_inner_feature_cache(
    outer_fold: OuterFold,
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    labels: LabelArray,
    *,
    perslay: PersLaySettings,
    outer_position: int,
    outer_fold_count: int,
) -> Mapping[
    tuple[
        IndexKey,
        IndexKey,
    ],
    tuple[
        FloatMatrix,
        FloatMatrix,
    ],
]:
    cache: dict[
        tuple[
            IndexKey,
            IndexKey,
        ],
        tuple[
            FloatMatrix,
            FloatMatrix,
        ],
    ] = {}

    inner_fold_count = len(
        outer_fold.inner_folds
    )

    for inner_position, inner_fold in enumerate(
        outer_fold.inner_folds,
        start=1,
    ):
        _report(
            "[Benchmark] PersLay: "
            f"outer {outer_position}/{outer_fold_count}, "
            f"inner {inner_position}/{inner_fold_count}"
        )

        model = _fit_perslay(
            sample_diagrams,
            labels,
            inner_fold.train_indices,
            perslay=perslay,
            seed=_derived_seed(
                perslay.seed,
                outer_fold.fold_index,
                "inner",
                inner_position - 1,
            ),
        )

        cache[
            (
                _index_key(
                    inner_fold.train_indices
                ),
                _index_key(
                    inner_fold.validation_indices
                ),
            )
        ] = (
            _perslay_batch_features(
                model,
                sample_diagrams,
                inner_fold.train_indices,
                feature_count=perslay.feature_count,
            ),
            _perslay_batch_features(
                model,
                sample_diagrams,
                inner_fold.validation_indices,
                feature_count=perslay.feature_count,
            ),
        )

    return MappingProxyType(
        cache
    )


def _perslay_candidate_score(
    candidate: _SvmCandidate,
    train_indices: IndexArray,
    validation_indices: IndexArray,
    *,
    cache: Mapping[
        tuple[
            IndexKey,
            IndexKey,
        ],
        tuple[
            FloatMatrix,
            FloatMatrix,
        ],
    ],
    labels: LabelArray,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
) -> float:
    train_features, validation_features = cache[
        (
            _index_key(
                train_indices
            ),
            _index_key(
                validation_indices
            ),
        )
    ]

    predictions = _fit_predict_svm(
        train_features,
        labels[
            train_indices
        ],
        validation_features,
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


def _select_perslay_candidate(
    outer_fold: OuterFold,
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    labels: LabelArray,
    *,
    perslay: PersLaySettings,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
    outer_position: int,
    outer_fold_count: int,
) -> _SvmCandidate:
    candidates = tuple(
        _SvmCandidate(
            c=float(
                c
            )
        )
        for c in classifier.c_values
    )

    cache = _perslay_inner_feature_cache(
        outer_fold,
        sample_diagrams,
        labels,
        perslay=perslay,
        outer_position=outer_position,
        outer_fold_count=outer_fold_count,
    )

    selection = select_candidate(
        candidates,
        outer_fold,
        lambda candidate, train_indices, validation_indices: (
            _perslay_candidate_score(
                candidate,
                train_indices,
                validation_indices,
                cache=cache,
                labels=labels,
                preprocessing=preprocessing,
                classifier=classifier,
            )
        ),
    )

    return selection.selected_candidate


def _run_perslay_benchmark(
    sample_diagrams: Sequence[
        Sequence[
            DiagramArray
        ]
    ],
    labels: LabelArray,
    plan: NestedCrossValidationPlan,
    *,
    perslay: PersLaySettings,
    preprocessing: PreprocessingSettings,
    classifier: ClassifierSettings,
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
) -> BenchmarkMethodResult:
    predictions = np.empty(
        labels.shape,
        dtype=np.int64,
    )

    outer_results: list[
        BenchmarkOuterFoldResult
    ] = []

    outer_fold_count = len(
        plan.folds
    )

    for outer_position, outer_fold in enumerate(
        plan.folds,
        start=1,
    ):
        selected = _select_perslay_candidate(
            outer_fold,
            sample_diagrams,
            labels,
            perslay=perslay,
            preprocessing=preprocessing,
            classifier=classifier,
            outer_position=outer_position,
            outer_fold_count=outer_fold_count,
        )

        _report(
            "[Benchmark] PersLay: "
            f"outer {outer_position}/{outer_fold_count}, final fit"
        )

        model = _fit_perslay(
            sample_diagrams,
            labels,
            outer_fold.train_indices,
            perslay=perslay,
            seed=_derived_seed(
                perslay.seed,
                outer_fold.fold_index,
                "outer",
            ),
        )

        train_features = _perslay_batch_features(
            model,
            sample_diagrams,
            outer_fold.train_indices,
            feature_count=perslay.feature_count,
        )

        test_features = _perslay_batch_features(
            model,
            sample_diagrams,
            outer_fold.test_indices,
            feature_count=perslay.feature_count,
        )

        fold_predictions = _fit_predict_svm(
            train_features,
            labels[
                outer_fold.train_indices
            ],
            test_features,
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
            BenchmarkOuterFoldResult(
                method=PERSLAY_METHOD,
                outer_fold=outer_fold.fold_index,
                train_sample_count=int(
                    outer_fold.train_indices.size
                ),
                test_sample_count=int(
                    outer_fold.test_indices.size
                ),
                selected_c=selected.c,
                selected_num_landscapes=None,
                selected_landscape_resolution=None,
                selected_image_resolution=None,
                selected_image_bandwidth=None,
                accuracy=fold_metrics.accuracy,
                macro_f1=fold_metrics.macro_f1,
            )
        )

    return _finalize_method_result(
        PERSLAY_METHOD,
        outer_results,
        labels,
        predictions,
        plan,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_seed=bootstrap_seed,
    )


def _harmonic_feature_matrix(
    common: CommonMetadata,
    benchmark: BenchmarkCatalog,
    sample_keys: Sequence[
        BenchmarkSampleKey
    ],
) -> FloatMatrix:
    result = np.vstack(
        tuple(
            benchmark.harmonic_features[
                key
            ].values
            for key in sample_keys
        )
    ).astype(
        np.float64,
        copy=False,
    )

    expected_feature_count = (
        2
        *
        common.generation
        .random_features
        .baseline_character_count
    )

    if (
        result.shape[
            1
        ]
        != expected_feature_count
    ):
        raise ValueError(
            "Benchmark harmonic-aggregation feature dimension "
            f"{result.shape[1]} does not match the expected "
            f"dimension {expected_feature_count}."
        )

    result.setflags(
        write=False
    )

    return result


def run_benchmark(
    common: CommonMetadata,
    benchmark: BenchmarkCatalog,
    config: RandomGraphClassificationConfig,
) -> BenchmarkResult:
    (
        sample_keys,
        labels,
        stratification_domain,
    ) = _benchmark_sample_domain(
        benchmark
    )

    _report(
        "[Benchmark] Creating nested cross-validation plan"
    )

    plan = make_nested_stratified_plan(
        labels,
        stratification_domain=stratification_domain,
        outer_folds=(
            config.cross_validation.outer_fold_count
        ),
        inner_folds=(
            config.cross_validation.inner_fold_count
        ),
        outer_seed=(
            config.cross_validation.outer_seed
        ),
        inner_seed=(
            config.cross_validation.inner_seed
        ),
    )

    _report(
        "[Benchmark] Computing graph-statistics features"
    )

    graph_statistics_features = (
        _graph_statistics_feature_matrix(
            common,
            benchmark,
            sample_keys,
        )
    )

    _report(
        "[Benchmark] Reading persistence diagrams"
    )

    sample_diagrams = _benchmark_diagrams_by_sample(
        benchmark,
        sample_keys,
        graphs_per_batch=(
            common.generation
            .benchmark
            .graphs_per_batch
        ),
    )

    _report(
        "[Benchmark] Aligning harmonic-aggregation features"
    )

    harmonic_features = _harmonic_feature_matrix(
        common,
        benchmark,
        sample_keys,
    )

    bootstrap_resamples = (
        config.bootstrap.resample_count
    )

    bootstrap_confidence_level = (
        config.bootstrap.confidence_level
    )

    bootstrap_seed = (
        config.bootstrap.benchmark_seed
    )

    _report(
        "[Benchmark] Evaluating graph statistics"
    )

    graph_statistics_result = (
        _run_fixed_feature_benchmark(
            GRAPH_STATISTICS_METHOD,
            graph_statistics_features,
            labels,
            plan,
            c_values=config.classifier.c_values,
            preprocessing=config.preprocessing,
            classifier=config.classifier,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence_level=(
                bootstrap_confidence_level
            ),
            bootstrap_seed=bootstrap_seed,
        )
    )

    _report(
        "[Benchmark] Evaluating persistence landscape"
    )

    landscape_result = (
        _run_persistence_landscape_benchmark(
            sample_diagrams,
            labels,
            plan,
            benchmark=config.benchmark,
            preprocessing=config.preprocessing,
            classifier=config.classifier,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence_level=(
                bootstrap_confidence_level
            ),
            bootstrap_seed=bootstrap_seed,
        )
    )

    _report(
        "[Benchmark] Evaluating persistence image"
    )

    image_result = (
        _run_persistence_image_benchmark(
            sample_diagrams,
            labels,
            plan,
            benchmark=config.benchmark,
            preprocessing=config.preprocessing,
            classifier=config.classifier,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence_level=(
                bootstrap_confidence_level
            ),
            bootstrap_seed=bootstrap_seed,
        )
    )

    _report(
        "[Benchmark] Evaluating PersLay"
    )

    perslay_result = (
        _run_perslay_benchmark(
            sample_diagrams,
            labels,
            plan,
            perslay=config.perslay,
            preprocessing=config.preprocessing,
            classifier=config.classifier,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence_level=(
                bootstrap_confidence_level
            ),
            bootstrap_seed=bootstrap_seed,
        )
    )

    _report(
        "[Benchmark] Evaluating harmonic aggregation"
    )

    harmonic_result = (
        _run_fixed_feature_benchmark(
            HARMONIC_AGGREGATION_METHOD,
            harmonic_features,
            labels,
            plan,
            c_values=config.classifier.c_values,
            preprocessing=config.preprocessing,
            classifier=config.classifier,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence_level=(
                bootstrap_confidence_level
            ),
            bootstrap_seed=bootstrap_seed,
        )
    )

    method_results = {
        GRAPH_STATISTICS_METHOD: (
            graph_statistics_result
        ),
        PERSISTENCE_LANDSCAPE_METHOD: (
            landscape_result
        ),
        PERSISTENCE_IMAGE_METHOD: (
            image_result
        ),
        PERSLAY_METHOD: (
            perslay_result
        ),
        HARMONIC_AGGREGATION_METHOD: (
            harmonic_result
        ),
    }

    methods = MappingProxyType(
        {
            method: method_results[
                method
            ]
            for method in BENCHMARK_METHODS
        }
    )

    _report(
        "[Benchmark] Complete"
    )

    return BenchmarkResult(
        sample_keys=sample_keys,
        labels=labels,
        cross_validation_plan=plan,
        methods=methods,
    )


def benchmark_outer_fold_rows(
    result: BenchmarkResult,
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

    for method in BENCHMARK_METHODS:
        for fold in (
            result.methods[
                method
            ].outer_folds
        ):
            rows.append(
                {
                    "method": fold.method,
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
                    "selected_num_landscapes": (
                        fold.selected_num_landscapes
                    ),
                    "selected_landscape_resolution": (
                        fold.selected_landscape_resolution
                    ),
                    "selected_image_resolution": (
                        fold.selected_image_resolution
                    ),
                    "selected_image_bandwidth": (
                        fold.selected_image_bandwidth
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


def benchmark_summary_rows(
    result: BenchmarkResult,
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

    for method in BENCHMARK_METHODS:
        method_result = result.methods[
            method
        ]

        pooled = (
            method_result.evaluation
            .pooled_metrics
        )

        fold_summary = (
            method_result.evaluation
            .fold_summary
        )

        accuracy_bootstrap = (
            method_result.bootstrap
            .metrics
            .accuracy
        )

        macro_f1_bootstrap = (
            method_result.bootstrap
            .metrics
            .macro_f1
        )

        rows.append(
            {
                "method": method,
                "sample_count": (
                    method_result.evaluation.sample_count
                ),
                "outer_fold_count": (
                    fold_summary.fold_count
                ),
                "pooled_accuracy": (
                    pooled.accuracy
                ),
                "pooled_macro_f1": (
                    pooled.macro_f1
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


def write_benchmark_result(
    layout: AnalysisOutputLayout,
    result: BenchmarkResult,
) -> None:
    write_benchmark_outer_folds(
        layout,
        benchmark_outer_fold_rows(
            result
        ),
    )

    write_benchmark_summary(
        layout,
        benchmark_summary_rows(
            result
        ),
    )


__all__ = [
    "BenchmarkMethodResult",
    "BenchmarkOuterFoldResult",
    "BenchmarkResult",
    "benchmark_outer_fold_rows",
    "benchmark_summary_rows",
    "run_benchmark",
    "write_benchmark_result",
]