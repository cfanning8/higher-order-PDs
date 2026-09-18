from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from core.cross_validation import (
    NestedCrossValidationPlan,
    OuterFold,
    make_nested_stratified_plan,
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
    BenchmarkSettings,
    ClassifierSettings,
    PreprocessingSettings,
    RealWorldDataConfig,
)
from .data import (
    DiagramArray,
    RealWorldData,
    RealWorldDataset,
)


FloatMatrix: TypeAlias = NDArray[np.float64]
LabelArray: TypeAlias = NDArray[np.int64]
PredictionArray: TypeAlias = NDArray[np.int64]
IndexArray: TypeAlias = NDArray[np.int64]
IndexKey: TypeAlias = tuple[int, ...]

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
class _BagPartition:
    graph_indices: tuple[
        tuple[int, ...],
        ...,
    ]
    labels: LabelArray


@dataclass(
    frozen=True,
    slots=True,
)
class BagPrediction:
    dataset: str
    method: str
    bag_size: int
    outer_fold: int
    bag_index: int
    graph_indices: tuple[int, ...]
    label: int
    prediction: int


@dataclass(
    frozen=True,
    slots=True,
)
class OuterFoldResult:
    dataset: str
    method: str
    bag_size: int
    outer_fold: int
    train_graph_count: int
    test_graph_count: int
    train_bag_count: int
    test_bag_count: int
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
class MethodBagSizeResult:
    dataset: str
    method: str
    bag_size: int
    outer_folds: tuple[
        OuterFoldResult,
        ...,
    ]
    predictions: tuple[
        BagPrediction,
        ...,
    ]
    evaluation: OutOfFoldEvaluation
    bootstrap: ClassificationBootstrapResult


@dataclass(
    frozen=True,
    slots=True,
)
class DatasetResult:
    dataset: str
    cross_validation_plan: NestedCrossValidationPlan
    methods: Mapping[
        str,
        Mapping[
            int,
            MethodBagSizeResult,
        ],
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class RealWorldExperimentResult:
    datasets: Mapping[
        str,
        DatasetResult,
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
        person=b"rw_mil_v2",
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


def _index_key(
    indices: IndexArray,
) -> IndexKey:
    return tuple(
        int(
            index
        )
        for index in indices
    )


def _partition_seed(
    config: RealWorldDataConfig,
    dataset: RealWorldDataset,
    *,
    outer_fold: int,
    partition: str,
    indices: IndexArray,
) -> int:
    return _derived_seed(
        config.mil.bag_seed,
        dataset.metadata.name,
        outer_fold,
        partition,
        _index_key(
            indices
        ),
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
                    threshold=(
                        preprocessing.variance_tolerance
                    ),
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


def _majority_class_predictions(
    train_labels: LabelArray,
    test_sample_count: int,
) -> PredictionArray:
    if train_labels.ndim != 1:
        raise ValueError(
            "Training labels must be one-dimensional."
        )

    if train_labels.size == 0:
        raise ValueError(
            "Cannot construct majority-class predictions "
            "from an empty training-label array."
        )

    if test_sample_count < 0:
        raise ValueError(
            "Test sample count must be nonnegative."
        )

    classes, counts = np.unique(
        train_labels,
        return_counts=True,
    )

    maximum_count = int(
        np.max(
            counts
        )
    )

    majority_class = int(
        np.min(
            classes[
                counts
                ==
                maximum_count
            ]
        )
    )

    return np.full(
        test_sample_count,
        majority_class,
        dtype=np.int64,
    )


def _has_retained_feature(
    train_features: FloatMatrix,
    *,
    threshold: float,
) -> bool:
    if train_features.ndim != 2:
        raise ValueError(
            "Training features must be two-dimensional."
        )

    if train_features.shape[
        0
    ] == 0:
        raise ValueError(
            "Training features must contain at least one sample."
        )

    if train_features.shape[
        1
    ] == 0:
        return False

    variances = np.var(
        train_features,
        axis=0,
        dtype=np.float64,
    )

    return bool(
        np.any(
            variances
            >
            threshold
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
    if train_features.ndim != 2:
        raise ValueError(
            "Training features must be two-dimensional."
        )

    if test_features.ndim != 2:
        raise ValueError(
            "Test features must be two-dimensional."
        )

    if train_labels.ndim != 1:
        raise ValueError(
            "Training labels must be one-dimensional."
        )

    if (
        train_features.shape[
            0
        ]
        !=
        train_labels.size
    ):
        raise ValueError(
            "Training feature and label counts must agree."
        )

    if (
        train_features.shape[
            1
        ]
        !=
        test_features.shape[
            1
        ]
    ):
        raise ValueError(
            "Training and test feature dimensions must agree."
        )

    if not _has_retained_feature(
        train_features,
        threshold=(
            preprocessing.variance_tolerance
        ),
    ):
        return _majority_class_predictions(
            train_labels,
            int(
                test_features.shape[
                    0
                ]
            ),
        )

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


def _make_bag_partition(
    dataset: RealWorldDataset,
    graph_indices: IndexArray,
    *,
    bag_size: int,
    seed: int,
) -> _BagPartition:
    classes = np.unique(
        dataset.labels
    )

    bag_graph_indices: list[
        tuple[int, ...]
    ] = []

    bag_labels: list[int] = []

    for class_label_value in classes:
        class_label = int(
            class_label_value
        )

        class_indices = np.asarray(
            tuple(
                int(
                    graph_index
                )
                for graph_index in graph_indices
                if dataset.labels[
                    int(
                        graph_index
                    )
                ]
                ==
                class_label
            ),
            dtype=np.int64,
        )

        bag_count = (
            int(
                class_indices.size
            )
            //
            bag_size
        )

        if bag_count == 0:
            raise ValueError(
                f"Dataset {dataset.metadata.name!r} has only "
                f"{class_indices.size} available graphs from "
                f"class {class_label} for bag size {bag_size}."
            )

        class_seed = _derived_seed(
            seed,
            class_label,
        )

        rng = np.random.default_rng(
            class_seed
        )

        shuffled = np.asarray(
            rng.permutation(
                class_indices
            ),
            dtype=np.int64,
        )

        used_count = (
            bag_count
            *
            bag_size
        )

        grouped = shuffled[
            :used_count
        ].reshape(
            bag_count,
            bag_size,
        )

        for group in grouped:
            bag_graph_indices.append(
                tuple(
                    int(
                        index
                    )
                    for index in group
                )
            )

            bag_labels.append(
                class_label
            )

    labels = np.asarray(
        bag_labels,
        dtype=np.int64,
    )

    labels.setflags(
        write=False
    )

    return _BagPartition(
        graph_indices=tuple(
            bag_graph_indices
        ),
        labels=labels,
    )


def _bags_for_indices(
    dataset: RealWorldDataset,
    graph_indices: IndexArray,
    *,
    bag_size: int,
    config: RealWorldDataConfig,
    outer_fold: int,
    partition: str,
) -> _BagPartition:
    return _make_bag_partition(
        dataset,
        graph_indices,
        bag_size=bag_size,
        seed=_partition_seed(
            config,
            dataset,
            outer_fold=outer_fold,
            partition=partition,
            indices=graph_indices,
        ),
    )


def _aggregate_matrix_bags(
    graph_features: FloatMatrix,
    bags: _BagPartition,
) -> FloatMatrix:
    rows = tuple(
        np.mean(
            graph_features[
                np.asarray(
                    graph_indices,
                    dtype=np.int64,
                )
            ],
            axis=0,
            dtype=np.float64,
        )
        for graph_indices in bags.graph_indices
    )

    result = np.asarray(
        rows,
        dtype=np.float64,
    )

    result.setflags(
        write=False
    )

    return result


def _graph_statistics_feature_matrix(
    dataset: RealWorldDataset,
) -> FloatMatrix:
    rows: list[
        NDArray[np.float64]
    ] = []

    graph_count = (
        dataset.metadata.graph_count
    )

    report_interval = max(
        1,
        graph_count
        //
        20,
    )

    for graph_index in range(
        graph_count
    ):
        rows.append(
            compute_graph_statistics(
                int(
                    dataset.vertex_counts[
                        graph_index
                    ]
                ),
                dataset.edges[
                    graph_index
                ],
            ).values
        )

        processed = (
            graph_index
            +
            1
        )

        if (
            processed == graph_count
            or processed
            %
            report_interval
            ==
            0
        ):
            _report(
                f"[Real world] {dataset.metadata.name}, "
                "graph statistics: "
                f"{processed}/{graph_count} graphs"
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


def _aggregate_graph_statistics_bags(
    graph_features: FloatMatrix,
    bags: _BagPartition,
) -> FloatMatrix:
    rows = tuple(
        aggregate_graph_statistics(
            graph_features[
                np.asarray(
                    graph_indices,
                    dtype=np.int64,
                )
            ]
        ).values
        for graph_indices in bags.graph_indices
    )

    result = np.asarray(
        rows,
        dtype=np.float64,
    )

    result.setflags(
        write=False
    )

    return result


def _graph_diagrams(
    dataset: RealWorldDataset,
    indices: IndexArray,
) -> tuple[
    DiagramArray,
    ...,
]:
    return tuple(
        dataset.persistence_diagrams[
            int(
                index
            )
        ]
        for index in indices
    )


def _has_nonempty_diagram(
    diagrams: Sequence[
        DiagramArray
    ],
) -> bool:
    return any(
        diagram.shape[
            0
        ]
        !=
        0
        for diagram in diagrams
    )


def _landscape_bag_features(
    fitted: FittedPersistenceLandscape | None,
    dataset: RealWorldDataset,
    bags: _BagPartition,
    *,
    num_landscapes: int,
    resolution: int,
) -> FloatMatrix:
    feature_count = (
        num_landscapes
        *
        resolution
    )

    if fitted is None:
        result = np.zeros(
            (
                len(
                    bags.graph_indices
                ),
                feature_count,
            ),
            dtype=np.float64,
        )

        result.setflags(
            write=False
        )

        return result

    rows: list[
        NDArray[np.float64]
    ] = []

    for graph_indices in bags.graph_indices:
        diagrams = tuple(
            dataset.persistence_diagrams[
                graph_index
            ]
            for graph_index in graph_indices
        )

        graph_features = (
            transform_persistence_landscapes(
                fitted,
                diagrams,
            )
        )

        rows.append(
            aggregate_persistence_landscapes(
                graph_features,
                num_landscapes=num_landscapes,
                resolution=resolution,
            ).values
        )

    result = np.asarray(
        rows,
        dtype=np.float64,
    )

    result.setflags(
        write=False
    )

    return result


def _image_bag_features(
    fitted: FittedPersistenceImage | None,
    dataset: RealWorldDataset,
    bags: _BagPartition,
    *,
    resolution: int,
    bandwidth: float,
) -> FloatMatrix:
    feature_count = (
        resolution
        *
        resolution
    )

    if fitted is None:
        result = np.zeros(
            (
                len(
                    bags.graph_indices
                ),
                feature_count,
            ),
            dtype=np.float64,
        )

        result.setflags(
            write=False
        )

        return result

    rows: list[
        NDArray[np.float64]
    ] = []

    for graph_indices in bags.graph_indices:
        diagrams = tuple(
            dataset.persistence_diagrams[
                graph_index
            ]
            for graph_index in graph_indices
        )

        graph_features = (
            transform_persistence_images(
                fitted,
                diagrams,
            )
        )

        rows.append(
            aggregate_persistence_images(
                graph_features,
                bandwidth=bandwidth,
                resolution_x=resolution,
                resolution_y=resolution,
            ).values
        )

    result = np.asarray(
        rows,
        dtype=np.float64,
    )

    result.setflags(
        write=False
    )

    return result


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


def _fixed_candidate_score(
    candidate: _SvmCandidate,
    train_indices: IndexArray,
    validation_indices: IndexArray,
    *,
    dataset: RealWorldDataset,
    graph_features: FloatMatrix,
    bag_size: int,
    config: RealWorldDataConfig,
    outer_fold: int,
    graph_statistics: bool,
) -> float:
    train_bags = _bags_for_indices(
        dataset,
        train_indices,
        bag_size=bag_size,
        config=config,
        outer_fold=outer_fold,
        partition="inner_train",
    )

    validation_bags = _bags_for_indices(
        dataset,
        validation_indices,
        bag_size=bag_size,
        config=config,
        outer_fold=outer_fold,
        partition="inner_validation",
    )

    aggregate = (
        _aggregate_graph_statistics_bags
        if graph_statistics
        else _aggregate_matrix_bags
    )

    train_features = aggregate(
        graph_features,
        train_bags,
    )

    validation_features = aggregate(
        graph_features,
        validation_bags,
    )

    predictions = _fit_predict_svm(
        train_features,
        train_bags.labels,
        validation_features,
        c=candidate.c,
        preprocessing=config.preprocessing,
        classifier=config.classifier,
    )

    return accuracy(
        validation_bags.labels,
        predictions,
    )


def _select_fixed_candidate(
    dataset: RealWorldDataset,
    graph_features: FloatMatrix,
    outer_fold: OuterFold,
    *,
    bag_size: int,
    config: RealWorldDataConfig,
    graph_statistics: bool,
) -> _SvmCandidate:
    candidates = tuple(
        _SvmCandidate(
            c=float(
                c
            )
        )
        for c in config.classifier.c_values
    )

    selection = select_candidate(
        candidates,
        outer_fold,
        lambda candidate, train_indices, validation_indices: (
            _fixed_candidate_score(
                candidate,
                train_indices,
                validation_indices,
                dataset=dataset,
                graph_features=graph_features,
                bag_size=bag_size,
                config=config,
                outer_fold=outer_fold.fold_index,
                graph_statistics=graph_statistics,
            )
        ),
    )

    return selection.selected_candidate


def _landscape_inner_feature_cache(
    dataset: RealWorldDataset,
    outer_fold: OuterFold,
    *,
    bag_size: int,
    config: RealWorldDataConfig,
) -> Mapping[
    tuple[
        IndexKey,
        IndexKey,
        _LandscapeRepresentation,
    ],
    tuple[
        FloatMatrix,
        LabelArray,
        FloatMatrix,
        LabelArray,
    ],
]:
    representations = _landscape_representations(
        config.benchmark
    )

    cache: dict[
        tuple[
            IndexKey,
            IndexKey,
            _LandscapeRepresentation,
        ],
        tuple[
            FloatMatrix,
            LabelArray,
            FloatMatrix,
            LabelArray,
        ],
    ] = {}

    for inner_position, inner_fold in enumerate(
        outer_fold.inner_folds,
        start=1,
    ):
        train_bags = _bags_for_indices(
            dataset,
            inner_fold.train_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="inner_train",
        )

        validation_bags = _bags_for_indices(
            dataset,
            inner_fold.validation_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="inner_validation",
        )

        training_diagrams = _graph_diagrams(
            dataset,
            inner_fold.train_indices,
        )

        for representation_position, representation in enumerate(
            representations,
            start=1,
        ):
            _report(
                f"[Real world] {dataset.metadata.name}, "
                f"bag size {bag_size}, persistence landscape: "
                f"outer {outer_fold.fold_index + 1}, "
                f"inner {inner_position}/"
                f"{len(outer_fold.inner_folds)}, "
                f"representation {representation_position}/"
                f"{len(representations)}"
            )

            fitted: FittedPersistenceLandscape | None

            if _has_nonempty_diagram(
                training_diagrams
            ):
                fitted = fit_persistence_landscape(
                    training_diagrams,
                    PersistenceLandscapeConfig(
                        num_landscapes=(
                            representation.num_landscapes
                        ),
                        resolution=(
                            representation.resolution
                        ),
                    ),
                )
            else:
                fitted = None

            cache[
                (
                    _index_key(
                        inner_fold.train_indices
                    ),
                    _index_key(
                        inner_fold.validation_indices
                    ),
                    representation,
                )
            ] = (
                _landscape_bag_features(
                    fitted,
                    dataset,
                    train_bags,
                    num_landscapes=(
                        representation.num_landscapes
                    ),
                    resolution=(
                        representation.resolution
                    ),
                ),
                train_bags.labels,
                _landscape_bag_features(
                    fitted,
                    dataset,
                    validation_bags,
                    num_landscapes=(
                        representation.num_landscapes
                    ),
                    resolution=(
                        representation.resolution
                    ),
                ),
                validation_bags.labels,
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
            LabelArray,
            FloatMatrix,
            LabelArray,
        ],
    ],
    config: RealWorldDataConfig,
) -> float:
    representation = _LandscapeRepresentation(
        num_landscapes=candidate.num_landscapes,
        resolution=candidate.resolution,
    )

    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = cache[
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
        train_labels,
        validation_features,
        c=candidate.c,
        preprocessing=config.preprocessing,
        classifier=config.classifier,
    )

    return accuracy(
        validation_labels,
        predictions,
    )


def _select_landscape_candidate(
    dataset: RealWorldDataset,
    outer_fold: OuterFold,
    *,
    bag_size: int,
    config: RealWorldDataConfig,
) -> _LandscapeCandidate:
    cache = _landscape_inner_feature_cache(
        dataset,
        outer_fold,
        bag_size=bag_size,
        config=config,
    )

    selection = select_candidate(
        _landscape_candidates(
            config.benchmark,
            config.classifier,
        ),
        outer_fold,
        lambda candidate, train_indices, validation_indices: (
            _landscape_candidate_score(
                candidate,
                train_indices,
                validation_indices,
                cache=cache,
                config=config,
            )
        ),
    )

    return selection.selected_candidate


def _persistence_image_inner_feature_cache(
    dataset: RealWorldDataset,
    outer_fold: OuterFold,
    *,
    bag_size: int,
    config: RealWorldDataConfig,
) -> Mapping[
    tuple[
        IndexKey,
        IndexKey,
        _PersistenceImageRepresentation,
    ],
    tuple[
        FloatMatrix,
        LabelArray,
        FloatMatrix,
        LabelArray,
    ],
]:
    representations = _persistence_image_representations(
        config.benchmark
    )

    cache: dict[
        tuple[
            IndexKey,
            IndexKey,
            _PersistenceImageRepresentation,
        ],
        tuple[
            FloatMatrix,
            LabelArray,
            FloatMatrix,
            LabelArray,
        ],
    ] = {}

    for inner_position, inner_fold in enumerate(
        outer_fold.inner_folds,
        start=1,
    ):
        train_bags = _bags_for_indices(
            dataset,
            inner_fold.train_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="inner_train",
        )

        validation_bags = _bags_for_indices(
            dataset,
            inner_fold.validation_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="inner_validation",
        )

        training_diagrams = _graph_diagrams(
            dataset,
            inner_fold.train_indices,
        )

        for representation_position, representation in enumerate(
            representations,
            start=1,
        ):
            _report(
                f"[Real world] {dataset.metadata.name}, "
                f"bag size {bag_size}, persistence image: "
                f"outer {outer_fold.fold_index + 1}, "
                f"inner {inner_position}/"
                f"{len(outer_fold.inner_folds)}, "
                f"representation {representation_position}/"
                f"{len(representations)}"
            )

            fitted: FittedPersistenceImage | None

            if _has_nonempty_diagram(
                training_diagrams
            ):
                fitted = fit_persistence_image(
                    training_diagrams,
                    PersistenceImageConfig(
                        bandwidth=(
                            representation.bandwidth
                        ),
                        resolution_x=(
                            representation.resolution
                        ),
                        resolution_y=(
                            representation.resolution
                        ),
                    ),
                )
            else:
                fitted = None

            cache[
                (
                    _index_key(
                        inner_fold.train_indices
                    ),
                    _index_key(
                        inner_fold.validation_indices
                    ),
                    representation,
                )
            ] = (
                _image_bag_features(
                    fitted,
                    dataset,
                    train_bags,
                    resolution=(
                        representation.resolution
                    ),
                    bandwidth=(
                        representation.bandwidth
                    ),
                ),
                train_bags.labels,
                _image_bag_features(
                    fitted,
                    dataset,
                    validation_bags,
                    resolution=(
                        representation.resolution
                    ),
                    bandwidth=(
                        representation.bandwidth
                    ),
                ),
                validation_bags.labels,
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
            LabelArray,
            FloatMatrix,
            LabelArray,
        ],
    ],
    config: RealWorldDataConfig,
) -> float:
    representation = _PersistenceImageRepresentation(
        resolution=candidate.resolution,
        bandwidth=candidate.bandwidth,
    )

    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = cache[
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
        train_labels,
        validation_features,
        c=candidate.c,
        preprocessing=config.preprocessing,
        classifier=config.classifier,
    )

    return accuracy(
        validation_labels,
        predictions,
    )


def _select_persistence_image_candidate(
    dataset: RealWorldDataset,
    outer_fold: OuterFold,
    *,
    bag_size: int,
    config: RealWorldDataConfig,
) -> _PersistenceImageCandidate:
    cache = _persistence_image_inner_feature_cache(
        dataset,
        outer_fold,
        bag_size=bag_size,
        config=config,
    )

    selection = select_candidate(
        _persistence_image_candidates(
            config.benchmark,
            config.classifier,
        ),
        outer_fold,
        lambda candidate, train_indices, validation_indices: (
            _persistence_image_candidate_score(
                candidate,
                train_indices,
                validation_indices,
                cache=cache,
                config=config,
            )
        ),
    )

    return selection.selected_candidate


def _finalize_bag_size_result(
    dataset: RealWorldDataset,
    method: str,
    bag_size: int,
    outer_results: Sequence[
        OuterFoldResult
    ],
    bag_predictions: Sequence[
        BagPrediction
    ],
    *,
    config: RealWorldDataConfig,
) -> MethodBagSizeResult:
    true_array = np.asarray(
        tuple(
            prediction.label
            for prediction in bag_predictions
        ),
        dtype=np.int64,
    )

    prediction_array = np.asarray(
        tuple(
            prediction.prediction
            for prediction in bag_predictions
        ),
        dtype=np.int64,
    )

    fold_id_array = np.asarray(
        tuple(
            prediction.outer_fold
            for prediction in bag_predictions
        ),
        dtype=np.int64,
    )

    evaluation = evaluate_out_of_fold_predictions(
        true_array,
        prediction_array,
        fold_id_array,
    )

    bootstrap = stratified_bootstrap(
        true_array,
        prediction_array,
        resamples=config.bootstrap.resample_count,
        confidence_level=(
            config.bootstrap.confidence_level
        ),
        seed=_derived_seed(
            config.bootstrap.seed,
            dataset.metadata.name,
            bag_size,
        ),
    )

    return MethodBagSizeResult(
        dataset=dataset.metadata.name,
        method=method,
        bag_size=bag_size,
        outer_folds=tuple(
            outer_results
        ),
        predictions=tuple(
            bag_predictions
        ),
        evaluation=evaluation,
        bootstrap=bootstrap,
    )


def _append_predictions(
    destination: list[
        BagPrediction
    ],
    *,
    dataset: RealWorldDataset,
    method: str,
    bag_size: int,
    outer_fold: int,
    bags: _BagPartition,
    predictions: PredictionArray,
) -> None:
    for bag_index, (
        graph_indices,
        label,
        prediction,
    ) in enumerate(
        zip(
            bags.graph_indices,
            bags.labels,
            predictions,
            strict=True,
        )
    ):
        destination.append(
            BagPrediction(
                dataset=dataset.metadata.name,
                method=method,
                bag_size=bag_size,
                outer_fold=outer_fold,
                bag_index=bag_index,
                graph_indices=graph_indices,
                label=int(
                    label
                ),
                prediction=int(
                    prediction
                ),
            )
        )


def _run_fixed_method_bag_size(
    dataset: RealWorldDataset,
    graph_features: FloatMatrix,
    plan: NestedCrossValidationPlan,
    *,
    method: str,
    bag_size: int,
    config: RealWorldDataConfig,
    graph_statistics: bool,
) -> MethodBagSizeResult:
    outer_results: list[
        OuterFoldResult
    ] = []

    bag_predictions: list[
        BagPrediction
    ] = []

    aggregate = (
        _aggregate_graph_statistics_bags
        if graph_statistics
        else _aggregate_matrix_bags
    )

    for outer_position, outer_fold in enumerate(
        plan.folds,
        start=1,
    ):
        _report(
            f"[Real world] {dataset.metadata.name}, {method}, "
            f"bag size {bag_size}: outer "
            f"{outer_position}/{len(plan.folds)}"
        )

        selected = _select_fixed_candidate(
            dataset,
            graph_features,
            outer_fold,
            bag_size=bag_size,
            config=config,
            graph_statistics=graph_statistics,
        )

        train_bags = _bags_for_indices(
            dataset,
            outer_fold.train_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="outer_train",
        )

        test_bags = _bags_for_indices(
            dataset,
            outer_fold.test_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="outer_test",
        )

        train_features = aggregate(
            graph_features,
            train_bags,
        )

        test_features = aggregate(
            graph_features,
            test_bags,
        )

        fold_predictions = _fit_predict_svm(
            train_features,
            train_bags.labels,
            test_features,
            c=selected.c,
            preprocessing=config.preprocessing,
            classifier=config.classifier,
        )

        fold_metrics = classification_metrics(
            test_bags.labels,
            fold_predictions,
        )

        outer_results.append(
            OuterFoldResult(
                dataset=dataset.metadata.name,
                method=method,
                bag_size=bag_size,
                outer_fold=outer_fold.fold_index,
                train_graph_count=int(
                    outer_fold.train_indices.size
                ),
                test_graph_count=int(
                    outer_fold.test_indices.size
                ),
                train_bag_count=int(
                    train_bags.labels.size
                ),
                test_bag_count=int(
                    test_bags.labels.size
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

        _append_predictions(
            bag_predictions,
            dataset=dataset,
            method=method,
            bag_size=bag_size,
            outer_fold=outer_fold.fold_index,
            bags=test_bags,
            predictions=fold_predictions,
        )

    return _finalize_bag_size_result(
        dataset,
        method,
        bag_size,
        outer_results,
        bag_predictions,
        config=config,
    )


def _run_landscape_bag_size(
    dataset: RealWorldDataset,
    plan: NestedCrossValidationPlan,
    *,
    bag_size: int,
    config: RealWorldDataConfig,
) -> MethodBagSizeResult:
    outer_results: list[
        OuterFoldResult
    ] = []

    bag_predictions: list[
        BagPrediction
    ] = []

    for outer_position, outer_fold in enumerate(
        plan.folds,
        start=1,
    ):
        _report(
            f"[Real world] {dataset.metadata.name}, "
            f"{PERSISTENCE_LANDSCAPE_METHOD}, bag size "
            f"{bag_size}: outer {outer_position}/{len(plan.folds)}"
        )

        selected = _select_landscape_candidate(
            dataset,
            outer_fold,
            bag_size=bag_size,
            config=config,
        )

        train_bags = _bags_for_indices(
            dataset,
            outer_fold.train_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="outer_train",
        )

        test_bags = _bags_for_indices(
            dataset,
            outer_fold.test_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="outer_test",
        )

        training_diagrams = _graph_diagrams(
            dataset,
            outer_fold.train_indices,
        )

        fitted: FittedPersistenceLandscape | None

        if _has_nonempty_diagram(
            training_diagrams
        ):
            fitted = fit_persistence_landscape(
                training_diagrams,
                PersistenceLandscapeConfig(
                    num_landscapes=(
                        selected.num_landscapes
                    ),
                    resolution=(
                        selected.resolution
                    ),
                ),
            )
        else:
            fitted = None

        train_features = _landscape_bag_features(
            fitted,
            dataset,
            train_bags,
            num_landscapes=(
                selected.num_landscapes
            ),
            resolution=selected.resolution,
        )

        test_features = _landscape_bag_features(
            fitted,
            dataset,
            test_bags,
            num_landscapes=(
                selected.num_landscapes
            ),
            resolution=selected.resolution,
        )

        fold_predictions = _fit_predict_svm(
            train_features,
            train_bags.labels,
            test_features,
            c=selected.c,
            preprocessing=config.preprocessing,
            classifier=config.classifier,
        )

        fold_metrics = classification_metrics(
            test_bags.labels,
            fold_predictions,
        )

        outer_results.append(
            OuterFoldResult(
                dataset=dataset.metadata.name,
                method=PERSISTENCE_LANDSCAPE_METHOD,
                bag_size=bag_size,
                outer_fold=outer_fold.fold_index,
                train_graph_count=int(
                    outer_fold.train_indices.size
                ),
                test_graph_count=int(
                    outer_fold.test_indices.size
                ),
                train_bag_count=int(
                    train_bags.labels.size
                ),
                test_bag_count=int(
                    test_bags.labels.size
                ),
                selected_c=selected.c,
                selected_num_landscapes=(
                    selected.num_landscapes
                ),
                selected_landscape_resolution=(
                    selected.resolution
                ),
                selected_image_resolution=None,
                selected_image_bandwidth=None,
                accuracy=fold_metrics.accuracy,
                macro_f1=fold_metrics.macro_f1,
            )
        )

        _append_predictions(
            bag_predictions,
            dataset=dataset,
            method=PERSISTENCE_LANDSCAPE_METHOD,
            bag_size=bag_size,
            outer_fold=outer_fold.fold_index,
            bags=test_bags,
            predictions=fold_predictions,
        )

    return _finalize_bag_size_result(
        dataset,
        PERSISTENCE_LANDSCAPE_METHOD,
        bag_size,
        outer_results,
        bag_predictions,
        config=config,
    )


def _run_persistence_image_bag_size(
    dataset: RealWorldDataset,
    plan: NestedCrossValidationPlan,
    *,
    bag_size: int,
    config: RealWorldDataConfig,
) -> MethodBagSizeResult:
    outer_results: list[
        OuterFoldResult
    ] = []

    bag_predictions: list[
        BagPrediction
    ] = []

    for outer_position, outer_fold in enumerate(
        plan.folds,
        start=1,
    ):
        _report(
            f"[Real world] {dataset.metadata.name}, "
            f"{PERSISTENCE_IMAGE_METHOD}, bag size {bag_size}: "
            f"outer {outer_position}/{len(plan.folds)}"
        )

        selected = _select_persistence_image_candidate(
            dataset,
            outer_fold,
            bag_size=bag_size,
            config=config,
        )

        train_bags = _bags_for_indices(
            dataset,
            outer_fold.train_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="outer_train",
        )

        test_bags = _bags_for_indices(
            dataset,
            outer_fold.test_indices,
            bag_size=bag_size,
            config=config,
            outer_fold=outer_fold.fold_index,
            partition="outer_test",
        )

        training_diagrams = _graph_diagrams(
            dataset,
            outer_fold.train_indices,
        )

        fitted: FittedPersistenceImage | None

        if _has_nonempty_diagram(
            training_diagrams
        ):
            fitted = fit_persistence_image(
                training_diagrams,
                PersistenceImageConfig(
                    bandwidth=selected.bandwidth,
                    resolution_x=selected.resolution,
                    resolution_y=selected.resolution,
                ),
            )
        else:
            fitted = None

        train_features = _image_bag_features(
            fitted,
            dataset,
            train_bags,
            resolution=selected.resolution,
            bandwidth=selected.bandwidth,
        )

        test_features = _image_bag_features(
            fitted,
            dataset,
            test_bags,
            resolution=selected.resolution,
            bandwidth=selected.bandwidth,
        )

        fold_predictions = _fit_predict_svm(
            train_features,
            train_bags.labels,
            test_features,
            c=selected.c,
            preprocessing=config.preprocessing,
            classifier=config.classifier,
        )

        fold_metrics = classification_metrics(
            test_bags.labels,
            fold_predictions,
        )

        outer_results.append(
            OuterFoldResult(
                dataset=dataset.metadata.name,
                method=PERSISTENCE_IMAGE_METHOD,
                bag_size=bag_size,
                outer_fold=outer_fold.fold_index,
                train_graph_count=int(
                    outer_fold.train_indices.size
                ),
                test_graph_count=int(
                    outer_fold.test_indices.size
                ),
                train_bag_count=int(
                    train_bags.labels.size
                ),
                test_bag_count=int(
                    test_bags.labels.size
                ),
                selected_c=selected.c,
                selected_num_landscapes=None,
                selected_landscape_resolution=None,
                selected_image_resolution=(
                    selected.resolution
                ),
                selected_image_bandwidth=(
                    selected.bandwidth
                ),
                accuracy=fold_metrics.accuracy,
                macro_f1=fold_metrics.macro_f1,
            )
        )

        _append_predictions(
            bag_predictions,
            dataset=dataset,
            method=PERSISTENCE_IMAGE_METHOD,
            bag_size=bag_size,
            outer_fold=outer_fold.fold_index,
            bags=test_bags,
            predictions=fold_predictions,
        )

    return _finalize_bag_size_result(
        dataset,
        PERSISTENCE_IMAGE_METHOD,
        bag_size,
        outer_results,
        bag_predictions,
        config=config,
    )


def _validate_bag_feasibility(
    dataset: RealWorldDataset,
    config: RealWorldDataConfig,
) -> None:
    _, counts = np.unique(
        dataset.labels,
        return_counts=True,
    )

    minimum_class_count = int(
        np.min(
            counts
        )
    )

    maximum_bag_size = max(
        config.mil.bag_sizes
    )

    outer_folds = (
        config.cross_validation.outer_fold_count
    )

    minimum_outer_test_count = (
        minimum_class_count
        //
        outer_folds
    )

    if minimum_outer_test_count < maximum_bag_size:
        raise ValueError(
            f"Dataset {dataset.metadata.name!r} cannot support "
            f"disjoint homogeneous bags of size "
            f"{maximum_bag_size} in every outer test fold with "
            f"{outer_folds} outer folds. The smallest class has "
            f"{minimum_class_count} graphs."
        )


def _run_dataset(
    dataset: RealWorldDataset,
    config: RealWorldDataConfig,
) -> DatasetResult:
    _validate_bag_feasibility(
        dataset,
        config,
    )

    _report(
        f"[Real world] {dataset.metadata.name}: "
        "creating graph-level nested cross-validation plan"
    )

    plan = make_nested_stratified_plan(
        dataset.labels,
        stratification_domain=dataset.labels,
        outer_folds=(
            config.cross_validation.outer_fold_count
        ),
        inner_folds=(
            config.cross_validation.inner_fold_count
        ),
        outer_seed=_derived_seed(
            config.cross_validation.outer_seed,
            dataset.metadata.name,
        ),
        inner_seed=_derived_seed(
            config.cross_validation.inner_seed,
            dataset.metadata.name,
        ),
    )

    _report(
        f"[Real world] {dataset.metadata.name}: "
        "computing graph-statistics features"
    )

    graph_statistics_features = (
        _graph_statistics_feature_matrix(
            dataset
        )
    )

    method_results: dict[
        str,
        Mapping[
            int,
            MethodBagSizeResult,
        ],
    ] = {}

    graph_statistics_results: dict[
        int,
        MethodBagSizeResult,
    ] = {}

    landscape_results: dict[
        int,
        MethodBagSizeResult,
    ] = {}

    image_results: dict[
        int,
        MethodBagSizeResult,
    ] = {}

    harmonic_results: dict[
        int,
        MethodBagSizeResult,
    ] = {}

    for bag_size in config.mil.bag_sizes:
        graph_statistics_results[
            bag_size
        ] = _run_fixed_method_bag_size(
            dataset,
            graph_statistics_features,
            plan,
            method=GRAPH_STATISTICS_METHOD,
            bag_size=bag_size,
            config=config,
            graph_statistics=True,
        )

        landscape_results[
            bag_size
        ] = _run_landscape_bag_size(
            dataset,
            plan,
            bag_size=bag_size,
            config=config,
        )

        image_results[
            bag_size
        ] = _run_persistence_image_bag_size(
            dataset,
            plan,
            bag_size=bag_size,
            config=config,
        )

        harmonic_results[
            bag_size
        ] = _run_fixed_method_bag_size(
            dataset,
            dataset.harmonic_features,
            plan,
            method=HARMONIC_AGGREGATION_METHOD,
            bag_size=bag_size,
            config=config,
            graph_statistics=False,
        )

    method_results[
        GRAPH_STATISTICS_METHOD
    ] = MappingProxyType(
        graph_statistics_results
    )

    method_results[
        PERSISTENCE_LANDSCAPE_METHOD
    ] = MappingProxyType(
        landscape_results
    )

    method_results[
        PERSISTENCE_IMAGE_METHOD
    ] = MappingProxyType(
        image_results
    )

    method_results[
        HARMONIC_AGGREGATION_METHOD
    ] = MappingProxyType(
        harmonic_results
    )

    return DatasetResult(
        dataset=dataset.metadata.name,
        cross_validation_plan=plan,
        methods=MappingProxyType(
            {
                method: method_results[
                    method
                ]
                for method in BENCHMARK_METHODS
            }
        ),
    )


def run_real_world_experiment(
    data: RealWorldData,
    config: RealWorldDataConfig,
) -> RealWorldExperimentResult:
    dataset_results: dict[
        str,
        DatasetResult,
    ] = {}

    for dataset_name, dataset in data.datasets.items():
        _report(
            f"[Real world] Starting {dataset_name}"
        )

        dataset_results[
            dataset_name
        ] = _run_dataset(
            dataset,
            config,
        )

        _report(
            f"[Real world] Completed {dataset_name}"
        )

    return RealWorldExperimentResult(
        datasets=MappingProxyType(
            dataset_results
        ),
    )


__all__ = [
    "BagPrediction",
    "DatasetResult",
    "MethodBagSizeResult",
    "OuterFoldResult",
    "RealWorldExperimentResult",
    "run_real_world_experiment",
]