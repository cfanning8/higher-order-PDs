from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final


RANDOM_GRAPH_CLASSIFICATION_CONFIG_SCHEMA_VERSION: Final[int] = 3

CPP_METADATA_SCHEMA_VERSION: Final[int] = 2
CPP_OUTPUT_SCHEMA_VERSION: Final[int] = 3

BENCHMARK_EXPERIMENT: Final[str] = "benchmark"
HARMONIC_AGGREGATION_ABLATION_EXPERIMENT: Final[str] = (
    "harmonic_aggregation_ablation"
)
BATCH_SIZE_SENSITIVITY_EXPERIMENT: Final[str] = (
    "batch_size_sensitivity"
)
RFF_SENSITIVITY_EXPERIMENT: Final[str] = "rff_sensitivity"
ROBUSTNESS_EXPERIMENT: Final[str] = "robustness"
SCALABILITY_EXPERIMENT: Final[str] = "scalability"

SUPPORTED_EXPERIMENTS: Final[tuple[str, ...]] = (
    BENCHMARK_EXPERIMENT,
    HARMONIC_AGGREGATION_ABLATION_EXPERIMENT,
    BATCH_SIZE_SENSITIVITY_EXPERIMENT,
    RFF_SENSITIVITY_EXPERIMENT,
    ROBUSTNESS_EXPERIMENT,
    SCALABILITY_EXPERIMENT,
)

CROSS_OBSERVATION_ABLATION: Final[str] = "cross_observation"
LINEAR_ABLATION: Final[str] = "linear"
NO_PREORDER_ABLATION: Final[str] = "no_preorder"

HARMONIC_AGGREGATION_ABLATION_KINDS: Final[tuple[str, ...]] = (
    CROSS_OBSERVATION_ABLATION,
    LINEAR_ABLATION,
    NO_PREORDER_ABLATION,
)

EDGE_DELETION_PERTURBATION: Final[str] = "edge_deletion"
EDGE_INSERTION_PERTURBATION: Final[str] = "edge_insertion"
DEGREE_PRESERVING_REWIRING_PERTURBATION: Final[str] = (
    "degree_preserving_rewiring"
)
FILTRATION_NOISE_PERTURBATION: Final[str] = "filtration_noise"

ROBUSTNESS_PERTURBATIONS: Final[tuple[str, ...]] = (
    EDGE_DELETION_PERTURBATION,
    EDGE_INSERTION_PERTURBATION,
    DEGREE_PRESERVING_REWIRING_PERTURBATION,
    FILTRATION_NOISE_PERTURBATION,
)

GRAPH_STATISTICS_METHOD: Final[str] = "graph_statistics"
PERSISTENCE_LANDSCAPE_METHOD: Final[str] = "persistence_landscape"
PERSISTENCE_IMAGE_METHOD: Final[str] = "persistence_image"
PERSLAY_METHOD: Final[str] = "perslay"
HARMONIC_AGGREGATION_METHOD: Final[str] = "harmonic_aggregation"

BENCHMARK_METHODS: Final[tuple[str, ...]] = (
    GRAPH_STATISTICS_METHOD,
    PERSISTENCE_LANDSCAPE_METHOD,
    PERSISTENCE_IMAGE_METHOD,
    PERSLAY_METHOD,
    HARMONIC_AGGREGATION_METHOD,
)

INNER_SELECTION_METRIC: Final[str] = "accuracy"
PRIMARY_CLASSIFICATION_METRIC: Final[str] = "accuracy"
SECONDARY_CLASSIFICATION_METRIC: Final[str] = "macro_f1"
CLASSIFICATION_ESTIMAND: Final[str] = "pooled_out_of_fold"

BOOTSTRAP_TARGET: Final[str] = (
    "conditional_fitted_nested_cv_pipeline_performance"
)
BOOTSTRAP_INTERVAL_METHOD: Final[str] = "percentile"

SVM_KERNEL: Final[str] = "rbf"
SVM_GAMMA: Final[str] = "scale"
SVM_CLASS_WEIGHT: Final[None] = None
SVM_PROBABILITY: Final[bool] = False

_UINT64_MAX: Final[int] = (1 << 64) - 1


@dataclass(
    frozen=True,
    slots=True,
)
class ExpectedRobustnessCondition:
    perturbation: str
    severity: float


EXPECTED_ROBUSTNESS_CONDITIONS: Final[
    tuple[ExpectedRobustnessCondition, ...]
] = (
    ExpectedRobustnessCondition(
        EDGE_DELETION_PERTURBATION,
        0.02,
    ),
    ExpectedRobustnessCondition(
        EDGE_DELETION_PERTURBATION,
        0.05,
    ),
    ExpectedRobustnessCondition(
        EDGE_DELETION_PERTURBATION,
        0.10,
    ),
    ExpectedRobustnessCondition(
        EDGE_DELETION_PERTURBATION,
        0.20,
    ),
    ExpectedRobustnessCondition(
        EDGE_INSERTION_PERTURBATION,
        0.02,
    ),
    ExpectedRobustnessCondition(
        EDGE_INSERTION_PERTURBATION,
        0.05,
    ),
    ExpectedRobustnessCondition(
        EDGE_INSERTION_PERTURBATION,
        0.10,
    ),
    ExpectedRobustnessCondition(
        EDGE_INSERTION_PERTURBATION,
        0.20,
    ),
    ExpectedRobustnessCondition(
        DEGREE_PRESERVING_REWIRING_PERTURBATION,
        0.02,
    ),
    ExpectedRobustnessCondition(
        DEGREE_PRESERVING_REWIRING_PERTURBATION,
        0.05,
    ),
    ExpectedRobustnessCondition(
        DEGREE_PRESERVING_REWIRING_PERTURBATION,
        0.10,
    ),
    ExpectedRobustnessCondition(
        DEGREE_PRESERVING_REWIRING_PERTURBATION,
        0.20,
    ),
    ExpectedRobustnessCondition(
        FILTRATION_NOISE_PERTURBATION,
        0.01,
    ),
    ExpectedRobustnessCondition(
        FILTRATION_NOISE_PERTURBATION,
        0.025,
    ),
    ExpectedRobustnessCondition(
        FILTRATION_NOISE_PERTURBATION,
        0.05,
    ),
    ExpectedRobustnessCondition(
        FILTRATION_NOISE_PERTURBATION,
        0.10,
    ),
)


@dataclass(
    frozen=True,
    slots=True,
)
class ExpectedCppDesign:
    metadata_schema_version: int
    output_schema_version: int

    benchmark_vertex_count: int
    benchmark_edge_count: int
    batches_per_model: int
    graphs_per_batch: int

    baseline_character_count: int
    aggregation_order: int
    highest_explicit_order: int

    rff_replicate_count: int
    rff_character_counts: tuple[int, ...]

    robustness_realization_count: int
    robustness_conditions: tuple[
        ExpectedRobustnessCondition,
        ...,
    ]

    cross_observation_ablation_realization_count: int

    batch_size_graph_counts: tuple[int, ...]

    scalability_graph_counts: tuple[int, ...]
    scalability_input_support_sizes: tuple[int, ...]
    scalability_character_counts: tuple[int, ...]
    scalability_repeat_count: int

    canonical_root_seed: int


EXPECTED_CPP_DESIGN: Final[ExpectedCppDesign] = ExpectedCppDesign(
    metadata_schema_version=CPP_METADATA_SCHEMA_VERSION,
    output_schema_version=CPP_OUTPUT_SCHEMA_VERSION,
    benchmark_vertex_count=100,
    benchmark_edge_count=200,
    batches_per_model=100,
    graphs_per_batch=4,
    baseline_character_count=256,
    aggregation_order=2,
    highest_explicit_order=2,
    rff_replicate_count=10,
    rff_character_counts=(
        64,
        128,
        256,
        512,
        1024,
    ),
    robustness_realization_count=10,
    robustness_conditions=EXPECTED_ROBUSTNESS_CONDITIONS,
    cross_observation_ablation_realization_count=10,
    batch_size_graph_counts=(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ),
    scalability_graph_counts=(
        1,
        2,
        4,
        8,
        16,
        32,
    ),
    scalability_input_support_sizes=(
        16,
        32,
        64,
        128,
        256,
    ),
    scalability_character_counts=(
        64,
        128,
        256,
        512,
        1024,
    ),
    scalability_repeat_count=30,
    canonical_root_seed=20260601,
)


def _require_bool(
    value: bool,
    *,
    description: str,
) -> bool:
    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{description.capitalize()} must be boolean."
        )

    return value


def _require_integer(
    value: int,
    *,
    description: str,
) -> int:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"{description.capitalize()} must be an integer."
        )

    return value


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


def _require_fold_count(
    value: int,
    *,
    description: str,
) -> int:
    parsed = _require_integer(
        value,
        description=description,
    )

    if parsed < 2:
        raise ValueError(
            f"{description.capitalize()} must be at least 2."
        )

    return parsed


def _require_uint64(
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


def _require_finite_scalar(
    value: float,
    *,
    description: str,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (
                int,
                float,
            ),
        )
    ):
        raise TypeError(
            f"{description.capitalize()} must be numeric."
        )

    parsed = float(
        value
    )

    if not math.isfinite(
        parsed
    ):
        raise ValueError(
            f"{description.capitalize()} must be finite."
        )

    return parsed


def _require_positive_scalar(
    value: float,
    *,
    description: str,
) -> float:
    parsed = _require_finite_scalar(
        value,
        description=description,
    )

    if parsed <= 0.0:
        raise ValueError(
            f"{description.capitalize()} must be strictly positive."
        )

    return parsed


def _require_nonnegative_scalar(
    value: float,
    *,
    description: str,
) -> float:
    parsed = _require_finite_scalar(
        value,
        description=description,
    )

    if parsed < 0.0:
        raise ValueError(
            f"{description.capitalize()} must be nonnegative."
        )

    return parsed


def _require_probability(
    value: float,
    *,
    description: str,
) -> float:
    parsed = _require_finite_scalar(
        value,
        description=description,
    )

    if not 0.0 < parsed < 1.0:
        raise ValueError(
            f"{description.capitalize()} must lie in (0, 1)."
        )

    return parsed


def _normalize_positive_integer_tuple(
    values: tuple[int, ...] | list[int],
    *,
    description: str,
) -> tuple[int, ...]:
    normalized = tuple(
        values
    )

    if not normalized:
        raise ValueError(
            f"{description.capitalize()} must not be empty."
        )

    return tuple(
        _require_positive_integer(
            value,
            description=f"{description} entry",
        )
        for value in normalized
    )


def _normalize_positive_float_tuple(
    values: tuple[float, ...] | list[float],
    *,
    description: str,
) -> tuple[float, ...]:
    normalized = tuple(
        values
    )

    if not normalized:
        raise ValueError(
            f"{description.capitalize()} must not be empty."
        )

    return tuple(
        _require_positive_scalar(
            value,
            description=f"{description} entry",
        )
        for value in normalized
    )


@dataclass(
    frozen=True,
    slots=True,
)
class ExperimentSelection:
    benchmark: bool = True
    harmonic_aggregation_ablation: bool = True
    batch_size_sensitivity: bool = True
    rff_sensitivity: bool = True
    robustness: bool = True
    scalability: bool = True

    def __post_init__(
        self,
    ) -> None:
        _require_bool(
            self.benchmark,
            description="benchmark selection",
        )

        _require_bool(
            self.harmonic_aggregation_ablation,
            description="harmonic-aggregation ablation selection",
        )

        _require_bool(
            self.batch_size_sensitivity,
            description="batch-size sensitivity selection",
        )

        _require_bool(
            self.rff_sensitivity,
            description="RFF sensitivity selection",
        )

        _require_bool(
            self.robustness,
            description="robustness selection",
        )

        _require_bool(
            self.scalability,
            description="scalability selection",
        )

        if not any(
            (
                self.benchmark,
                self.harmonic_aggregation_ablation,
                self.batch_size_sensitivity,
                self.rff_sensitivity,
                self.robustness,
                self.scalability,
            )
        ):
            raise ValueError(
                "At least one random-graph-classification "
                "experiment must be enabled."
            )

    def enabled_experiments(
        self,
    ) -> tuple[str, ...]:
        enabled: list[str] = []

        if self.benchmark:
            enabled.append(
                BENCHMARK_EXPERIMENT
            )

        if self.harmonic_aggregation_ablation:
            enabled.append(
                HARMONIC_AGGREGATION_ABLATION_EXPERIMENT
            )

        if self.batch_size_sensitivity:
            enabled.append(
                BATCH_SIZE_SENSITIVITY_EXPERIMENT
            )

        if self.rff_sensitivity:
            enabled.append(
                RFF_SENSITIVITY_EXPERIMENT
            )

        if self.robustness:
            enabled.append(
                ROBUSTNESS_EXPERIMENT
            )

        if self.scalability:
            enabled.append(
                SCALABILITY_EXPERIMENT
            )

        return tuple(
            enabled
        )


@dataclass(
    frozen=True,
    slots=True,
)
class CrossValidationSettings:
    outer_fold_count: int = 10
    inner_fold_count: int = 5

    outer_seed: int = 6849471059393190731
    inner_seed: int = 12053211736635842437

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "outer_fold_count",
            _require_fold_count(
                self.outer_fold_count,
                description="outer fold count",
            ),
        )

        object.__setattr__(
            self,
            "inner_fold_count",
            _require_fold_count(
                self.inner_fold_count,
                description="inner fold count",
            ),
        )

        object.__setattr__(
            self,
            "outer_seed",
            _require_uint64(
                self.outer_seed,
                description="outer cross-validation seed",
            ),
        )

        object.__setattr__(
            self,
            "inner_seed",
            _require_uint64(
                self.inner_seed,
                description="inner cross-validation seed",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class PreprocessingSettings:
    variance_tolerance: float = 0.0

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "variance_tolerance",
            _require_nonnegative_scalar(
                self.variance_tolerance,
                description="feature variance tolerance",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class ClassifierSettings:
    c_values: tuple[float, ...] = (
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0,
        10.0,
        100.0,
        1.0e3,
        1.0e4,
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "c_values",
            _normalize_positive_float_tuple(
                self.c_values,
                description="SVM C values",
            ),
        )

    @property
    def kernel(
        self,
    ) -> str:
        return SVM_KERNEL

    @property
    def gamma(
        self,
    ) -> str:
        return SVM_GAMMA

    @property
    def class_weight(
        self,
    ) -> None:
        return SVM_CLASS_WEIGHT

    @property
    def probability(
        self,
    ) -> bool:
        return SVM_PROBABILITY

    @property
    def selection_metric(
        self,
    ) -> str:
        return INNER_SELECTION_METRIC


@dataclass(
    frozen=True,
    slots=True,
)
class BootstrapSettings:
    resample_count: int = 10_000
    confidence_level: float = 0.95

    benchmark_seed: int = 1524974878092561023
    ablation_seed: int = 5417834325182179367
    batch_size_seed: int = 9173827419318802191
    rff_seed: int = 12309768185157444137
    robustness_seed: int = 16114472213211239267

    def __post_init__(
        self,
    ) -> None:
        resample_count = _require_positive_integer(
            self.resample_count,
            description="bootstrap resample count",
        )

        if resample_count < 2:
            raise ValueError(
                "Bootstrap resample count must be at least 2."
            )

        object.__setattr__(
            self,
            "resample_count",
            resample_count,
        )

        object.__setattr__(
            self,
            "confidence_level",
            _require_probability(
                self.confidence_level,
                description="bootstrap confidence level",
            ),
        )

        object.__setattr__(
            self,
            "benchmark_seed",
            _require_uint64(
                self.benchmark_seed,
                description="benchmark bootstrap seed",
            ),
        )

        object.__setattr__(
            self,
            "ablation_seed",
            _require_uint64(
                self.ablation_seed,
                description="ablation bootstrap seed",
            ),
        )

        object.__setattr__(
            self,
            "batch_size_seed",
            _require_uint64(
                self.batch_size_seed,
                description="batch-size bootstrap seed",
            ),
        )

        object.__setattr__(
            self,
            "rff_seed",
            _require_uint64(
                self.rff_seed,
                description="RFF bootstrap seed",
            ),
        )

        object.__setattr__(
            self,
            "robustness_seed",
            _require_uint64(
                self.robustness_seed,
                description="robustness bootstrap seed",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkSettings:
    persistence_landscape_num_landscapes: tuple[int, ...] = (
        8,
    )

    persistence_landscape_resolutions: tuple[int, ...] = (
        64,
    )

    persistence_image_square_resolutions: tuple[int, ...] = (
        8,
    )

    persistence_image_bandwidths: tuple[float, ...] = (
        0.025,
        0.05,
        0.10,
        0.20,
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "persistence_landscape_num_landscapes",
            _normalize_positive_integer_tuple(
                self.persistence_landscape_num_landscapes,
                description="persistence-landscape counts",
            ),
        )

        object.__setattr__(
            self,
            "persistence_landscape_resolutions",
            _normalize_positive_integer_tuple(
                self.persistence_landscape_resolutions,
                description="persistence-landscape resolutions",
            ),
        )

        object.__setattr__(
            self,
            "persistence_image_square_resolutions",
            _normalize_positive_integer_tuple(
                self.persistence_image_square_resolutions,
                description="persistence-image square resolutions",
            ),
        )

        object.__setattr__(
            self,
            "persistence_image_bandwidths",
            _normalize_positive_float_tuple(
                self.persistence_image_bandwidths,
                description="persistence-image bandwidths",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class PersLaySettings:
    feature_count: int = 64
    epochs: int = 25
    learning_rate: float = 1.0e-3
    seed: int = 9723458291045837211

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "feature_count",
            _require_positive_integer(
                self.feature_count,
                description="PersLay feature count",
            ),
        )

        object.__setattr__(
            self,
            "epochs",
            _require_positive_integer(
                self.epochs,
                description="PersLay epoch count",
            ),
        )

        object.__setattr__(
            self,
            "learning_rate",
            _require_positive_scalar(
                self.learning_rate,
                description="PersLay learning rate",
            ),
        )

        object.__setattr__(
            self,
            "seed",
            _require_uint64(
                self.seed,
                description="PersLay seed",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RandomGraphClassificationConfig:
    expected_cpp_design: ExpectedCppDesign
    selection: ExperimentSelection
    cross_validation: CrossValidationSettings
    preprocessing: PreprocessingSettings
    classifier: ClassifierSettings
    bootstrap: BootstrapSettings
    benchmark: BenchmarkSettings
    perslay: PersLaySettings

    def __post_init__(
        self,
    ) -> None:
        fields = (
            (
                self.expected_cpp_design,
                ExpectedCppDesign,
                "expected C++ design",
            ),
            (
                self.selection,
                ExperimentSelection,
                "experiment selection",
            ),
            (
                self.cross_validation,
                CrossValidationSettings,
                "cross-validation settings",
            ),
            (
                self.preprocessing,
                PreprocessingSettings,
                "preprocessing settings",
            ),
            (
                self.classifier,
                ClassifierSettings,
                "classifier settings",
            ),
            (
                self.bootstrap,
                BootstrapSettings,
                "bootstrap settings",
            ),
            (
                self.benchmark,
                BenchmarkSettings,
                "benchmark settings",
            ),
            (
                self.perslay,
                PersLaySettings,
                "PersLay settings",
            ),
        )

        for value, expected_type, description in fields:
            if not isinstance(
                value,
                expected_type,
            ):
                raise TypeError(
                    f"{description.capitalize()} must be a "
                    f"{expected_type.__name__} instance."
                )


def default_random_graph_classification_config(
) -> RandomGraphClassificationConfig:
    return RandomGraphClassificationConfig(
        expected_cpp_design=EXPECTED_CPP_DESIGN,
        selection=ExperimentSelection(
            benchmark=True,
            harmonic_aggregation_ablation=True,
            batch_size_sensitivity=True,
            rff_sensitivity=True,
            robustness=True,
            scalability=True,
        ),
        cross_validation=CrossValidationSettings(
            outer_fold_count=10,
            inner_fold_count=5,
            outer_seed=6849471059393190731,
            inner_seed=12053211736635842437,
        ),
        preprocessing=PreprocessingSettings(
            variance_tolerance=0.0,
        ),
        classifier=ClassifierSettings(
            c_values=(
                1.0e-3,
                1.0e-2,
                1.0e-1,
                1.0,
                10.0,
                100.0,
                1.0e3,
                1.0e4,
            ),
        ),
        bootstrap=BootstrapSettings(
            resample_count=10_000,
            confidence_level=0.95,
            benchmark_seed=1524974878092561023,
            ablation_seed=5417834325182179367,
            batch_size_seed=9173827419318802191,
            rff_seed=12309768185157444137,
            robustness_seed=16114472213211239267,
        ),
        benchmark=BenchmarkSettings(
            persistence_landscape_num_landscapes=(
                8,
            ),
            persistence_landscape_resolutions=(
                64,
            ),
            persistence_image_square_resolutions=(
                8,
            ),
            persistence_image_bandwidths=(
                0.025,
                0.05,
                0.10,
                0.20,
            ),
        ),
        perslay=PersLaySettings(
            feature_count=64,
            epochs=25,
            learning_rate=1.0e-3,
            seed=9723458291045837211,
        ),
    )


__all__ = [
    "BATCH_SIZE_SENSITIVITY_EXPERIMENT",
    "BENCHMARK_EXPERIMENT",
    "BENCHMARK_METHODS",
    "BOOTSTRAP_INTERVAL_METHOD",
    "BOOTSTRAP_TARGET",
    "BenchmarkSettings",
    "BootstrapSettings",
    "CLASSIFICATION_ESTIMAND",
    "CPP_METADATA_SCHEMA_VERSION",
    "CPP_OUTPUT_SCHEMA_VERSION",
    "CROSS_OBSERVATION_ABLATION",
    "ClassifierSettings",
    "CrossValidationSettings",
    "DEGREE_PRESERVING_REWIRING_PERTURBATION",
    "EDGE_DELETION_PERTURBATION",
    "EDGE_INSERTION_PERTURBATION",
    "EXPECTED_CPP_DESIGN",
    "EXPECTED_ROBUSTNESS_CONDITIONS",
    "ExpectedCppDesign",
    "ExpectedRobustnessCondition",
    "ExperimentSelection",
    "FILTRATION_NOISE_PERTURBATION",
    "GRAPH_STATISTICS_METHOD",
    "HARMONIC_AGGREGATION_ABLATION_EXPERIMENT",
    "HARMONIC_AGGREGATION_ABLATION_KINDS",
    "HARMONIC_AGGREGATION_METHOD",
    "INNER_SELECTION_METRIC",
    "LINEAR_ABLATION",
    "NO_PREORDER_ABLATION",
    "PERSISTENCE_IMAGE_METHOD",
    "PERSISTENCE_LANDSCAPE_METHOD",
    "PERSLAY_METHOD",
    "PRIMARY_CLASSIFICATION_METRIC",
    "PersLaySettings",
    "RANDOM_GRAPH_CLASSIFICATION_CONFIG_SCHEMA_VERSION",
    "RFF_SENSITIVITY_EXPERIMENT",
    "ROBUSTNESS_EXPERIMENT",
    "ROBUSTNESS_PERTURBATIONS",
    "RandomGraphClassificationConfig",
    "SCALABILITY_EXPERIMENT",
    "SECONDARY_CLASSIFICATION_METRIC",
    "SUPPORTED_EXPERIMENTS",
    "default_random_graph_classification_config",
]