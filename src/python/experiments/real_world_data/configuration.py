from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final


REAL_WORLD_DATA_ANALYSIS_SCHEMA_VERSION: Final[int] = 4

EXPERIMENT_NAME: Final[str] = "real_world_data"

DATASETS: Final[tuple[str, ...]] = (
    "AIDS",
    "BZR",
    "COX2",
    "MUTAG",
    "Mutagenicity",
    "NCI1",
    "NCI109",
    "PROTEINS",
)

BAG_SIZES: Final[tuple[int, ...]] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
)

CHARACTER_COUNT: Final[int] = 256
HARMONIC_FEATURE_COUNT: Final[int] = 2 * CHARACTER_COUNT
PERSLAY_FEATURE_COUNT: Final[int] = 64

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

SVM_KERNEL: Final[str] = "rbf"
SVM_GAMMA: Final[str] = "scale"
SVM_CLASS_WEIGHT: Final[None] = None
SVM_PROBABILITY: Final[bool] = False

_UINT64_MAX: Final[int] = (1 << 64) - 1


def _require_integer(
    value: int,
    *,
    description: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
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


def _require_nonnegative_real(
    value: float,
    *,
    description: str,
) -> float:
    if (
        isinstance(value, bool)
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

    if (
        not math.isfinite(parsed)
        or parsed < 0.0
    ):
        raise ValueError(
            f"{description.capitalize()} must be finite and "
            "nonnegative."
        )

    return parsed


def _require_positive_real(
    value: float,
    *,
    description: str,
) -> float:
    if (
        isinstance(value, bool)
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

    if (
        not math.isfinite(parsed)
        or parsed <= 0.0
    ):
        raise ValueError(
            f"{description.capitalize()} must be finite and "
            "positive."
        )

    return parsed


def _require_probability(
    value: float,
    *,
    description: str,
) -> float:
    if (
        isinstance(value, bool)
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

    if (
        not math.isfinite(parsed)
        or not 0.0 < parsed < 1.0
    ):
        raise ValueError(
            f"{description.capitalize()} must lie in (0, 1)."
        )

    return parsed


def _normalize_positive_float_tuple(
    values: tuple[float, ...],
    *,
    description: str,
) -> tuple[float, ...]:
    normalized = tuple(
        float(value)
        for value in values
    )

    if not normalized:
        raise ValueError(
            f"{description.capitalize()} must not be empty."
        )

    if any(
        (
            not math.isfinite(value)
            or value <= 0.0
        )
        for value in normalized
    ):
        raise ValueError(
            f"{description.capitalize()} must contain only "
            "positive finite values."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{description.capitalize()} must not contain "
            "duplicates."
        )

    return normalized


def _normalize_positive_integer_tuple(
    values: tuple[int, ...],
    *,
    description: str,
) -> tuple[int, ...]:
    normalized = tuple(
        _require_positive_integer(
            value,
            description=f"{description} entry",
        )
        for value in values
    )

    if not normalized:
        raise ValueError(
            f"{description.capitalize()} must not be empty."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{description.capitalize()} must not contain "
            "duplicates."
        )

    return normalized


def _normalize_bag_sizes(
    values: tuple[int, ...],
) -> tuple[int, ...]:
    return _normalize_positive_integer_tuple(
        values,
        description="bag sizes",
    )


@dataclass(
    frozen=True,
    slots=True,
)
class CrossValidationSettings:
    outer_fold_count: int = 5
    inner_fold_count: int = 3

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
class MilSettings:
    bag_sizes: tuple[int, ...] = BAG_SIZES
    bag_seed: int = 11390238479102561291

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "bag_sizes",
            _normalize_bag_sizes(
                self.bag_sizes
            ),
        )

        object.__setattr__(
            self,
            "bag_seed",
            _require_uint64(
                self.bag_seed,
                description="MIL bag-construction seed",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class PreprocessingSettings:
    variance_tolerance: float = 1.0e-10

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "variance_tolerance",
            _require_nonnegative_real(
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
        1.0e-2,
        1.0e-1,
        1.0,
        10.0,
        100.0,
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


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkSettings:
    persistence_landscape_num_landscapes: tuple[int, ...] = (
        3,
    )

    persistence_landscape_resolutions: tuple[int, ...] = (
        24,
    )

    persistence_image_square_resolutions: tuple[int, ...] = (
        8,
    )

    persistence_image_bandwidths: tuple[float, ...] = (
        0.10,
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
class PerslaySettings:
    feature_count: int = PERSLAY_FEATURE_COUNT
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
            _require_positive_real(
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
class BootstrapSettings:
    resample_count: int = 10_000
    confidence_level: float = 0.95
    seed: int = 1524974878092561023

    def __post_init__(
        self,
    ) -> None:
        count = _require_positive_integer(
            self.resample_count,
            description="bootstrap resample count",
        )

        if count < 2:
            raise ValueError(
                "Bootstrap resample count must be at least 2."
            )

        object.__setattr__(
            self,
            "resample_count",
            count,
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
            "seed",
            _require_uint64(
                self.seed,
                description="bootstrap seed",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RealWorldDataConfig:
    cross_validation: CrossValidationSettings
    mil: MilSettings
    preprocessing: PreprocessingSettings
    classifier: ClassifierSettings
    benchmark: BenchmarkSettings
    perslay: PerslaySettings
    bootstrap: BootstrapSettings

    def __post_init__(
        self,
    ) -> None:
        fields = (
            (
                self.cross_validation,
                CrossValidationSettings,
                "cross-validation settings",
            ),
            (
                self.mil,
                MilSettings,
                "MIL settings",
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
                self.benchmark,
                BenchmarkSettings,
                "benchmark settings",
            ),
            (
                self.perslay,
                PerslaySettings,
                "PersLay settings",
            ),
            (
                self.bootstrap,
                BootstrapSettings,
                "bootstrap settings",
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


def default_real_world_data_config(
) -> RealWorldDataConfig:
    return RealWorldDataConfig(
        cross_validation=CrossValidationSettings(),
        mil=MilSettings(),
        preprocessing=PreprocessingSettings(),
        classifier=ClassifierSettings(),
        benchmark=BenchmarkSettings(),
        perslay=PerslaySettings(),
        bootstrap=BootstrapSettings(),
    )


__all__ = [
    "BAG_SIZES",
    "BENCHMARK_METHODS",
    "BenchmarkSettings",
    "BootstrapSettings",
    "CHARACTER_COUNT",
    "CLASSIFICATION_ESTIMAND",
    "ClassifierSettings",
    "CrossValidationSettings",
    "DATASETS",
    "EXPERIMENT_NAME",
    "GRAPH_STATISTICS_METHOD",
    "HARMONIC_AGGREGATION_METHOD",
    "HARMONIC_FEATURE_COUNT",
    "INNER_SELECTION_METRIC",
    "MilSettings",
    "PERSISTENCE_IMAGE_METHOD",
    "PERSISTENCE_LANDSCAPE_METHOD",
    "PERSLAY_FEATURE_COUNT",
    "PERSLAY_METHOD",
    "PRIMARY_CLASSIFICATION_METRIC",
    "PerslaySettings",
    "PreprocessingSettings",
    "REAL_WORLD_DATA_ANALYSIS_SCHEMA_VERSION",
    "RealWorldDataConfig",
    "SECONDARY_CLASSIFICATION_METRIC",
    "default_real_world_data_config",
]