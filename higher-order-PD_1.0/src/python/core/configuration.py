from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final


DEFAULT_OUTER_FOLDS: Final[int] = 10
DEFAULT_INNER_FOLDS: Final[int] = 5
DEFAULT_BOOTSTRAP_RESAMPLES: Final[int] = 10_000
DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95

DEFAULT_SVM_C_VALUES: Final[tuple[float, ...]] = (
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    1.0e1,
    1.0e2,
)

DEFAULT_SVM_CACHE_SIZE_MB: Final[float] = 1024.0
DEFAULT_VARIANCE_TOLERANCE: Final[float] = 1.0e-12

DEFAULT_RANDOM_SEED: Final[int] = 0
DEFAULT_WORKER_COUNT: Final[int] = 1


def _require_instance(
    value: object,
    expected_type: type,
    description: str,
) -> None:
    if not isinstance(description, str):
        raise TypeError(
            "A validation description must be a string."
        )

    if not description:
        raise ValueError(
            "A validation description must not be empty."
        )

    if not isinstance(value, expected_type):
        raise TypeError(
            f"{description} must be an instance of "
            f"{expected_type.__name__}."
        )


def _validate_integer(
    value: object,
    description: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{description} must be an integer."
        )

    return value


def _validate_positive_integer(
    value: object,
    description: str,
) -> int:
    parsed = _validate_integer(
        value,
        description,
    )

    if parsed <= 0:
        raise ValueError(
            f"{description} must be positive."
        )

    return parsed


def _validate_integer_at_least(
    value: object,
    minimum: int,
    description: str,
) -> int:
    parsed = _validate_integer(
        value,
        description,
    )

    minimum_value = _validate_integer(
        minimum,
        "The minimum integer value",
    )

    if parsed < minimum_value:
        raise ValueError(
            f"{description} must be at least "
            f"{minimum_value}."
        )

    return parsed


def _validate_nonnegative_integer(
    value: object,
    description: str,
) -> int:
    parsed = _validate_integer(
        value,
        description,
    )

    if parsed < 0:
        raise ValueError(
            f"{description} must be nonnegative."
        )

    return parsed


def _validate_real(
    value: object,
    description: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(
            f"{description} must be a real number."
        )

    parsed = float(value)

    if not math.isfinite(parsed):
        raise ValueError(
            f"{description} must be finite."
        )

    if parsed == 0.0:
        parsed = 0.0

    return parsed


def _validate_positive_real(
    value: object,
    description: str,
) -> float:
    parsed = _validate_real(
        value,
        description,
    )

    if parsed <= 0.0:
        raise ValueError(
            f"{description} must be positive."
        )

    return parsed


def _validate_nonnegative_real(
    value: object,
    description: str,
) -> float:
    parsed = _validate_real(
        value,
        description,
    )

    if parsed < 0.0:
        raise ValueError(
            f"{description} must be nonnegative."
        )

    return parsed


def _validate_open_unit_interval(
    value: object,
    description: str,
) -> float:
    parsed = _validate_real(
        value,
        description,
    )

    if not 0.0 < parsed < 1.0:
        raise ValueError(
            f"{description} must lie strictly between "
            "zero and one."
        )

    return parsed


def _validate_closed_unit_interval(
    value: object,
    description: str,
) -> float:
    parsed = _validate_real(
        value,
        description,
    )

    if not 0.0 <= parsed <= 1.0:
        raise ValueError(
            f"{description} must lie between zero and "
            "one inclusive."
        )

    return parsed


def _normalize_positive_real_tuple(
    values: object,
    description: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{description} must be an iterable of "
            "positive real numbers."
        )

    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(
            f"{description} must be an iterable of "
            "positive real numbers."
        ) from error

    if not raw_values:
        raise ValueError(
            f"{description} must not be empty."
        )

    normalized = tuple(
        _validate_positive_real(
            value,
            f"{description} entry {index}",
        )
        for index, value in enumerate(
            raw_values,
            start=1,
        )
    )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{description} must not contain duplicate "
            "values."
        )

    return normalized


@dataclass(frozen=True)
class CrossValidationConfig:
    outer_folds: int = DEFAULT_OUTER_FOLDS
    inner_folds: int = DEFAULT_INNER_FOLDS
    random_seed: int = DEFAULT_RANDOM_SEED
    shuffle: bool = True
    stratified: bool = True

    def __post_init__(self) -> None:
        outer_folds = _validate_integer_at_least(
            self.outer_folds,
            2,
            "The outer cross-validation fold count",
        )

        inner_folds = _validate_integer_at_least(
            self.inner_folds,
            2,
            "The inner cross-validation fold count",
        )

        random_seed = _validate_nonnegative_integer(
            self.random_seed,
            "The cross-validation random seed",
        )

        if not isinstance(self.shuffle, bool):
            raise TypeError(
                "The cross-validation shuffle flag "
                "must be a boolean."
            )

        if not isinstance(self.stratified, bool):
            raise TypeError(
                "The cross-validation stratification "
                "flag must be a boolean."
            )

        if not self.stratified:
            raise ValueError(
                "The experimental protocol requires "
                "stratified cross-validation."
            )

        if not self.shuffle:
            raise ValueError(
                "The experimental protocol requires "
                "shuffled cross-validation so the "
                "configured random seed determines the "
                "shared fold assignment."
            )

        object.__setattr__(
            self,
            "outer_folds",
            outer_folds,
        )

        object.__setattr__(
            self,
            "inner_folds",
            inner_folds,
        )

        object.__setattr__(
            self,
            "random_seed",
            random_seed,
        )


@dataclass(frozen=True)
class StatisticsConfig:
    bootstrap_resamples: int = (
        DEFAULT_BOOTSTRAP_RESAMPLES
    )
    confidence_level: float = (
        DEFAULT_CONFIDENCE_LEVEL
    )
    random_seed: int = DEFAULT_RANDOM_SEED
    paired_bootstrap: bool = True
    stratified_bootstrap: bool = True

    def __post_init__(self) -> None:
        bootstrap_resamples = (
            _validate_positive_integer(
                self.bootstrap_resamples,
                "The bootstrap resample count",
            )
        )

        confidence_level = (
            _validate_open_unit_interval(
                self.confidence_level,
                "The confidence level",
            )
        )

        random_seed = _validate_nonnegative_integer(
            self.random_seed,
            "The bootstrap random seed",
        )

        if not isinstance(
            self.paired_bootstrap,
            bool,
        ):
            raise TypeError(
                "The paired-bootstrap flag must be a "
                "boolean."
            )

        if not isinstance(
            self.stratified_bootstrap,
            bool,
        ):
            raise TypeError(
                "The stratified-bootstrap flag must be "
                "a boolean."
            )

        if not self.paired_bootstrap:
            raise ValueError(
                "The experimental protocol requires a "
                "paired bootstrap."
            )

        if not self.stratified_bootstrap:
            raise ValueError(
                "The experimental protocol requires a "
                "stratified bootstrap."
            )

        object.__setattr__(
            self,
            "bootstrap_resamples",
            bootstrap_resamples,
        )

        object.__setattr__(
            self,
            "confidence_level",
            confidence_level,
        )

        object.__setattr__(
            self,
            "random_seed",
            random_seed,
        )


@dataclass(frozen=True)
class SvmConfig:
    c_values: tuple[float, ...] = (
        DEFAULT_SVM_C_VALUES
    )
    kernel: str = "linear"
    class_weight: str | None = None
    cache_size_mb: float = (
        DEFAULT_SVM_CACHE_SIZE_MB
    )
    probability: bool = False

    def __post_init__(self) -> None:
        c_values = _normalize_positive_real_tuple(
            self.c_values,
            "The SVM C grid",
        )

        if not isinstance(self.kernel, str):
            raise TypeError(
                "The SVM kernel must be a string."
            )

        kernel = self.kernel.strip()

        if not kernel:
            raise ValueError(
                "The SVM kernel must not be empty."
            )

        if kernel != "linear":
            raise ValueError(
                "The current experimental protocol "
                "requires a linear SVM kernel."
            )

        if (
            self.class_weight is not None
            and not isinstance(
                self.class_weight,
                str,
            )
        ):
            raise TypeError(
                "The SVM class weight must be None or "
                "a string."
            )

        class_weight = self.class_weight

        if isinstance(class_weight, str):
            class_weight = class_weight.strip()

            if not class_weight:
                raise ValueError(
                    "The SVM class-weight string must "
                    "not be empty."
                )

            if class_weight != "balanced":
                raise ValueError(
                    "The SVM class weight must be None "
                    "or 'balanced'."
                )

        cache_size_mb = _validate_positive_real(
            self.cache_size_mb,
            "The SVM cache size in megabytes",
        )

        if not isinstance(self.probability, bool):
            raise TypeError(
                "The SVM probability flag must be a "
                "boolean."
            )

        object.__setattr__(
            self,
            "c_values",
            c_values,
        )

        object.__setattr__(
            self,
            "kernel",
            kernel,
        )

        object.__setattr__(
            self,
            "class_weight",
            class_weight,
        )

        object.__setattr__(
            self,
            "cache_size_mb",
            cache_size_mb,
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    remove_low_variance: bool = True
    variance_tolerance: float = (
        DEFAULT_VARIANCE_TOLERANCE
    )
    standardize: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.remove_low_variance,
            bool,
        ):
            raise TypeError(
                "The low-variance filtering flag must "
                "be a boolean."
            )

        variance_tolerance = (
            _validate_nonnegative_real(
                self.variance_tolerance,
                "The variance tolerance",
            )
        )

        if not isinstance(self.standardize, bool):
            raise TypeError(
                "The standardization flag must be a "
                "boolean."
            )

        if not self.remove_low_variance:
            raise ValueError(
                "The current experimental protocol "
                "requires low-variance coordinate "
                "filtering."
            )

        if not self.standardize:
            raise ValueError(
                "The current experimental protocol "
                "requires training-fold "
                "standardization."
            )

        object.__setattr__(
            self,
            "variance_tolerance",
            variance_tolerance,
        )


@dataclass(frozen=True)
class SerializationConfig:
    save_configuration: bool = True
    save_fold_assignments: bool = True
    save_selected_hyperparameters: bool = True
    save_out_of_fold_predictions: bool = True
    save_metrics: bool = True
    save_preprocessing_metadata: bool = True
    save_inner_fold_models: bool = False
    save_outer_fold_models: bool = False
    save_final_refit_model: bool = False

    def __post_init__(self) -> None:
        boolean_fields = (
            (
                self.save_configuration,
                "The configuration serialization flag",
            ),
            (
                self.save_fold_assignments,
                "The fold-assignment serialization flag",
            ),
            (
                self.save_selected_hyperparameters,
                "The selected-hyperparameter "
                "serialization flag",
            ),
            (
                self.save_out_of_fold_predictions,
                "The out-of-fold prediction "
                "serialization flag",
            ),
            (
                self.save_metrics,
                "The metric serialization flag",
            ),
            (
                self.save_preprocessing_metadata,
                "The preprocessing-metadata "
                "serialization flag",
            ),
            (
                self.save_inner_fold_models,
                "The inner-fold model serialization "
                "flag",
            ),
            (
                self.save_outer_fold_models,
                "The outer-fold model serialization "
                "flag",
            ),
            (
                self.save_final_refit_model,
                "The final-refit model serialization "
                "flag",
            ),
        )

        for value, description in boolean_fields:
            if not isinstance(value, bool):
                raise TypeError(
                    f"{description} must be a boolean."
                )

        if self.save_inner_fold_models:
            raise ValueError(
                "The analysis configuration does not "
                "persist inner-fold SVM models."
            )

        if self.save_outer_fold_models:
            raise ValueError(
                "The analysis configuration does not "
                "persist outer-fold SVM models."
            )


@dataclass(frozen=True)
class PlottingConfig:
    dpi: int = 300
    save_png: bool = True
    save_pdf: bool = True
    close_figures: bool = True

    def __post_init__(self) -> None:
        dpi = _validate_positive_integer(
            self.dpi,
            "The plotting resolution",
        )

        if not isinstance(self.save_png, bool):
            raise TypeError(
                "The PNG-output flag must be a boolean."
            )

        if not isinstance(self.save_pdf, bool):
            raise TypeError(
                "The PDF-output flag must be a boolean."
            )

        if not isinstance(
            self.close_figures,
            bool,
        ):
            raise TypeError(
                "The figure-closing flag must be a "
                "boolean."
            )

        if not (
            self.save_png
            or self.save_pdf
        ):
            raise ValueError(
                "At least one plotting output format "
                "must be enabled."
            )

        object.__setattr__(
            self,
            "dpi",
            dpi,
        )


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    overwrite: bool = False
    validate_before_reuse: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                "The cache-enabled flag must be a "
                "boolean."
            )

        if not isinstance(self.overwrite, bool):
            raise TypeError(
                "The cache-overwrite flag must be a "
                "boolean."
            )

        if not isinstance(
            self.validate_before_reuse,
            bool,
        ):
            raise TypeError(
                "The cache-validation flag must be a "
                "boolean."
            )

        if (
            self.overwrite
            and not self.enabled
        ):
            raise ValueError(
                "Cache overwrite cannot be enabled "
                "while caching is disabled."
            )

        if (
            self.enabled
            and not self.validate_before_reuse
        ):
            raise ValueError(
                "Reusable cached analysis artifacts "
                "must be validated before use."
            )


@dataclass(frozen=True)
class ExecutionConfig:
    worker_count: int = DEFAULT_WORKER_COUNT
    random_seed: int = DEFAULT_RANDOM_SEED
    fail_fast: bool = True

    def __post_init__(self) -> None:
        worker_count = _validate_positive_integer(
            self.worker_count,
            "The analysis worker count",
        )

        random_seed = _validate_nonnegative_integer(
            self.random_seed,
            "The analysis execution random seed",
        )

        if not isinstance(self.fail_fast, bool):
            raise TypeError(
                "The fail-fast flag must be a boolean."
            )

        object.__setattr__(
            self,
            "worker_count",
            worker_count,
        )

        object.__setattr__(
            self,
            "random_seed",
            random_seed,
        )


@dataclass(frozen=True)
class CoreAnalysisConfig:
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig
    )
    statistics: StatisticsConfig = field(
        default_factory=StatisticsConfig
    )
    svm: SvmConfig = field(
        default_factory=SvmConfig
    )
    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )
    serialization: SerializationConfig = field(
        default_factory=SerializationConfig
    )
    plotting: PlottingConfig = field(
        default_factory=PlottingConfig
    )
    cache: CacheConfig = field(
        default_factory=CacheConfig
    )
    execution: ExecutionConfig = field(
        default_factory=ExecutionConfig
    )

    def __post_init__(self) -> None:
        _require_instance(
            self.cross_validation,
            CrossValidationConfig,
            "The cross-validation configuration",
        )

        _require_instance(
            self.statistics,
            StatisticsConfig,
            "The statistics configuration",
        )

        _require_instance(
            self.svm,
            SvmConfig,
            "The SVM configuration",
        )

        _require_instance(
            self.preprocessing,
            PreprocessingConfig,
            "The preprocessing configuration",
        )

        _require_instance(
            self.serialization,
            SerializationConfig,
            "The serialization configuration",
        )

        _require_instance(
            self.plotting,
            PlottingConfig,
            "The plotting configuration",
        )

        _require_instance(
            self.cache,
            CacheConfig,
            "The cache configuration",
        )

        _require_instance(
            self.execution,
            ExecutionConfig,
            "The execution configuration",
        )