from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, TypeAlias, TypeVar

import numpy as np
from gudhi.representations import Landscape
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
DiagramArray: TypeAlias = NDArray[np.float64]

PERSISTENCE_LANDSCAPE_SCHEMA_VERSION: Final[int] = 1

T = TypeVar("T")


class PersistenceLandscapeError(RuntimeError):
    pass


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


def _normalize_iterable(
    values: Iterable[T],
    *,
    description: str,
) -> tuple[T, ...]:
    if isinstance(
        values,
        (str, bytes),
    ):
        raise TypeError(
            f"{description.capitalize()} must be an iterable, not "
            "a string."
        )

    try:
        return tuple(
            values
        )
    except TypeError as error:
        raise TypeError(
            f"{description.capitalize()} must be iterable."
        ) from error


def _canonicalize_diagram(
    diagram: ArrayLike,
    *,
    description: str,
) -> DiagramArray:
    raw = np.asarray(
        diagram
    )

    if (
        raw.ndim == 1
        and raw.size == 0
    ):
        raw = np.empty(
            (
                0,
                2,
            ),
            dtype=np.float64,
        )

    if raw.ndim != 2:
        raise ValueError(
            f"{description.capitalize()} must be two-dimensional."
        )

    if raw.shape[1] != 2:
        raise ValueError(
            f"{description.capitalize()} must have shape "
            "(interval_count, 2)."
        )

    try:
        converted = np.asarray(
            raw,
            dtype=np.float64,
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{description.capitalize()} must contain numeric "
            "birth and death coordinates."
        ) from error

    if (
        converted.shape[0] != 0
        and not bool(
            np.all(
                np.isfinite(
                    converted
                )
            )
        )
    ):
        raise ValueError(
            f"{description.capitalize()} contains a nonfinite "
            "persistence coordinate."
        )

    if (
        converted.shape[0] != 0
        and bool(
            np.any(
                converted[
                    :,
                    0
                ]
                >=
                converted[
                    :,
                    1
                ]
            )
        )
    ):
        raise ValueError(
            f"{description.capitalize()} contains an interval with "
            "birth greater than or equal to death."
        )

    if converted.shape[0] != 0:
        order = np.lexsort(
            (
                converted[
                    :,
                    1
                ],
                converted[
                    :,
                    0
                ],
            )
        )

        converted = converted[
            order
        ]
    else:
        converted = np.empty(
            (
                0,
                2,
            ),
            dtype=np.float64,
        )

    result = np.array(
        converted,
        dtype=np.float64,
        order="C",
        copy=True,
    )

    result.setflags(
        write=False
    )

    return result


def persistence_landscape_dimension(
    num_landscapes: int,
    resolution: int,
) -> int:
    parsed_num_landscapes = _require_positive_integer(
        num_landscapes,
        description="persistence-landscape count",
    )

    parsed_resolution = _require_positive_integer(
        resolution,
        description="persistence-landscape resolution",
    )

    return (
        parsed_num_landscapes
        *
        parsed_resolution
    )


def batch_persistence_landscape_dimension(
    num_landscapes: int,
    resolution: int,
) -> int:
    return persistence_landscape_dimension(
        num_landscapes,
        resolution,
    )


def persistence_landscape_feature_names(
    num_landscapes: int,
    resolution: int,
) -> tuple[str, ...]:
    parsed_num_landscapes = _require_positive_integer(
        num_landscapes,
        description="persistence-landscape count",
    )

    parsed_resolution = _require_positive_integer(
        resolution,
        description="persistence-landscape resolution",
    )

    return tuple(
        (
            f"landscape_layer_{layer_index:03d}_"
            f"sample_{sample_index:04d}"
        )
        for layer_index in range(
            1,
            parsed_num_landscapes
            +
            1,
        )
        for sample_index in range(
            parsed_resolution
        )
    )


def batch_persistence_landscape_feature_names(
    num_landscapes: int,
    resolution: int,
) -> tuple[str, ...]:
    return tuple(
        f"mean_{name}"
        for name in persistence_landscape_feature_names(
            num_landscapes,
            resolution,
        )
    )


@dataclass(
    frozen=True,
    slots=True,
)
class PersistenceLandscapeConfig:
    num_landscapes: int
    resolution: int

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "num_landscapes",
            _require_positive_integer(
                self.num_landscapes,
                description="persistence-landscape count",
            ),
        )

        object.__setattr__(
            self,
            "resolution",
            _require_positive_integer(
                self.resolution,
                description="persistence-landscape resolution",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class FittedPersistenceLandscape:
    num_landscapes: int
    resolution: int
    sample_range_minimum: float
    sample_range_maximum: float
    grid: FloatArray
    _transformer: Landscape

    @property
    def feature_count(
        self,
    ) -> int:
        return persistence_landscape_dimension(
            self.num_landscapes,
            self.resolution,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class PersistenceLandscapeFeatures:
    num_landscapes: int
    resolution: int
    values: FloatArray


@dataclass(
    frozen=True,
    slots=True,
)
class BatchPersistenceLandscapeFeatures:
    graph_count: int
    num_landscapes: int
    resolution: int
    values: FloatArray


def _fitted_landscape_state(
    transformer: Landscape,
    config: PersistenceLandscapeConfig,
) -> tuple[
    float,
    float,
    FloatArray,
]:
    if int(
        transformer.num_landscapes
    ) != config.num_landscapes:
        raise PersistenceLandscapeError(
            "GUDHI Landscape fitted with an unexpected "
            "num_landscapes value."
        )

    if int(
        transformer.resolution
    ) != config.resolution:
        raise PersistenceLandscapeError(
            "GUDHI Landscape fitted with an unexpected resolution."
        )

    if bool(
        transformer.keep_endpoints
    ):
        raise PersistenceLandscapeError(
            "GUDHI Landscape must use keep_endpoints=False."
        )

    if not hasattr(
        transformer,
        "sample_range_fixed_",
    ):
        raise PersistenceLandscapeError(
            "The fitted GUDHI Landscape does not expose "
            "sample_range_fixed_."
        )

    sample_range = np.asarray(
        transformer.sample_range_fixed_,
        dtype=np.float64,
    )

    if (
        sample_range.shape != (
            2,
        )
        or not bool(
            np.all(
                np.isfinite(
                    sample_range
                )
            )
        )
        or sample_range[
            0
        ]
        >=
        sample_range[
            1
        ]
    ):
        raise PersistenceLandscapeError(
            "GUDHI returned an invalid fitted persistence-landscape "
            "sample range."
        )

    if not hasattr(
        transformer,
        "grid_",
    ):
        raise PersistenceLandscapeError(
            "The fitted GUDHI Landscape does not expose grid_."
        )

    grid = np.asarray(
        transformer.grid_,
        dtype=np.float64,
    )

    if (
        grid.ndim != 1
        or grid.size != config.resolution
        or not bool(
            np.all(
                np.isfinite(
                    grid
                )
            )
        )
        or (
            grid.size > 1
            and bool(
                np.any(
                    np.diff(
                        grid
                    )
                    <=
                    0.0
                )
            )
        )
    ):
        raise PersistenceLandscapeError(
            "GUDHI returned an invalid persistence-landscape "
            "sampling grid."
        )

    frozen_grid = np.array(
        grid,
        dtype=np.float64,
        order="C",
        copy=True,
    )

    frozen_grid.setflags(
        write=False
    )

    return (
        float(
            sample_range[
                0
            ]
        ),
        float(
            sample_range[
                1
            ]
        ),
        frozen_grid,
    )


def fit_persistence_landscape(
    training_diagrams: Iterable[ArrayLike],
    config: PersistenceLandscapeConfig,
) -> FittedPersistenceLandscape:
    """
    Fit the landscape sampling range from training diagrams only.

    Nested cross-validation must call this function separately on
    each inner-training domain and again on the selected outer-
    training domain. Validation and test diagrams must never enter
    this fit.
    """

    if not isinstance(
        config,
        PersistenceLandscapeConfig,
    ):
        raise TypeError(
            "config must be a PersistenceLandscapeConfig instance."
        )

    diagrams = _normalize_iterable(
        training_diagrams,
        description="training persistence diagrams",
    )

    if not diagrams:
        raise ValueError(
            "Persistence-landscape fitting requires at least one "
            "training diagram."
        )

    canonical_diagrams = tuple(
        _canonicalize_diagram(
            diagram,
            description=(
                f"training persistence diagram {diagram_index}"
            ),
        )
        for diagram_index, diagram in enumerate(
            diagrams
        )
    )

    nonempty_diagrams = tuple(
        diagram
        for diagram in canonical_diagrams
        if diagram.shape[
            0
        ]
        !=
        0
    )

    if not nonempty_diagrams:
        raise ValueError(
            "Persistence-landscape fitting requires at least one "
            "nonempty training diagram."
        )

    transformer = Landscape(
        num_landscapes=config.num_landscapes,
        resolution=config.resolution,
        sample_range=[
            np.nan,
            np.nan,
        ],
        keep_endpoints=False,
    )

    transformer.fit(
        list(
            nonempty_diagrams
        )
    )

    (
        sample_range_minimum,
        sample_range_maximum,
        grid,
    ) = _fitted_landscape_state(
        transformer,
        config,
    )

    return FittedPersistenceLandscape(
        num_landscapes=config.num_landscapes,
        resolution=config.resolution,
        sample_range_minimum=sample_range_minimum,
        sample_range_maximum=sample_range_maximum,
        grid=grid,
        _transformer=transformer,
    )


def transform_persistence_landscapes(
    fitted: FittedPersistenceLandscape,
    diagrams: Iterable[ArrayLike],
) -> FloatArray:
    if not isinstance(
        fitted,
        FittedPersistenceLandscape,
    ):
        raise TypeError(
            "fitted must be a FittedPersistenceLandscape instance."
        )

    normalized = _normalize_iterable(
        diagrams,
        description="persistence diagrams",
    )

    feature_count = fitted.feature_count

    if not normalized:
        result = np.empty(
            (
                0,
                feature_count,
            ),
            dtype=np.float64,
        )

        result.setflags(
            write=False
        )

        return result

    canonical_diagrams = tuple(
        _canonicalize_diagram(
            diagram,
            description=(
                f"persistence diagram {diagram_index}"
            ),
        )
        for diagram_index, diagram in enumerate(
            normalized
        )
    )

    result = np.zeros(
        (
            len(
                canonical_diagrams
            ),
            feature_count,
        ),
        dtype=np.float64,
    )

    nonempty_indices = tuple(
        index
        for index, diagram in enumerate(
            canonical_diagrams
        )
        if diagram.shape[
            0
        ]
        !=
        0
    )

    if nonempty_indices:
        nonempty_diagrams = [
            canonical_diagrams[
                index
            ]
            for index in nonempty_indices
        ]

        transformed = np.asarray(
            fitted._transformer.transform(
                nonempty_diagrams
            ),
            dtype=np.float64,
        )

        expected_shape = (
            len(
                nonempty_indices
            ),
            feature_count,
        )

        if transformed.shape != expected_shape:
            raise PersistenceLandscapeError(
                "GUDHI returned a persistence-landscape feature "
                "matrix with an unexpected shape."
            )

        if not bool(
            np.all(
                np.isfinite(
                    transformed
                )
            )
        ):
            raise PersistenceLandscapeError(
                "GUDHI returned nonfinite persistence-landscape "
                "features."
            )

        result[
            np.asarray(
                nonempty_indices,
                dtype=np.int64,
            ),
            :,
        ] = transformed

    result.setflags(
        write=False
    )

    return result


def transform_persistence_landscape(
    fitted: FittedPersistenceLandscape,
    diagram: ArrayLike,
) -> PersistenceLandscapeFeatures:
    matrix = transform_persistence_landscapes(
        fitted,
        (
            diagram,
        ),
    )

    return PersistenceLandscapeFeatures(
        num_landscapes=fitted.num_landscapes,
        resolution=fitted.resolution,
        values=matrix[
            0
        ],
    )


def _as_graph_feature_matrix(
    values: ArrayLike,
    *,
    expected_columns: int,
) -> FloatArray:
    matrix = np.asarray(
        values,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Graph-level persistence-landscape features must form "
            "a two-dimensional matrix."
        )

    if matrix.shape[
        0
    ] == 0:
        raise ValueError(
            "Persistence-landscape batch aggregation requires at "
            "least one graph."
        )

    if matrix.shape[
        1
    ] != expected_columns:
        raise ValueError(
            "Graph-level persistence-landscape feature dimension "
            f"{matrix.shape[1]} does not match the expected "
            f"dimension {expected_columns}."
        )

    return matrix


def aggregate_persistence_landscapes(
    graph_features: ArrayLike,
    *,
    num_landscapes: int,
    resolution: int,
) -> BatchPersistenceLandscapeFeatures:
    """
    Aggregate graph persistence landscapes by coordinatewise mean.
    """

    parsed_num_landscapes = _require_positive_integer(
        num_landscapes,
        description="persistence-landscape count",
    )

    parsed_resolution = _require_positive_integer(
        resolution,
        description="persistence-landscape resolution",
    )

    graph_feature_count = persistence_landscape_dimension(
        parsed_num_landscapes,
        parsed_resolution,
    )

    matrix = _as_graph_feature_matrix(
        graph_features,
        expected_columns=graph_feature_count,
    )

    values = np.asarray(
        np.mean(
            matrix,
            axis=0,
            dtype=np.float64,
        ),
        dtype=np.float64,
    )

    values.setflags(
        write=False
    )

    return BatchPersistenceLandscapeFeatures(
        graph_count=int(
            matrix.shape[
                0
            ]
        ),
        num_landscapes=parsed_num_landscapes,
        resolution=parsed_resolution,
        values=values,
    )


__all__ = [
    "BatchPersistenceLandscapeFeatures",
    "FittedPersistenceLandscape",
    "PERSISTENCE_LANDSCAPE_SCHEMA_VERSION",
    "PersistenceLandscapeConfig",
    "PersistenceLandscapeError",
    "PersistenceLandscapeFeatures",
    "aggregate_persistence_landscapes",
    "batch_persistence_landscape_dimension",
    "batch_persistence_landscape_feature_names",
    "fit_persistence_landscape",
    "persistence_landscape_dimension",
    "persistence_landscape_feature_names",
    "transform_persistence_landscape",
    "transform_persistence_landscapes",
]