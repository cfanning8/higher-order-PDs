from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, TypeAlias, TypeVar

import numpy as np
from gudhi.representations import PersistenceImage
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
DiagramArray: TypeAlias = NDArray[np.float64]

PERSISTENCE_IMAGE_SCHEMA_VERSION: Final[int] = 2

T = TypeVar("T")


class PersistenceImageError(RuntimeError):
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


def _require_positive_scalar(
    value: float,
    *,
    description: str,
) -> float:
    if (
        isinstance(
            value,
            (bool, np.bool_),
        )
        or not isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
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
        not np.isfinite(
            parsed
        )
        or parsed <= 0.0
    ):
        raise ValueError(
            f"{description.capitalize()} must be finite and "
            "strictly positive."
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


def _persistence_weight(
    point: ArrayLike,
) -> float:
    return float(
        point[
            1
        ]
    )


def persistence_image_dimension(
    resolution_x: int,
    resolution_y: int,
) -> int:
    parsed_resolution_x = _require_positive_integer(
        resolution_x,
        description="persistence-image x resolution",
    )

    parsed_resolution_y = _require_positive_integer(
        resolution_y,
        description="persistence-image y resolution",
    )

    return (
        parsed_resolution_x
        *
        parsed_resolution_y
    )


def batch_persistence_image_dimension(
    resolution_x: int,
    resolution_y: int,
) -> int:
    return persistence_image_dimension(
        resolution_x,
        resolution_y,
    )


def persistence_image_feature_names(
    resolution_x: int,
    resolution_y: int,
) -> tuple[str, ...]:
    parsed_resolution_x = _require_positive_integer(
        resolution_x,
        description="persistence-image x resolution",
    )

    parsed_resolution_y = _require_positive_integer(
        resolution_y,
        description="persistence-image y resolution",
    )

    return tuple(
        (
            f"image_y_{y_index:04d}_"
            f"x_{x_index:04d}"
        )
        for y_index in range(
            parsed_resolution_y
        )
        for x_index in range(
            parsed_resolution_x
        )
    )


def batch_persistence_image_feature_names(
    resolution_x: int,
    resolution_y: int,
) -> tuple[str, ...]:
    return tuple(
        f"mean_{name}"
        for name in persistence_image_feature_names(
            resolution_x,
            resolution_y,
        )
    )


@dataclass(
    frozen=True,
    slots=True,
)
class PersistenceImageConfig:
    bandwidth: float
    resolution_x: int
    resolution_y: int

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "bandwidth",
            _require_positive_scalar(
                self.bandwidth,
                description="persistence-image bandwidth",
            ),
        )

        object.__setattr__(
            self,
            "resolution_x",
            _require_positive_integer(
                self.resolution_x,
                description="persistence-image x resolution",
            ),
        )

        object.__setattr__(
            self,
            "resolution_y",
            _require_positive_integer(
                self.resolution_y,
                description="persistence-image y resolution",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class FittedPersistenceImage:
    bandwidth: float
    resolution_x: int
    resolution_y: int
    birth_minimum: float
    birth_maximum: float
    persistence_minimum: float
    persistence_maximum: float
    _transformer: PersistenceImage

    @property
    def feature_count(
        self,
    ) -> int:
        return persistence_image_dimension(
            self.resolution_x,
            self.resolution_y,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class PersistenceImageFeatures:
    bandwidth: float
    resolution_x: int
    resolution_y: int
    values: FloatArray


@dataclass(
    frozen=True,
    slots=True,
)
class BatchPersistenceImageFeatures:
    graph_count: int
    bandwidth: float
    resolution_x: int
    resolution_y: int
    values: FloatArray


def _fitted_image_state(
    transformer: PersistenceImage,
    config: PersistenceImageConfig,
    *,
    require_positive_width: bool,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    if float(
        transformer.bandwidth
    ) != config.bandwidth:
        raise PersistenceImageError(
            "GUDHI PersistenceImage fitted with an unexpected "
            "bandwidth."
        )

    resolution = tuple(
        int(
            value
        )
        for value in transformer.resolution
    )

    if resolution != (
        config.resolution_x,
        config.resolution_y,
    ):
        raise PersistenceImageError(
            "GUDHI PersistenceImage fitted with an unexpected "
            "resolution."
        )

    if transformer.weight is not _persistence_weight:
        raise PersistenceImageError(
            "GUDHI PersistenceImage does not use the required "
            "linear persistence weight."
        )

    if not hasattr(
        transformer,
        "im_range_fixed_",
    ):
        raise PersistenceImageError(
            "The fitted GUDHI PersistenceImage does not expose "
            "im_range_fixed_."
        )

    fitted_range = np.asarray(
        transformer.im_range_fixed_,
        dtype=np.float64,
    )

    if (
        fitted_range.shape != (
            4,
        )
        or not bool(
            np.all(
                np.isfinite(
                    fitted_range
                )
            )
        )
    ):
        raise PersistenceImageError(
            "GUDHI returned an invalid fitted persistence-image "
            "range."
        )

    birth_minimum = float(
        fitted_range[
            0
        ]
    )

    birth_maximum = float(
        fitted_range[
            1
        ]
    )

    persistence_minimum = float(
        fitted_range[
            2
        ]
    )

    persistence_maximum = float(
        fitted_range[
            3
        ]
    )

    if birth_minimum > birth_maximum:
        raise PersistenceImageError(
            "GUDHI returned a persistence-image birth range with "
            "negative width."
        )

    if persistence_minimum > persistence_maximum:
        raise PersistenceImageError(
            "GUDHI returned a persistence-image persistence range "
            "with negative width."
        )

    if require_positive_width:
        if birth_minimum >= birth_maximum:
            raise PersistenceImageError(
                "GUDHI returned a persistence-image birth range "
                "without positive width."
            )

        if persistence_minimum >= persistence_maximum:
            raise PersistenceImageError(
                "GUDHI returned a persistence-image persistence "
                "range without positive width."
            )

    return (
        birth_minimum,
        birth_maximum,
        persistence_minimum,
        persistence_maximum,
    )


def _positive_width_range(
    minimum: float,
    maximum: float,
    *,
    bandwidth: float,
    description: str,
) -> tuple[
    float,
    float,
]:
    if minimum > maximum:
        raise PersistenceImageError(
            f"{description.capitalize()} has negative width."
        )

    if minimum < maximum:
        return (
            minimum,
            maximum,
        )

    center = minimum

    expanded_minimum = (
        center
        -
        bandwidth
    )

    expanded_maximum = (
        center
        +
        bandwidth
    )

    if (
        not np.isfinite(
            expanded_minimum
        )
        or not np.isfinite(
            expanded_maximum
        )
        or expanded_minimum >= expanded_maximum
    ):
        raise PersistenceImageError(
            f"{description.capitalize()} could not be expanded to "
            "positive width."
        )

    return (
        float(
            expanded_minimum
        ),
        float(
            expanded_maximum
        ),
    )


def _make_persistence_image_transformer(
    config: PersistenceImageConfig,
    *,
    image_range: tuple[
        float,
        float,
        float,
        float,
    ]
    | None,
) -> PersistenceImage:
    if image_range is None:
        im_range = [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]

    else:
        im_range = [
            float(
                value
            )
            for value in image_range
        ]

    return PersistenceImage(
        bandwidth=config.bandwidth,
        weight=_persistence_weight,
        resolution=[
            config.resolution_x,
            config.resolution_y,
        ],
        im_range=im_range,
    )


def fit_persistence_image(
    training_diagrams: Iterable[ArrayLike],
    config: PersistenceImageConfig,
) -> FittedPersistenceImage:
    """
    Fit the persistence-image range from training diagrams only.

    Nested cross-validation must call this function separately on
    each inner-training domain and again on the selected outer-
    training domain. Validation and test diagrams must never enter
    this fit.

    If a training-only fitted birth or persistence range has zero
    width, this function expands that coordinate symmetrically by
    one Gaussian bandwidth and refits with the resulting fixed
    image range.
    """

    if not isinstance(
        config,
        PersistenceImageConfig,
    ):
        raise TypeError(
            "config must be a PersistenceImageConfig instance."
        )

    diagrams = _normalize_iterable(
        training_diagrams,
        description="training persistence diagrams",
    )

    if not diagrams:
        raise ValueError(
            "Persistence-image fitting requires at least one "
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
            "Persistence-image fitting requires at least one "
            "nonempty training diagram."
        )

    transformer = _make_persistence_image_transformer(
        config,
        image_range=None,
    )

    transformer.fit(
        list(
            nonempty_diagrams
        )
    )

    (
        birth_minimum,
        birth_maximum,
        persistence_minimum,
        persistence_maximum,
    ) = _fitted_image_state(
        transformer,
        config,
        require_positive_width=False,
    )

    (
        adjusted_birth_minimum,
        adjusted_birth_maximum,
    ) = _positive_width_range(
        birth_minimum,
        birth_maximum,
        bandwidth=config.bandwidth,
        description="persistence-image birth range",
    )

    (
        adjusted_persistence_minimum,
        adjusted_persistence_maximum,
    ) = _positive_width_range(
        persistence_minimum,
        persistence_maximum,
        bandwidth=config.bandwidth,
        description="persistence-image persistence range",
    )

    range_changed = (
        adjusted_birth_minimum
        !=
        birth_minimum
        or
        adjusted_birth_maximum
        !=
        birth_maximum
        or
        adjusted_persistence_minimum
        !=
        persistence_minimum
        or
        adjusted_persistence_maximum
        !=
        persistence_maximum
    )

    if range_changed:
        transformer = _make_persistence_image_transformer(
            config,
            image_range=(
                adjusted_birth_minimum,
                adjusted_birth_maximum,
                adjusted_persistence_minimum,
                adjusted_persistence_maximum,
            ),
        )

        transformer.fit(
            list(
                nonempty_diagrams
            )
        )

    (
        birth_minimum,
        birth_maximum,
        persistence_minimum,
        persistence_maximum,
    ) = _fitted_image_state(
        transformer,
        config,
        require_positive_width=True,
    )

    return FittedPersistenceImage(
        bandwidth=config.bandwidth,
        resolution_x=config.resolution_x,
        resolution_y=config.resolution_y,
        birth_minimum=birth_minimum,
        birth_maximum=birth_maximum,
        persistence_minimum=persistence_minimum,
        persistence_maximum=persistence_maximum,
        _transformer=transformer,
    )


def transform_persistence_images(
    fitted: FittedPersistenceImage,
    diagrams: Iterable[ArrayLike],
) -> FloatArray:
    if not isinstance(
        fitted,
        FittedPersistenceImage,
    ):
        raise TypeError(
            "fitted must be a FittedPersistenceImage instance."
        )

    normalized = _normalize_iterable(
        diagrams,
        description="persistence diagrams",
    )

    feature_count = fitted.feature_count

    diagram_count = len(
        normalized
    )

    if diagram_count == 0:
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
            diagram_count,
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
            raise PersistenceImageError(
                "GUDHI returned a persistence-image feature matrix "
                "with an unexpected shape."
            )

        if not bool(
            np.all(
                np.isfinite(
                    transformed
                )
            )
        ):
            raise PersistenceImageError(
                "GUDHI returned nonfinite persistence-image "
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


def transform_persistence_image(
    fitted: FittedPersistenceImage,
    diagram: ArrayLike,
) -> PersistenceImageFeatures:
    matrix = transform_persistence_images(
        fitted,
        (
            diagram,
        ),
    )

    return PersistenceImageFeatures(
        bandwidth=fitted.bandwidth,
        resolution_x=fitted.resolution_x,
        resolution_y=fitted.resolution_y,
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
            "Graph-level persistence-image features must form a "
            "two-dimensional matrix."
        )

    if matrix.shape[
        0
    ] == 0:
        raise ValueError(
            "Persistence-image batch aggregation requires at least "
            "one graph."
        )

    if matrix.shape[
        1
    ] != expected_columns:
        raise ValueError(
            "Graph-level persistence-image feature dimension "
            f"{matrix.shape[1]} does not match the expected "
            f"dimension {expected_columns}."
        )

    if not bool(
        np.all(
            np.isfinite(
                matrix
            )
        )
    ):
        raise ValueError(
            "Graph-level persistence-image features must be finite."
        )

    return matrix


def aggregate_persistence_images(
    graph_features: ArrayLike,
    *,
    bandwidth: float,
    resolution_x: int,
    resolution_y: int,
) -> BatchPersistenceImageFeatures:
    """
    Aggregate graph persistence images by coordinatewise mean.
    """

    parsed_bandwidth = _require_positive_scalar(
        bandwidth,
        description="persistence-image bandwidth",
    )

    parsed_resolution_x = _require_positive_integer(
        resolution_x,
        description="persistence-image x resolution",
    )

    parsed_resolution_y = _require_positive_integer(
        resolution_y,
        description="persistence-image y resolution",
    )

    graph_feature_count = persistence_image_dimension(
        parsed_resolution_x,
        parsed_resolution_y,
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

    if not bool(
        np.all(
            np.isfinite(
                values
            )
        )
    ):
        raise PersistenceImageError(
            "Persistence-image aggregation produced nonfinite "
            "features."
        )

    values.setflags(
        write=False
    )

    return BatchPersistenceImageFeatures(
        graph_count=int(
            matrix.shape[
                0
            ]
        ),
        bandwidth=parsed_bandwidth,
        resolution_x=parsed_resolution_x,
        resolution_y=parsed_resolution_y,
        values=values,
    )


__all__ = [
    "BatchPersistenceImageFeatures",
    "FittedPersistenceImage",
    "PERSISTENCE_IMAGE_SCHEMA_VERSION",
    "PersistenceImageConfig",
    "PersistenceImageError",
    "PersistenceImageFeatures",
    "aggregate_persistence_images",
    "batch_persistence_image_dimension",
    "batch_persistence_image_feature_names",
    "fit_persistence_image",
    "persistence_image_dimension",
    "persistence_image_feature_names",
    "transform_persistence_image",
    "transform_persistence_images",
]