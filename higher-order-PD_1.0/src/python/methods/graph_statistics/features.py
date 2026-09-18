from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
EdgeArray: TypeAlias = NDArray[np.int64]

GRAPH_STATISTICS_SCHEMA_VERSION: Final[int] = 2

GRAPH_STATISTIC_NAMES: Final[tuple[str, ...]] = (
    "density",
    "mean_degree",
    "average_clustering",
)

GRAPH_STATISTIC_COUNT: Final[int] = len(
    GRAPH_STATISTIC_NAMES
)

BATCH_GRAPH_STATISTIC_NAMES: Final[tuple[str, ...]] = tuple(
    f"mean_{name}"
    for name in GRAPH_STATISTIC_NAMES
)

BATCH_GRAPH_STATISTIC_COUNT: Final[int] = len(
    BATCH_GRAPH_STATISTIC_NAMES
)


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


def _require_positive_vertex_count(
    value: int,
) -> int:
    parsed = _require_integer(
        value,
        description="vertex count",
    )

    if parsed <= 0:
        raise ValueError(
            "Vertex count must be positive."
        )

    return parsed


def _as_edges(
    edges: ArrayLike,
    *,
    vertex_count: int,
) -> EdgeArray:
    raw = np.asarray(
        edges
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
            dtype=np.int64,
        )

    if raw.ndim != 2:
        raise ValueError(
            "Graph edges must form a two-dimensional array."
        )

    if raw.shape[1] != 2:
        raise ValueError(
            "Graph edges must have shape (edge_count, 2)."
        )

    if np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError(
            "Graph edge endpoints must be integers, not booleans."
        )

    if not np.issubdtype(
        raw.dtype,
        np.integer,
    ):
        raise TypeError(
            "Graph edge endpoints must have an integer dtype."
        )

    converted = np.asarray(
        raw,
        dtype=np.int64,
    )

    if not np.array_equal(
        raw,
        converted,
    ):
        raise ValueError(
            "Graph edge endpoints cannot be represented exactly as "
            "signed 64-bit integers."
        )

    if converted.shape[0] == 0:
        result = np.empty(
            (
                0,
                2,
            ),
            dtype=np.int64,
        )

        result.setflags(
            write=False
        )

        return result

    first = converted[
        :,
        0
    ]

    second = converted[
        :,
        1
    ]

    if bool(
        np.any(
            first < 0
        )
        or np.any(
            second < 0
        )
    ):
        raise ValueError(
            "Graph edge endpoints must be nonnegative."
        )

    if bool(
        np.any(
            first >= vertex_count
        )
        or np.any(
            second >= vertex_count
        )
    ):
        raise ValueError(
            "Graph edge endpoint lies outside the vertex range."
        )

    if bool(
        np.any(
            first == second
        )
    ):
        raise ValueError(
            "Graph must not contain self-loops."
        )

    canonical = np.empty(
        converted.shape,
        dtype=np.int64,
    )

    canonical[
        :,
        0
    ] = np.minimum(
        first,
        second,
    )

    canonical[
        :,
        1
    ] = np.maximum(
        first,
        second,
    )

    order = np.lexsort(
        (
            canonical[
                :,
                1
            ],
            canonical[
                :,
                0
            ],
        )
    )

    canonical = canonical[
        order
    ]

    if canonical.shape[0] > 1:
        duplicates = np.all(
            canonical[
                1:
            ]
            ==
            canonical[
                :-1
            ],
            axis=1,
        )

        if bool(
            np.any(
                duplicates
            )
        ):
            raise ValueError(
                "Graph contains duplicate undirected edges."
            )

    result = np.array(
        canonical,
        dtype=np.int64,
        order="C",
        copy=True,
    )

    result.setflags(
        write=False
    )

    return result


def _as_graph_feature_matrix(
    values: ArrayLike,
) -> FloatArray:
    matrix = np.asarray(
        values,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Graph-statistic feature matrix must be two-dimensional."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Graph-statistic feature matrix must contain at least "
            "one graph."
        )

    if matrix.shape[1] != GRAPH_STATISTIC_COUNT:
        raise ValueError(
            "Graph-statistic feature matrix has an incompatible "
            f"feature dimension {matrix.shape[1]}; expected "
            f"{GRAPH_STATISTIC_COUNT}."
        )

    result = np.array(
        matrix,
        dtype=np.float64,
        order="C",
        copy=True,
    )

    result.setflags(
        write=False
    )

    return result


@dataclass(
    frozen=True,
    slots=True,
)
class GraphStatisticFeatures:
    values: FloatArray


@dataclass(
    frozen=True,
    slots=True,
)
class BatchGraphStatisticFeatures:
    graph_count: int
    values: FloatArray


def compute_graph_statistics(
    vertex_count: int,
    edges: ArrayLike,
) -> GraphStatisticFeatures:
    parsed_vertex_count = _require_positive_vertex_count(
        vertex_count
    )

    canonical_edges = _as_edges(
        edges,
        vertex_count=parsed_vertex_count,
    )

    edge_count = int(
        canonical_edges.shape[
            0
        ]
    )

    graph = nx.Graph()

    graph.add_nodes_from(
        range(
            parsed_vertex_count
        )
    )

    if edge_count != 0:
        graph.add_edges_from(
            (
                int(
                    edge[
                        0
                    ]
                ),
                int(
                    edge[
                        1
                    ]
                ),
            )
            for edge in canonical_edges
        )

    if edge_count == 0:
        degrees = np.zeros(
            parsed_vertex_count,
            dtype=np.int64,
        )
    else:
        degrees = np.bincount(
            canonical_edges.reshape(
                -1
            ),
            minlength=parsed_vertex_count,
        ).astype(
            np.int64,
            copy=False,
        )

    density = (
        0.0
        if parsed_vertex_count == 1
        else (
            2.0
            *
            edge_count
            /
            (
                parsed_vertex_count
                *
                (
                    parsed_vertex_count
                    -
                    1
                )
            )
        )
    )

    mean_degree = float(
        np.mean(
            degrees
        )
    )

    average_clustering = float(
        nx.average_clustering(
            graph
        )
    )

    values = np.asarray(
        (
            density,
            mean_degree,
            average_clustering,
        ),
        dtype=np.float64,
    )

    values.setflags(
        write=False
    )

    return GraphStatisticFeatures(
        values=values
    )


def aggregate_graph_statistics(
    graph_features: ArrayLike,
) -> BatchGraphStatisticFeatures:
    matrix = _as_graph_feature_matrix(
        graph_features
    )

    graph_count = int(
        matrix.shape[
            0
        ]
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

    return BatchGraphStatisticFeatures(
        graph_count=graph_count,
        values=values,
    )


__all__ = [
    "BATCH_GRAPH_STATISTIC_COUNT",
    "BATCH_GRAPH_STATISTIC_NAMES",
    "BatchGraphStatisticFeatures",
    "GRAPH_STATISTIC_COUNT",
    "GRAPH_STATISTIC_NAMES",
    "GRAPH_STATISTICS_SCHEMA_VERSION",
    "GraphStatisticFeatures",
    "aggregate_graph_statistics",
    "compute_graph_statistics",
]