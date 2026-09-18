from __future__ import annotations

import csv
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .configuration import (
    CHARACTER_COUNT,
    DATASETS,
    HARMONIC_FEATURE_COUNT,
)


FloatMatrix: TypeAlias = NDArray[np.float64]
FloatVector: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
LabelArray: TypeAlias = NDArray[np.int64]
EdgeArray: TypeAlias = NDArray[np.int64]
DiagramArray: TypeAlias = NDArray[np.float64]

GraphKey: TypeAlias = tuple[str, int]


class RealWorldDataError(
    RuntimeError
):
    pass


def _fail(
    message: str,
) -> RealWorldDataError:
    return RealWorldDataError(
        message
    )


@dataclass(
    frozen=True,
    slots=True,
)
class DatasetMetadata:
    name: str
    graph_count: int
    vertex_count: int
    edge_count: int
    class_count: int


@dataclass(
    frozen=True,
    slots=True,
)
class GraphRecord:
    dataset: str
    graph_index: int
    label: int
    vertex_count: int
    edge_count: int
    filtration_level_count: int
    persistence_atom_count: int

    @property
    def key(
        self,
    ) -> GraphKey:
        return (
            self.dataset,
            self.graph_index,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RealWorldDataset:
    metadata: DatasetMetadata
    graph_indices: IndexArray
    labels: LabelArray
    vertex_counts: IndexArray
    edges: tuple[
        EdgeArray,
        ...,
    ]
    persistence_diagrams: tuple[
        DiagramArray,
        ...,
    ]
    harmonic_features: FloatMatrix


@dataclass(
    frozen=True,
    slots=True,
)
class RealWorldData:
    root: Path
    tu_root: Path
    datasets: Mapping[
        str,
        RealWorldDataset,
    ]


def _require_finalized_file(
    path: Path,
) -> Path:
    if path.is_file():
        return path

    temporary = Path(
        str(path)
        + ".tmp"
    )

    if temporary.exists():
        raise _fail(
            "Raw C++ output is incomplete. "
            f"Finalized file '{path}' is absent while "
            f"working file '{temporary}' exists."
        )

    raise _fail(
        f"Required file is absent: '{path}'."
    )


def _iter_csv_rows(
    path: Path,
    expected_header: Sequence[str],
) -> Iterator[
    tuple[str, ...]
]:
    finalized = _require_finalized_file(
        path
    )

    try:
        with finalized.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            reader = csv.reader(
                stream
            )

            try:
                actual_header = tuple(
                    next(
                        reader
                    )
                )
            except StopIteration as error:
                raise _fail(
                    f"CSV file '{finalized}' is empty."
                ) from error

            expected = tuple(
                expected_header
            )

            if actual_header != expected:
                raise _fail(
                    f"CSV schema mismatch in '{finalized}'. "
                    f"Expected {expected!r}, found "
                    f"{actual_header!r}."
                )

            width = len(
                expected
            )

            for row_index, row in enumerate(
                reader,
                start=2,
            ):
                if len(row) != width:
                    raise _fail(
                        f"CSV file '{finalized}' row "
                        f"{row_index} has {len(row)} fields, "
                        f"expected {width}."
                    )

                yield tuple(
                    row
                )

    except (
        OSError,
        UnicodeError,
        csv.Error,
    ) as error:
        raise _fail(
            f"Could not read CSV file '{finalized}': "
            f"{error}"
        ) from error


def _parse_integer(
    value: str,
    *,
    description: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> int:
    try:
        parsed = int(
            value,
            10,
        )
    except ValueError as error:
        raise _fail(
            f"{description} must be an integer."
        ) from error

    if str(parsed) != value:
        raise _fail(
            f"{description} has noncanonical integer "
            f"representation {value!r}."
        )

    if (
        nonnegative
        and parsed < 0
    ):
        raise _fail(
            f"{description} must be nonnegative."
        )

    if (
        positive
        and parsed <= 0
    ):
        raise _fail(
            f"{description} must be positive."
        )

    return parsed


def _parse_real(
    value: str,
    *,
    description: str,
) -> float:
    try:
        parsed = float(
            value
        )
    except ValueError as error:
        raise _fail(
            f"{description} must be numeric."
        ) from error

    if not math.isfinite(
        parsed
    ):
        raise _fail(
            f"{description} must be finite."
        )

    if parsed == 0.0:
        return 0.0

    return parsed


def _harmonic_header(
) -> tuple[str, ...]:
    columns = [
        "dataset",
        "graph_index",
        "label",
    ]

    for character_index in range(
        CHARACTER_COUNT
    ):
        prefix = (
            f"character_{character_index}"
        )

        columns.extend(
            (
                f"{prefix}_real",
                f"{prefix}_imag",
            )
        )

    return tuple(
        columns
    )


def _load_dataset_metadata(
    root: Path,
) -> Mapping[
    str,
    DatasetMetadata,
]:
    path = root / "datasets.csv"

    header = (
        "dataset",
        "graph_count",
        "vertex_count",
        "edge_count",
        "class_count",
    )

    metadata: dict[
        str,
        DatasetMetadata,
    ] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        name = row[0]

        if name not in DATASETS:
            raise _fail(
                f"datasets.csv contains unsupported dataset "
                f"{name!r}."
            )

        if name in metadata:
            raise _fail(
                f"datasets.csv contains duplicate dataset "
                f"{name!r}."
            )

        metadata[
            name
        ] = DatasetMetadata(
            name=name,
            graph_count=_parse_integer(
                row[1],
                description=f"{name} graph count",
                positive=True,
            ),
            vertex_count=_parse_integer(
                row[2],
                description=f"{name} vertex count",
                nonnegative=True,
            ),
            edge_count=_parse_integer(
                row[3],
                description=f"{name} edge count",
                nonnegative=True,
            ),
            class_count=_parse_integer(
                row[4],
                description=f"{name} class count",
                positive=True,
            ),
        )

    if tuple(metadata) != DATASETS:
        raise _fail(
            "datasets.csv does not contain the expected datasets "
            "in canonical order."
        )

    return MappingProxyType(
        metadata
    )


def _load_graph_records(
    root: Path,
) -> Mapping[
    GraphKey,
    GraphRecord,
]:
    path = root / "graphs.csv"

    header = (
        "dataset",
        "graph_index",
        "label",
        "vertex_count",
        "edge_count",
        "filtration_level_count",
        "persistence_atom_count",
    )

    records: dict[
        GraphKey,
        GraphRecord,
    ] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        dataset = row[0]

        if dataset not in DATASETS:
            raise _fail(
                f"graphs.csv contains unsupported dataset "
                f"{dataset!r}."
            )

        record = GraphRecord(
            dataset=dataset,
            graph_index=_parse_integer(
                row[1],
                description="graph index",
                nonnegative=True,
            ),
            label=_parse_integer(
                row[2],
                description="graph label",
            ),
            vertex_count=_parse_integer(
                row[3],
                description="graph vertex count",
                positive=True,
            ),
            edge_count=_parse_integer(
                row[4],
                description="graph edge count",
                nonnegative=True,
            ),
            filtration_level_count=_parse_integer(
                row[5],
                description="filtration level count",
                nonnegative=True,
            ),
            persistence_atom_count=_parse_integer(
                row[6],
                description="persistence atom count",
                nonnegative=True,
            ),
        )

        if record.key in records:
            raise _fail(
                "graphs.csv contains duplicate graph key "
                f"{record.key!r}."
            )

        records[
            record.key
        ] = record

    return MappingProxyType(
        records
    )


def _load_persistence_diagrams(
    root: Path,
    graph_records: Mapping[
        GraphKey,
        GraphRecord,
    ],
) -> Mapping[
    GraphKey,
    DiagramArray,
]:
    path = root / "persistence_diagrams.csv"

    header = (
        "dataset",
        "graph_index",
        "atom_index",
        "birth",
        "death",
        "coefficient",
    )

    atoms: dict[
        GraphKey,
        list[
            tuple[
                int,
                float,
                float,
                int,
            ]
        ],
    ] = {
        key: []
        for key in graph_records
    }

    for row in _iter_csv_rows(
        path,
        header,
    ):
        dataset = row[0]

        graph_index = _parse_integer(
            row[1],
            description="persistence graph index",
            nonnegative=True,
        )

        atom_index = _parse_integer(
            row[2],
            description="persistence atom index",
            nonnegative=True,
        )

        key = (
            dataset,
            graph_index,
        )

        if key not in graph_records:
            raise _fail(
                "persistence_diagrams.csv contains graph key "
                f"{key!r} absent from graphs.csv."
            )

        birth = _parse_real(
            row[3],
            description=f"persistence birth for graph {key!r}",
        )

        death = _parse_real(
            row[4],
            description=f"persistence death for graph {key!r}",
        )

        if birth >= death:
            raise _fail(
                f"Persistence atom {atom_index} for graph {key!r} "
                "must satisfy birth < death."
            )

        coefficient_value = _parse_real(
            row[5],
            description=(
                f"persistence coefficient for graph {key!r}"
            ),
        )

        coefficient = int(
            coefficient_value
        )

        if (
            coefficient <= 0
            or float(coefficient) != coefficient_value
        ):
            raise _fail(
                f"Persistence coefficient for graph {key!r} must "
                "be a positive integral multiplicity."
            )

        atoms[
            key
        ].append(
            (
                atom_index,
                birth,
                death,
                coefficient,
            )
        )

    diagrams: dict[
        GraphKey,
        DiagramArray,
    ] = {}

    for key, graph_record in graph_records.items():
        graph_atoms = sorted(
            atoms[
                key
            ],
            key=lambda atom: atom[0],
        )

        if len(graph_atoms) != graph_record.persistence_atom_count:
            raise _fail(
                f"Graph {key!r} has {len(graph_atoms)} persistence "
                "atom rows, but graphs.csv declares "
                f"{graph_record.persistence_atom_count}."
            )

        expected_atom_indices = tuple(
            range(
                len(
                    graph_atoms
                )
            )
        )

        actual_atom_indices = tuple(
            atom[
                0
            ]
            for atom in graph_atoms
        )

        if actual_atom_indices != expected_atom_indices:
            raise _fail(
                f"Graph {key!r} persistence atom indices are not "
                "contiguous from zero."
            )

        rows: list[
            tuple[
                float,
                float,
            ]
        ] = []

        for (
            _,
            birth,
            death,
            coefficient,
        ) in graph_atoms:
            rows.extend(
                (
                    birth,
                    death,
                )
                for _ in range(
                    coefficient
                )
            )

        diagram = np.asarray(
            rows,
            dtype=np.float64,
        ).reshape(
            -1,
            2,
        )

        diagram.setflags(
            write=False
        )

        diagrams[
            key
        ] = diagram

    return MappingProxyType(
        diagrams
    )


def _load_harmonic_features(
    root: Path,
    graph_records: Mapping[
        GraphKey,
        GraphRecord,
    ],
) -> Mapping[
    GraphKey,
    FloatVector,
]:
    path = root / "harmonic_features.csv"

    features: dict[
        GraphKey,
        FloatVector,
    ] = {}

    for row in _iter_csv_rows(
        path,
        _harmonic_header(),
    ):
        dataset = row[0]

        graph_index = _parse_integer(
            row[1],
            description="harmonic graph index",
            nonnegative=True,
        )

        label = _parse_integer(
            row[2],
            description="harmonic graph label",
        )

        key: GraphKey = (
            dataset,
            graph_index,
        )

        try:
            graph_record = graph_records[
                key
            ]
        except KeyError as error:
            raise _fail(
                "harmonic_features.csv contains graph key "
                f"{key!r} absent from graphs.csv."
            ) from error

        if label != graph_record.label:
            raise _fail(
                "harmonic_features.csv label disagrees with "
                f"graphs.csv for graph {key!r}."
            )

        if key in features:
            raise _fail(
                "harmonic_features.csv contains duplicate graph "
                f"key {key!r}."
            )

        values = np.asarray(
            tuple(
                _parse_real(
                    value,
                    description=(
                        f"harmonic coordinate for graph {key!r}"
                    ),
                )
                for value in row[3:]
            ),
            dtype=np.float64,
        )

        if values.size != HARMONIC_FEATURE_COUNT:
            raise _fail(
                f"Graph {key!r} has {values.size} harmonic "
                f"coordinates, expected "
                f"{HARMONIC_FEATURE_COUNT}."
            )

        values.setflags(
            write=False
        )

        features[
            key
        ] = values

    if set(features) != set(graph_records):
        missing = tuple(
            sorted(
                set(graph_records)
                -
                set(features)
            )
        )

        unexpected = tuple(
            sorted(
                set(features)
                -
                set(graph_records)
            )
        )

        raise _fail(
            "Harmonic-feature graph domain disagrees with "
            "graphs.csv. "
            f"Missing={missing[:5]!r}, "
            f"unexpected={unexpected[:5]!r}."
        )

    return MappingProxyType(
        features
    )


def _read_text_lines(
    path: Path,
) -> tuple[str, ...]:
    finalized = _require_finalized_file(
        path
    )

    try:
        with finalized.open(
            "r",
            encoding="utf-8",
        ) as stream:
            return tuple(
                line.strip()
                for line in stream
                if line.strip()
            )
    except (
        OSError,
        UnicodeError,
    ) as error:
        raise _fail(
            f"Could not read TU file '{finalized}': {error}"
        ) from error


def _load_tu_topology(
    tu_root: Path,
    dataset_name: str,
    records: Sequence[
        GraphRecord
    ],
) -> tuple[
    EdgeArray,
    ...,
]:
    dataset_root = (
        tu_root
        /
        dataset_name
    )

    indicator_path = (
        dataset_root
        /
        f"{dataset_name}_graph_indicator.txt"
    )

    adjacency_path = (
        dataset_root
        /
        f"{dataset_name}_A.txt"
    )

    indicator_lines = _read_text_lines(
        indicator_path
    )

    graph_count = len(
        records
    )

    indicators: list[int] = []

    local_vertex_indices: list[int] = []

    vertex_counts = [
        0
        for _ in range(
            graph_count
        )
    ]

    for global_position, line in enumerate(
        indicator_lines,
        start=1,
    ):
        graph_identifier = _parse_integer(
            line,
            description=(
                f"{dataset_name} graph indicator at global "
                f"vertex {global_position}"
            ),
            positive=True,
        )

        graph_index = (
            graph_identifier
            -
            1
        )

        if (
            graph_index < 0
            or graph_index >= graph_count
        ):
            raise _fail(
                f"{dataset_name} graph indicator at global vertex "
                f"{global_position} lies outside the graph domain."
            )

        indicators.append(
            graph_index
        )

        local_vertex_indices.append(
            vertex_counts[
                graph_index
            ]
        )

        vertex_counts[
            graph_index
        ] += 1

    for graph_index, record in enumerate(
        records
    ):
        if vertex_counts[
            graph_index
        ] != record.vertex_count:
            raise _fail(
                f"{dataset_name} graph {graph_index} has "
                f"{vertex_counts[graph_index]} TU vertices, but "
                f"graphs.csv declares {record.vertex_count}."
            )

    edge_sets: list[
        set[
            tuple[
                int,
                int,
            ]
        ]
    ] = [
        set()
        for _ in range(
            graph_count
        )
    ]

    adjacency_lines = _read_text_lines(
        adjacency_path
    )

    global_vertex_count = len(
        indicators
    )

    for row_index, line in enumerate(
        adjacency_lines,
        start=1,
    ):
        fields = tuple(
            field.strip()
            for field in line.split(
                ","
            )
        )

        if len(fields) != 2:
            raise _fail(
                f"{dataset_name} adjacency row {row_index} must "
                "contain exactly two comma-separated endpoints."
            )

        global_u = _parse_integer(
            fields[
                0
            ],
            description=(
                f"{dataset_name} adjacency row {row_index} "
                "first endpoint"
            ),
            positive=True,
        )

        global_v = _parse_integer(
            fields[
                1
            ],
            description=(
                f"{dataset_name} adjacency row {row_index} "
                "second endpoint"
            ),
            positive=True,
        )

        if (
            global_u > global_vertex_count
            or global_v > global_vertex_count
        ):
            raise _fail(
                f"{dataset_name} adjacency row {row_index} "
                "contains an endpoint outside the vertex domain."
            )

        u_position = (
            global_u
            -
            1
        )

        v_position = (
            global_v
            -
            1
        )

        graph_u = indicators[
            u_position
        ]

        graph_v = indicators[
            v_position
        ]

        if graph_u != graph_v:
            raise _fail(
                f"{dataset_name} adjacency row {row_index} "
                "crosses graph boundaries."
            )

        local_u = local_vertex_indices[
            u_position
        ]

        local_v = local_vertex_indices[
            v_position
        ]

        if local_u == local_v:
            continue

        edge_sets[
            graph_u
        ].add(
            (
                min(
                    local_u,
                    local_v,
                ),
                max(
                    local_u,
                    local_v,
                ),
            )
        )

    edges: list[
        EdgeArray
    ] = []

    for graph_index, (
        record,
        edge_set,
    ) in enumerate(
        zip(
            records,
            edge_sets,
            strict=True,
        )
    ):
        if len(edge_set) != record.edge_count:
            raise _fail(
                f"{dataset_name} graph {graph_index} has "
                f"{len(edge_set)} unique TU edges, but graphs.csv "
                f"declares {record.edge_count}."
            )

        edge_array = np.asarray(
            tuple(
                sorted(
                    edge_set
                )
            ),
            dtype=np.int64,
        ).reshape(
            -1,
            2,
        )

        edge_array.setflags(
            write=False
        )

        edges.append(
            edge_array
        )

    return tuple(
        edges
    )


def _assemble_dataset(
    metadata: DatasetMetadata,
    graph_records: Mapping[
        GraphKey,
        GraphRecord,
    ],
    persistence_diagrams: Mapping[
        GraphKey,
        DiagramArray,
    ],
    harmonic_features: Mapping[
        GraphKey,
        FloatVector,
    ],
    *,
    tu_root: Path,
) -> RealWorldDataset:
    records = tuple(
        sorted(
            (
                record
                for record in graph_records.values()
                if record.dataset == metadata.name
            ),
            key=lambda record: record.graph_index,
        )
    )

    if len(records) != metadata.graph_count:
        raise _fail(
            f"{metadata.name} has {len(records)} graph rows, "
            f"but datasets.csv declares "
            f"{metadata.graph_count}."
        )

    observed_indices = tuple(
        record.graph_index
        for record in records
    )

    expected_indices = tuple(
        range(
            metadata.graph_count
        )
    )

    if observed_indices != expected_indices:
        raise _fail(
            f"{metadata.name} graph indices are not contiguous "
            "from zero."
        )

    total_vertices = sum(
        record.vertex_count
        for record in records
    )

    total_edges = sum(
        record.edge_count
        for record in records
    )

    labels = tuple(
        record.label
        for record in records
    )

    if total_vertices != metadata.vertex_count:
        raise _fail(
            f"{metadata.name} vertex total disagrees with "
            "datasets.csv."
        )

    if total_edges != metadata.edge_count:
        raise _fail(
            f"{metadata.name} edge total disagrees with "
            "datasets.csv."
        )

    if len(set(labels)) != metadata.class_count:
        raise _fail(
            f"{metadata.name} class count disagrees with "
            "datasets.csv."
        )

    graph_indices = np.asarray(
        observed_indices,
        dtype=np.int64,
    )

    label_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    vertex_counts = np.asarray(
        tuple(
            record.vertex_count
            for record in records
        ),
        dtype=np.int64,
    )

    feature_matrix = np.vstack(
        tuple(
            harmonic_features[
                (
                    metadata.name,
                    record.graph_index,
                )
            ]
            for record in records
        )
    ).astype(
        np.float64,
        copy=False,
    )

    if feature_matrix.shape != (
        metadata.graph_count,
        HARMONIC_FEATURE_COUNT,
    ):
        raise _fail(
            f"{metadata.name} harmonic feature matrix has shape "
            f"{feature_matrix.shape!r}, expected "
            f"({metadata.graph_count}, "
            f"{HARMONIC_FEATURE_COUNT})."
        )

    diagrams = tuple(
        persistence_diagrams[
            (
                metadata.name,
                record.graph_index,
            )
        ]
        for record in records
    )

    edges = _load_tu_topology(
        tu_root,
        metadata.name,
        records,
    )

    graph_indices.setflags(
        write=False
    )

    label_array.setflags(
        write=False
    )

    vertex_counts.setflags(
        write=False
    )

    feature_matrix.setflags(
        write=False
    )

    return RealWorldDataset(
        metadata=metadata,
        graph_indices=graph_indices,
        labels=label_array,
        vertex_counts=vertex_counts,
        edges=edges,
        persistence_diagrams=diagrams,
        harmonic_features=feature_matrix,
    )


def load_real_world_data(
    root: str | Path,
    tu_root: str | Path,
) -> RealWorldData:
    local_root = Path(
        root
    ).expanduser().resolve()

    local_tu_root = Path(
        tu_root
    ).expanduser().resolve()

    if (
        not local_root.exists()
        or not local_root.is_dir()
    ):
        raise _fail(
            "Real-world raw-data root does not exist: "
            f"'{local_root}'."
        )

    if (
        not local_tu_root.exists()
        or not local_tu_root.is_dir()
    ):
        raise _fail(
            "TU dataset root does not exist: "
            f"'{local_tu_root}'."
        )

    metadata = _load_dataset_metadata(
        local_root
    )

    graph_records = _load_graph_records(
        local_root
    )

    persistence_diagrams = (
        _load_persistence_diagrams(
            local_root,
            graph_records,
        )
    )

    harmonic_features = (
        _load_harmonic_features(
            local_root,
            graph_records,
        )
    )

    datasets = {
        name: _assemble_dataset(
            metadata[
                name
            ],
            graph_records,
            persistence_diagrams,
            harmonic_features,
            tu_root=local_tu_root,
        )
        for name in DATASETS
    }

    return RealWorldData(
        root=local_root,
        tu_root=local_tu_root,
        datasets=MappingProxyType(
            datasets
        ),
    )


__all__ = [
    "DatasetMetadata",
    "DiagramArray",
    "EdgeArray",
    "GraphRecord",
    "RealWorldData",
    "RealWorldDataError",
    "RealWorldDataset",
    "load_real_world_data",
]