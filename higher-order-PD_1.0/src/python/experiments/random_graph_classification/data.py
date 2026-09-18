from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray


METADATA_SCHEMA_VERSION: Final[int] = 2
OUTPUT_SCHEMA_VERSION: Final[int] = 3
EXPERIMENT_NAME: Final[str] = "random_graph_classification"

_UINT64_MAX: Final[int] = (1 << 64) - 1


FloatVector: TypeAlias = NDArray[np.float64]

BenchmarkSampleKey: TypeAlias = tuple[int, int, int]
BenchmarkGraphKey: TypeAlias = tuple[int, int, int, int]

AblationKey: TypeAlias = tuple[
    int,
    int,
    int,
    str,
    int,
]

BatchSizeKey: TypeAlias = tuple[
    int,
    int,
    int,
    int,
]

RffKey: TypeAlias = tuple[
    int,
    int,
    int,
    int,
    int,
]

RobustnessConditionKey: TypeAlias = tuple[
    int,
    int,
    int,
    int,
    int,
]

RobustnessFeatureKey: TypeAlias = tuple[
    int,
    int,
    int,
    int,
    int,
]

RobustnessGraphKey: TypeAlias = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
]

ScalabilityKey: TypeAlias = int


class RandomGraphClassificationDataError(
    RuntimeError
):
    pass


def _fail(
    message: str,
) -> RandomGraphClassificationDataError:
    return RandomGraphClassificationDataError(
        message
    )


@dataclass(
    frozen=True,
    slots=True,
)
class BuildMetadata:
    created_utc: str
    compiler: str
    compiler_version: str
    build_type: str
    operating_system: str
    executable: str


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkGenerationConfig:
    vertex_count: int
    edge_count: int
    batches_per_model: int
    graphs_per_batch: int


@dataclass(
    frozen=True,
    slots=True,
)
class RandomFeatureGenerationConfig:
    baseline_character_count: int
    replicate_count: int
    character_counts: tuple[int, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessDesignCondition:
    perturbation: str
    severity: float


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessGenerationConfig:
    realization_count: int
    conditions: tuple[
        RobustnessDesignCondition,
        ...,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class ScalabilityGenerationConfig:
    graphs_per_batch: tuple[int, ...]
    input_support_sizes: tuple[int, ...]
    character_counts: tuple[int, ...]
    repeat_count: int


@dataclass(
    frozen=True,
    slots=True,
)
class GenerationConfig:
    aggregation_order: int
    highest_explicit_order: int
    benchmark: BenchmarkGenerationConfig
    random_features: RandomFeatureGenerationConfig
    robustness: RobustnessGenerationConfig
    batch_size_graphs_per_batch: tuple[int, ...]
    scalability: ScalabilityGenerationConfig
    cross_observation_ablation_realizations: int
    canonical_root_seed: int
    root_seed_override: int | None
    effective_root_seed: int


@dataclass(
    frozen=True,
    slots=True,
)
class ManifestMetadata:
    created_utc: str
    task_count: int
    studies: tuple[str, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class ModelMetadata:
    model_index: int
    stable_id: str
    display_name: str
    variant: str
    vertex_count: int
    edge_count: int
    parameter_condition_count: int


@dataclass(
    frozen=True,
    slots=True,
)
class ModelParameter:
    model_index: int
    model_id: str
    parameter_condition_index: int
    parameter: str
    value: str


@dataclass(
    frozen=True,
    slots=True,
)
class CommonMetadata:
    root: Path
    generation: GenerationConfig
    build: BuildMetadata
    manifest: ManifestMetadata
    models: tuple[
        ModelMetadata,
        ...,
    ]
    model_by_index: Mapping[
        int,
        ModelMetadata,
    ]
    model_parameters: tuple[
        ModelParameter,
        ...,
    ]
    parameter_condition_indices: Mapping[
        int,
        tuple[int, ...],
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkBatch:
    model_index: int
    model_id: str
    model_name: str
    model_variant: str
    parameter_condition_index: int
    batch_index: int
    graphs_per_batch: int
    latent_seed: int
    character_seed: int

    @property
    def key(
        self,
    ) -> BenchmarkSampleKey:
        return (
            self.model_index,
            self.parameter_condition_index,
            self.batch_index,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class SerializedEdge:
    u: int
    v: int
    filtration_step: int
    filtration_time: float


@dataclass(
    frozen=True,
    slots=True,
)
class SerializedGraph:
    model_index: int
    model_id: str
    parameter_condition_index: int
    batch_index: int
    graph_index: int
    latent_seed: int
    sampling_seed: int
    vertex_count: int
    edge_count: int
    edges: tuple[
        SerializedEdge,
        ...,
    ]

    @property
    def key(
        self,
    ) -> BenchmarkGraphKey:
        return (
            self.model_index,
            self.parameter_condition_index,
            self.batch_index,
            self.graph_index,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class GraphFieldRecord:
    model_index: int
    model_id: str
    parameter_condition_index: int
    batch_index: int
    graph_index: int
    field_name: str
    value: float


@dataclass(
    frozen=True,
    slots=True,
)
class VertexFieldRecord:
    model_index: int
    model_id: str
    parameter_condition_index: int
    batch_index: int
    graph_index: int
    vertex_index: int
    field_name: str
    value: float


@dataclass(
    frozen=True,
    slots=True,
)
class EdgeFieldRecord:
    model_index: int
    model_id: str
    parameter_condition_index: int
    batch_index: int
    graph_index: int
    edge_index: int
    u: int
    v: int
    field_name: str
    value: float


@dataclass(
    frozen=True,
    slots=True,
)
class PersistenceAtom:
    birth: float
    death: float
    coefficient: float


@dataclass(
    frozen=True,
    slots=True,
)
class PersistenceDiagram:
    atoms: tuple[
        PersistenceAtom,
        ...,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class HarmonicFeature:
    key: tuple[Any, ...]
    character_seed: int | None
    values: FloatVector


@dataclass(
    frozen=True,
    slots=True,
)
class AblationCondition:
    model_index: int
    model_id: str
    parameter_condition_index: int
    batch_index: int
    ablation_kind: str
    realization_index: int
    graphs_per_batch: int
    aggregation_order: int
    character_count: int
    character_seed: int
    ablation_seed: int | None

    @property
    def key(
        self,
    ) -> AblationKey:
        return (
            self.model_index,
            self.parameter_condition_index,
            self.batch_index,
            self.ablation_kind,
            self.realization_index,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class BatchSizeCondition:
    model_index: int
    model_id: str
    parameter_condition_index: int
    batch_index: int
    maximum_graphs_per_batch: int
    character_seed: int
    condition_count: int

    @property
    def sample_key(
        self,
    ) -> BenchmarkSampleKey:
        return (
            self.model_index,
            self.parameter_condition_index,
            self.batch_index,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RffCondition:
    model_index: int
    model_id: str
    parameter_condition_index: int
    batch_index: int
    replicate_index: int
    character_seed: int
    condition_count: int

    @property
    def key(
        self,
    ) -> tuple[
        int,
        int,
        int,
        int,
    ]:
        return (
            self.model_index,
            self.parameter_condition_index,
            self.batch_index,
            self.replicate_index,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessCondition:
    model_index: int
    model_id: str
    parameter_condition_index: int
    batch_index: int
    condition_index: int
    perturbation: str
    severity: float
    realization_index: int
    graphs_per_batch: int
    character_seed: int

    @property
    def key(
        self,
    ) -> RobustnessConditionKey:
        return (
            self.model_index,
            self.parameter_condition_index,
            self.batch_index,
            self.condition_index,
            self.realization_index,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessGraph:
    model_index: int
    model_id: str
    parameter_condition_index: int
    batch_index: int
    graph_index: int
    condition_index: int
    perturbation: str
    severity: float
    realization_index: int
    perturbation_seed: int
    vertex_count: int
    edge_count: int
    edges: tuple[
        SerializedEdge,
        ...,
    ]

    @property
    def key(
        self,
    ) -> RobustnessGraphKey:
        return (
            self.model_index,
            self.parameter_condition_index,
            self.batch_index,
            self.graph_index,
            self.condition_index,
            self.realization_index,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessGraphDiagram:
    graph: RobustnessGraph
    diagram: PersistenceDiagram


@dataclass(
    frozen=True,
    slots=True,
)
class ScalabilityCondition:
    scalability_condition_index: int
    repeat_index: int
    graphs_per_batch: int
    input_support_size: int
    character_count: int
    sample_seed: int
    character_seed: int


@dataclass(
    frozen=True,
    slots=True,
)
class ScalabilityMeasurement:
    condition: ScalabilityCondition
    representation_seconds: float


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkGraphCollection:
    path: Path
    model_by_index: Mapping[
        int,
        ModelMetadata,
    ]

    def __iter__(
        self,
    ) -> Iterator[SerializedGraph]:
        return _iter_benchmark_graph_file(
            self.path,
            self.model_by_index,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkDiagramCollection:
    graph_collection: BenchmarkGraphCollection
    path: Path

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            BenchmarkGraphKey,
            PersistenceDiagram,
        ]
    ]:
        return _iter_benchmark_diagrams(
            self.graph_collection,
            self.path,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class GraphFieldCollection:
    path: Path

    def __iter__(
        self,
    ) -> Iterator[GraphFieldRecord]:
        return _iter_graph_fields(
            self.path
        )


@dataclass(
    frozen=True,
    slots=True,
)
class VertexFieldCollection:
    path: Path

    def __iter__(
        self,
    ) -> Iterator[VertexFieldRecord]:
        return _iter_vertex_fields(
            self.path
        )


@dataclass(
    frozen=True,
    slots=True,
)
class EdgeFieldCollection:
    path: Path

    def __iter__(
        self,
    ) -> Iterator[EdgeFieldRecord]:
        return _iter_edge_fields(
            self.path
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessGraphCollection:
    path: Path
    model_by_index: Mapping[
        int,
        ModelMetadata,
    ]
    conditions: Mapping[
        RobustnessConditionKey,
        RobustnessCondition,
    ]

    def __iter__(
        self,
    ) -> Iterator[RobustnessGraph]:
        return _iter_robustness_graph_file(
            self.path,
            self.model_by_index,
            self.conditions,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessDiagramCollection:
    graph_collection: RobustnessGraphCollection
    path: Path
    conditions: Mapping[
        RobustnessConditionKey,
        RobustnessCondition,
    ]

    def __iter__(
        self,
    ) -> Iterator[
        RobustnessGraphDiagram
    ]:
        return _iter_robustness_diagrams(
            self.graph_collection,
            self.path,
            self.conditions,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkCatalog:
    batches: tuple[
        BenchmarkBatch,
        ...,
    ]
    batch_by_key: Mapping[
        BenchmarkSampleKey,
        BenchmarkBatch,
    ]
    graphs: BenchmarkGraphCollection
    graph_fields: GraphFieldCollection
    vertex_fields: VertexFieldCollection
    edge_fields: EdgeFieldCollection
    diagrams: BenchmarkDiagramCollection
    harmonic_features: Mapping[
        BenchmarkSampleKey,
        HarmonicFeature,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class AblationCatalog:
    conditions: Mapping[
        AblationKey,
        AblationCondition,
    ]
    harmonic_features: Mapping[
        AblationKey,
        HarmonicFeature,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class BatchSizeCatalog:
    conditions: Mapping[
        BenchmarkSampleKey,
        BatchSizeCondition,
    ]
    harmonic_features: Mapping[
        BatchSizeKey,
        HarmonicFeature,
    ]
    graph_counts: tuple[int, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class RffCatalog:
    conditions: Mapping[
        tuple[int, int, int, int],
        RffCondition,
    ]
    harmonic_features: Mapping[
        RffKey,
        HarmonicFeature,
    ]
    character_counts: tuple[int, ...]
    replicate_indices: tuple[int, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class RobustnessCatalog:
    conditions: Mapping[
        RobustnessConditionKey,
        RobustnessCondition,
    ]
    harmonic_features: Mapping[
        RobustnessFeatureKey,
        HarmonicFeature,
    ]
    graphs: RobustnessGraphCollection
    diagrams: RobustnessDiagramCollection


@dataclass(
    frozen=True,
    slots=True,
)
class ScalabilityCatalog:
    measurements: Mapping[
        ScalabilityKey,
        ScalabilityMeasurement,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class RandomGraphClassificationData:
    common: CommonMetadata
    benchmark: BenchmarkCatalog
    ablation: AblationCatalog
    batch_size: BatchSizeCatalog
    rff: RffCatalog
    robustness: RobustnessCatalog
    scalability: ScalabilityCatalog


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            value = json.load(
                stream
            )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise _fail(
            f"Could not read JSON file {path!s}: {error}"
        ) from error

    if not isinstance(
        value,
        dict,
    ):
        raise _fail(
            f"JSON file {path!s} must contain an object."
        )

    return value


def _require_mapping(
    value: Any,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise _fail(
            f"{description} must be an object."
        )

    return value


def _require_list(
    value: Any,
    description: str,
) -> list[Any]:
    if not isinstance(
        value,
        list,
    ):
        raise _fail(
            f"{description} must be an array."
        )

    return value


def _require_string(
    value: Any,
    description: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise _fail(
            f"{description} must be a string."
        )

    return value


def _require_int(
    value: Any,
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
        raise _fail(
            f"{description} must be an integer."
        )

    return value


def _require_positive_int(
    value: Any,
    description: str,
) -> int:
    parsed = _require_int(
        value,
        description,
    )

    if parsed <= 0:
        raise _fail(
            f"{description} must be positive."
        )

    return parsed


def _require_uint64(
    value: Any,
    description: str,
) -> int:
    parsed = _require_int(
        value,
        description,
    )

    if (
        parsed < 0
        or parsed > _UINT64_MAX
    ):
        raise _fail(
            f"{description} must lie in the unsigned 64-bit range."
        )

    return parsed


def _require_optional_uint64(
    value: Any,
    description: str,
) -> int | None:
    if value is None:
        return None

    return _require_uint64(
        value,
        description,
    )


def _require_float(
    value: Any,
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
        raise _fail(
            f"{description} must be numeric."
        )

    parsed = float(
        value
    )

    if not math.isfinite(
        parsed
    ):
        raise _fail(
            f"{description} must be finite."
        )

    if parsed == 0.0:
        return 0.0

    return parsed


def _parse_int(
    value: str,
    description: str,
) -> int:
    try:
        return int(
            value
        )
    except ValueError as error:
        raise _fail(
            f"{description} must be an integer."
        ) from error


def _parse_positive_int(
    value: str,
    description: str,
) -> int:
    parsed = _parse_int(
        value,
        description,
    )

    if parsed <= 0:
        raise _fail(
            f"{description} must be positive."
        )

    return parsed


def _parse_uint64(
    value: str,
    description: str,
) -> int:
    parsed = _parse_int(
        value,
        description,
    )

    if (
        parsed < 0
        or parsed > _UINT64_MAX
    ):
        raise _fail(
            f"{description} must lie in the unsigned 64-bit range."
        )

    return parsed


def _parse_optional_uint64(
    value: str,
    description: str,
) -> int | None:
    if value == "":
        return None

    return _parse_uint64(
        value,
        description,
    )


def _parse_float(
    value: str,
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


def _require_finalized_file(
    path: Path,
) -> Path:
    if path.is_file():
        return path

    temporary = Path(
        str(
            path
        )
        + ".tmp"
    )

    if temporary.exists():
        raise _fail(
            f"Raw C++ output is incomplete: finalized file "
            f"{path!s} is absent while working file "
            f"{temporary!s} exists."
        )

    raise _fail(
        f"Required raw C++ output file is absent: {path!s}."
    )


def _iter_csv_rows(
    path: Path,
    header: Sequence[str],
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
                    f"CSV file {finalized!s} is empty."
                ) from error

            expected_header = tuple(
                header
            )

            if actual_header != expected_header:
                raise _fail(
                    f"CSV schema mismatch in {finalized!s}."
                )

            width = len(
                expected_header
            )

            for row_index, row in enumerate(
                reader,
                start=2,
            ):
                if len(
                    row
                ) != width:
                    raise _fail(
                        f"CSV file {finalized!s} row "
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
            f"Could not read CSV file {finalized!s}: {error}"
        ) from error


def _require_output_schema(
    row: Sequence[str],
    path: Path,
) -> None:
    if (
        _parse_int(
            row[0],
            f"{path.name} schema version",
        )
        != OUTPUT_SCHEMA_VERSION
    ):
        raise _fail(
            f"CSV file {path!s} uses an incompatible "
            "output schema version."
        )


def _freeze_vector(
    values: Sequence[float],
) -> FloatVector:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array.setflags(
        write=False
    )

    return array


def _configuration_from_json(
    configuration: Mapping[str, Any],
) -> tuple[
    GenerationConfig,
    BuildMetadata,
]:
    if (
        _require_int(
            configuration.get(
                "schema_version"
            ),
            "configuration schema_version",
        )
        != METADATA_SCHEMA_VERSION
    ):
        raise _fail(
            "configuration.json uses an incompatible metadata "
            "schema version."
        )

    if (
        _require_string(
            configuration.get(
                "experiment"
            ),
            "configuration experiment",
        )
        != EXPERIMENT_NAME
    ):
        raise _fail(
            "configuration.json names an incompatible experiment."
        )

    harmonic = _require_mapping(
        configuration.get(
            "harmonic_aggregation"
        ),
        "harmonic_aggregation",
    )

    benchmark = _require_mapping(
        configuration.get(
            "benchmark"
        ),
        "benchmark",
    )

    random_features = _require_mapping(
        configuration.get(
            "random_features"
        ),
        "random_features",
    )

    robustness = _require_mapping(
        configuration.get(
            "robustness"
        ),
        "robustness",
    )

    batch_size = _require_mapping(
        configuration.get(
            "batch_size_sensitivity"
        ),
        "batch_size_sensitivity",
    )

    scalability = _require_mapping(
        configuration.get(
            "scalability"
        ),
        "scalability",
    )

    seed = _require_mapping(
        configuration.get(
            "seed"
        ),
        "seed",
    )

    build_object = _require_mapping(
        configuration.get(
            "build"
        ),
        "build",
    )

    robustness_conditions = tuple(
        RobustnessDesignCondition(
            perturbation=_require_string(
                _require_mapping(
                    condition,
                    "robustness condition",
                ).get(
                    "perturbation"
                ),
                "robustness perturbation",
            ),
            severity=_require_float(
                _require_mapping(
                    condition,
                    "robustness condition",
                ).get(
                    "severity"
                ),
                "robustness severity",
            ),
        )
        for condition in _require_list(
            robustness.get(
                "conditions"
            ),
            "robustness conditions",
        )
    )

    generation = GenerationConfig(
        aggregation_order=_require_positive_int(
            harmonic.get(
                "aggregation_order"
            ),
            "aggregation_order",
        ),
        highest_explicit_order=_require_positive_int(
            harmonic.get(
                "highest_explicit_order"
            ),
            "highest_explicit_order",
        ),
        benchmark=BenchmarkGenerationConfig(
            vertex_count=_require_positive_int(
                benchmark.get(
                    "vertex_count"
                ),
                "benchmark vertex_count",
            ),
            edge_count=_require_positive_int(
                benchmark.get(
                    "edge_count"
                ),
                "benchmark edge_count",
            ),
            batches_per_model=_require_positive_int(
                benchmark.get(
                    "batches_per_model"
                ),
                "benchmark batches_per_model",
            ),
            graphs_per_batch=_require_positive_int(
                benchmark.get(
                    "graphs_per_batch"
                ),
                "benchmark graphs_per_batch",
            ),
        ),
        random_features=RandomFeatureGenerationConfig(
            baseline_character_count=_require_positive_int(
                random_features.get(
                    "baseline_character_count"
                ),
                "baseline_character_count",
            ),
            replicate_count=_require_positive_int(
                random_features.get(
                    "replicate_count"
                ),
                "RFF replicate_count",
            ),
            character_counts=tuple(
                _require_positive_int(
                    value,
                    "RFF character count",
                )
                for value in _require_list(
                    random_features.get(
                        "character_counts"
                    ),
                    "RFF character_counts",
                )
            ),
        ),
        robustness=RobustnessGenerationConfig(
            realization_count=_require_positive_int(
                robustness.get(
                    "realization_count"
                ),
                "robustness realization_count",
            ),
            conditions=robustness_conditions,
        ),
        batch_size_graphs_per_batch=tuple(
            _require_positive_int(
                value,
                "batch-size graphs_per_batch",
            )
            for value in _require_list(
                batch_size.get(
                    "graphs_per_batch"
                ),
                "batch-size graphs_per_batch",
            )
        ),
        scalability=ScalabilityGenerationConfig(
            graphs_per_batch=tuple(
                _require_positive_int(
                    value,
                    "scalability graphs_per_batch",
                )
                for value in _require_list(
                    scalability.get(
                        "graphs_per_batch"
                    ),
                    "scalability graphs_per_batch",
                )
            ),
            input_support_sizes=tuple(
                _require_positive_int(
                    value,
                    "scalability input support size",
                )
                for value in _require_list(
                    scalability.get(
                        "input_support_sizes"
                    ),
                    "scalability input_support_sizes",
                )
            ),
            character_counts=tuple(
                _require_positive_int(
                    value,
                    "scalability character count",
                )
                for value in _require_list(
                    scalability.get(
                        "character_counts"
                    ),
                    "scalability character_counts",
                )
            ),
            repeat_count=_require_positive_int(
                scalability.get(
                    "repeat_count"
                ),
                "scalability repeat_count",
            ),
        ),
        cross_observation_ablation_realizations=(
            _require_positive_int(
                configuration.get(
                    "cross_observation_ablation_realizations"
                ),
                "cross_observation_ablation_realizations",
            )
        ),
        canonical_root_seed=_require_uint64(
            seed.get(
                "canonical_root_seed"
            ),
            "canonical_root_seed",
        ),
        root_seed_override=_require_optional_uint64(
            seed.get(
                "root_seed_override"
            ),
            "root_seed_override",
        ),
        effective_root_seed=_require_uint64(
            seed.get(
                "effective_root_seed"
            ),
            "effective_root_seed",
        ),
    )

    build = BuildMetadata(
        created_utc=_require_string(
            build_object.get(
                "created_utc"
            ),
            "build created_utc",
        ),
        compiler=_require_string(
            build_object.get(
                "compiler"
            ),
            "build compiler",
        ),
        compiler_version=_require_string(
            build_object.get(
                "compiler_version"
            ),
            "build compiler_version",
        ),
        build_type=_require_string(
            build_object.get(
                "build_type"
            ),
            "build build_type",
        ),
        operating_system=_require_string(
            build_object.get(
                "operating_system"
            ),
            "build operating_system",
        ),
        executable=_require_string(
            build_object.get(
                "executable"
            ),
            "build executable",
        ),
    )

    return (
        generation,
        build,
    )


def _manifest_from_json(
    manifest: Mapping[str, Any],
) -> ManifestMetadata:
    if (
        _require_int(
            manifest.get(
                "schema_version"
            ),
            "manifest schema_version",
        )
        != METADATA_SCHEMA_VERSION
    ):
        raise _fail(
            "manifest.json uses an incompatible metadata "
            "schema version."
        )

    if (
        _require_string(
            manifest.get(
                "experiment"
            ),
            "manifest experiment",
        )
        != EXPERIMENT_NAME
    ):
        raise _fail(
            "manifest.json names an incompatible experiment."
        )

    if (
        _require_string(
            manifest.get(
                "status"
            ),
            "manifest status",
        )
        != "complete"
    ):
        raise _fail(
            "The C++ experiment must be complete before "
            "paper analysis begins."
        )

    studies = tuple(
        _require_string(
            study,
            "manifest study",
        )
        for study in _require_list(
            manifest.get(
                "studies"
            ),
            "manifest studies",
        )
    )

    return ManifestMetadata(
        created_utc=_require_string(
            manifest.get(
                "created_utc"
            ),
            "manifest created_utc",
        ),
        task_count=_require_positive_int(
            manifest.get(
                "task_count"
            ),
            "manifest task_count",
        ),
        studies=studies,
    )


def _load_models(
    root: Path,
) -> tuple[
    tuple[ModelMetadata, ...],
    Mapping[int, ModelMetadata],
]:
    path = root / "models.csv"

    header = (
        "model_index",
        "stable_id",
        "display_name",
        "variant",
        "vertex_count",
        "edge_count",
        "parameter_condition_count",
    )

    models: list[ModelMetadata] = []
    by_index: dict[int, ModelMetadata] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        model = ModelMetadata(
            model_index=_parse_int(
                row[0],
                "model index",
            ),
            stable_id=row[1],
            display_name=row[2],
            variant=row[3],
            vertex_count=_parse_positive_int(
                row[4],
                "model vertex count",
            ),
            edge_count=_parse_positive_int(
                row[5],
                "model edge count",
            ),
            parameter_condition_count=(
                _parse_positive_int(
                    row[6],
                    "model parameter condition count",
                )
            ),
        )

        if model.model_index in by_index:
            raise _fail(
                "models.csv contains a duplicate model index."
            )

        by_index[
            model.model_index
        ] = model

        models.append(
            model
        )

    models.sort(
        key=lambda model: model.model_index
    )

    return (
        tuple(
            models
        ),
        MappingProxyType(
            by_index
        ),
    )


def _load_model_parameters(
    root: Path,
    models: Sequence[ModelMetadata],
) -> tuple[
    tuple[ModelParameter, ...],
    Mapping[int, tuple[int, ...]],
]:
    path = root / "model_parameters.csv"

    header = (
        "model_index",
        "model_id",
        "parameter_condition_index",
        "parameter",
        "value",
    )

    parameters: list[
        ModelParameter
    ] = []

    conditions: dict[
        int,
        set[int],
    ] = {
        model.model_index: set()
        for model in models
    }

    by_model = {
        model.model_index: model
        for model in models
    }

    for row in _iter_csv_rows(
        path,
        header,
    ):
        model_index = _parse_int(
            row[0],
            "model parameter model index",
        )

        model_id = row[1]

        if (
            model_index not in by_model
            or by_model[
                model_index
            ].stable_id
            != model_id
        ):
            raise _fail(
                "model_parameters.csv contains an incompatible "
                "model identity."
            )

        condition_index = _parse_int(
            row[2],
            "parameter condition index",
        )

        parameters.append(
            ModelParameter(
                model_index=model_index,
                model_id=model_id,
                parameter_condition_index=condition_index,
                parameter=row[3],
                value=row[4],
            )
        )

        conditions[
            model_index
        ].add(
            condition_index
        )

    normalized: dict[
        int,
        tuple[int, ...],
    ] = {}

    for model in models:
        if (
            model.parameter_condition_count
            ==
            1
            and not conditions[
                model.model_index
            ]
        ):
            normalized[
                model.model_index
            ] = (
                0,
            )

            continue

        normalized[
            model.model_index
        ] = tuple(
            sorted(
                conditions[
                    model.model_index
                ]
            )
        )

    return (
        tuple(
            parameters
        ),
        MappingProxyType(
            normalized
        ),
    )


def load_common_metadata(
    root: str | Path,
) -> CommonMetadata:
    local_root = Path(
        root
    ).expanduser().resolve()

    configuration_path = _require_finalized_file(
        local_root
        / "configuration.json"
    )

    manifest_path = _require_finalized_file(
        local_root
        / "manifest.json"
    )

    generation, build = _configuration_from_json(
        _load_json_object(
            configuration_path
        )
    )

    manifest = _manifest_from_json(
        _load_json_object(
            manifest_path
        )
    )

    if (
        manifest.created_utc
        != build.created_utc
    ):
        raise _fail(
            "configuration.json and manifest.json disagree on "
            "the experiment creation timestamp."
        )

    models, model_by_index = _load_models(
        local_root
    )

    (
        model_parameters,
        parameter_condition_indices,
    ) = _load_model_parameters(
        local_root,
        models,
    )

    return CommonMetadata(
        root=local_root,
        generation=generation,
        build=build,
        manifest=manifest,
        models=models,
        model_by_index=model_by_index,
        model_parameters=model_parameters,
        parameter_condition_indices=(
            parameter_condition_indices
        ),
    )


def _require_model_identity(
    model_by_index: Mapping[int, ModelMetadata],
    model_index: int,
    model_id: str,
    description: str,
) -> ModelMetadata:
    try:
        model = model_by_index[
            model_index
        ]
    except KeyError as error:
        raise _fail(
            f"{description} refers to unknown model "
            f"index {model_index}."
        ) from error

    if model.stable_id != model_id:
        raise _fail(
            f"{description} model ID does not match "
            "its model index."
        )

    return model


def _load_benchmark_batches(
    common: CommonMetadata,
) -> tuple[
    tuple[BenchmarkBatch, ...],
    Mapping[
        BenchmarkSampleKey,
        BenchmarkBatch,
    ],
]:
    path = (
        common.root
        / "benchmark"
        / "batches.csv"
    )

    header = (
        "schema_version",
        "model_index",
        "model_id",
        "model_name",
        "model_variant",
        "parameter_condition_index",
        "batch_index",
        "graphs_per_batch",
        "latent_seed",
        "character_seed",
    )

    batches: list[
        BenchmarkBatch
    ] = []

    by_key: dict[
        BenchmarkSampleKey,
        BenchmarkBatch,
    ] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        model_index = _parse_int(
            row[1],
            "benchmark model index",
        )

        _require_model_identity(
            common.model_by_index,
            model_index,
            row[2],
            "Benchmark batch",
        )

        batch = BenchmarkBatch(
            model_index=model_index,
            model_id=row[2],
            model_name=row[3],
            model_variant=row[4],
            parameter_condition_index=_parse_int(
                row[5],
                "benchmark parameter condition index",
            ),
            batch_index=_parse_int(
                row[6],
                "benchmark batch index",
            ),
            graphs_per_batch=_parse_positive_int(
                row[7],
                "benchmark graphs_per_batch",
            ),
            latent_seed=_parse_uint64(
                row[8],
                "benchmark latent seed",
            ),
            character_seed=_parse_uint64(
                row[9],
                "benchmark character seed",
            ),
        )

        if batch.key in by_key:
            raise _fail(
                "benchmark/batches.csv contains a duplicate "
                f"sample key {batch.key!r}."
            )

        by_key[
            batch.key
        ] = batch

        batches.append(
            batch
        )

    batches.sort(
        key=lambda batch: batch.key
    )

    return (
        tuple(
            batches
        ),
        MappingProxyType(
            by_key
        ),
    )


def _iter_benchmark_graph_file(
    path: Path,
    model_by_index: Mapping[
        int,
        ModelMetadata,
    ],
) -> Iterator[SerializedGraph]:
    header = (
        "schema_version",
        "record_type",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "latent_seed",
        "sampling_seed",
        "vertex_count",
        "edge_count",
        "u",
        "v",
        "filtration_step",
        "filtration_time",
    )

    active: dict[str, Any] | None = None
    edges: list[
        SerializedEdge
    ] = []

    def finish(
    ) -> SerializedGraph | None:
        nonlocal active
        nonlocal edges

        if active is None:
            return None

        graph = SerializedGraph(
            model_index=active[
                "model_index"
            ],
            model_id=active[
                "model_id"
            ],
            parameter_condition_index=active[
                "parameter_condition_index"
            ],
            batch_index=active[
                "batch_index"
            ],
            graph_index=active[
                "graph_index"
            ],
            latent_seed=active[
                "latent_seed"
            ],
            sampling_seed=active[
                "sampling_seed"
            ],
            vertex_count=active[
                "vertex_count"
            ],
            edge_count=active[
                "edge_count"
            ],
            edges=tuple(
                edges
            ),
        )

        if len(
            graph.edges
        ) != graph.edge_count:
            raise _fail(
                "Benchmark graph edge count does not match "
                "its edge rows."
            )

        active = None
        edges = []

        return graph

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        record_type = row[1]

        model_index = _parse_int(
            row[2],
            "benchmark graph model index",
        )

        _require_model_identity(
            model_by_index,
            model_index,
            row[3],
            "Benchmark graph",
        )

        key = (
            model_index,
            _parse_int(
                row[4],
                "benchmark graph parameter condition index",
            ),
            _parse_int(
                row[5],
                "benchmark graph batch index",
            ),
            _parse_int(
                row[6],
                "benchmark graph index",
            ),
        )

        if record_type == "graph":
            completed = finish()

            if completed is not None:
                yield completed

            active = {
                "model_index": key[0],
                "model_id": row[3],
                "parameter_condition_index": key[1],
                "batch_index": key[2],
                "graph_index": key[3],
                "latent_seed": _parse_uint64(
                    row[7],
                    "benchmark latent seed",
                ),
                "sampling_seed": _parse_uint64(
                    row[8],
                    "benchmark sampling seed",
                ),
                "vertex_count": _parse_positive_int(
                    row[9],
                    "benchmark vertex count",
                ),
                "edge_count": _parse_int(
                    row[10],
                    "benchmark edge count",
                ),
            }

            edges = []

            continue

        if record_type != "edge":
            raise _fail(
                "benchmark/graphs.csv contains an unknown "
                f"record type {record_type!r}."
            )

        if active is None:
            raise _fail(
                "Benchmark edge row appears before its graph row."
            )

        active_key = (
            active[
                "model_index"
            ],
            active[
                "parameter_condition_index"
            ],
            active[
                "batch_index"
            ],
            active[
                "graph_index"
            ],
        )

        if key != active_key:
            raise _fail(
                "Benchmark edge row appears outside its graph block."
            )

        edges.append(
            SerializedEdge(
                u=_parse_int(
                    row[11],
                    "benchmark edge u",
                ),
                v=_parse_int(
                    row[12],
                    "benchmark edge v",
                ),
                filtration_step=_parse_int(
                    row[13],
                    "benchmark filtration step",
                ),
                filtration_time=_parse_float(
                    row[14],
                    "benchmark filtration time",
                ),
            )
        )

    completed = finish()

    if completed is not None:
        yield completed


def _iter_graph_fields(
    path: Path,
) -> Iterator[GraphFieldRecord]:
    header = (
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "field_name",
        "value",
    )

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        yield GraphFieldRecord(
            model_index=_parse_int(
                row[1],
                "graph-field model index",
            ),
            model_id=row[2],
            parameter_condition_index=_parse_int(
                row[3],
                "graph-field parameter condition index",
            ),
            batch_index=_parse_int(
                row[4],
                "graph-field batch index",
            ),
            graph_index=_parse_int(
                row[5],
                "graph-field graph index",
            ),
            field_name=row[6],
            value=_parse_float(
                row[7],
                "graph-field value",
            ),
        )


def _iter_vertex_fields(
    path: Path,
) -> Iterator[VertexFieldRecord]:
    header = (
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "vertex_index",
        "field_name",
        "value",
    )

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        yield VertexFieldRecord(
            model_index=_parse_int(
                row[1],
                "vertex-field model index",
            ),
            model_id=row[2],
            parameter_condition_index=_parse_int(
                row[3],
                "vertex-field parameter condition index",
            ),
            batch_index=_parse_int(
                row[4],
                "vertex-field batch index",
            ),
            graph_index=_parse_int(
                row[5],
                "vertex-field graph index",
            ),
            vertex_index=_parse_int(
                row[6],
                "vertex-field vertex index",
            ),
            field_name=row[7],
            value=_parse_float(
                row[8],
                "vertex-field value",
            ),
        )


def _iter_edge_fields(
    path: Path,
) -> Iterator[EdgeFieldRecord]:
    header = (
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "edge_index",
        "u",
        "v",
        "field_name",
        "value",
    )

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        yield EdgeFieldRecord(
            model_index=_parse_int(
                row[1],
                "edge-field model index",
            ),
            model_id=row[2],
            parameter_condition_index=_parse_int(
                row[3],
                "edge-field parameter condition index",
            ),
            batch_index=_parse_int(
                row[4],
                "edge-field batch index",
            ),
            graph_index=_parse_int(
                row[5],
                "edge-field graph index",
            ),
            edge_index=_parse_int(
                row[6],
                "edge-field edge index",
            ),
            u=_parse_int(
                row[7],
                "edge-field u",
            ),
            v=_parse_int(
                row[8],
                "edge-field v",
            ),
            field_name=row[9],
            value=_parse_float(
                row[10],
                "edge-field value",
            ),
        )


def _iter_grouped_benchmark_atoms(
    path: Path,
) -> Iterator[
    tuple[
        BenchmarkGraphKey,
        tuple[
            PersistenceAtom,
            ...,
        ],
    ]
]:
    header = (
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "birth",
        "death",
        "coefficient",
    )

    active_key: BenchmarkGraphKey | None = None
    atoms: list[
        PersistenceAtom
    ] = []

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        key: BenchmarkGraphKey = (
            _parse_int(
                row[1],
                "benchmark diagram model index",
            ),
            _parse_int(
                row[3],
                "benchmark diagram parameter condition index",
            ),
            _parse_int(
                row[4],
                "benchmark diagram batch index",
            ),
            _parse_int(
                row[5],
                "benchmark diagram graph index",
            ),
        )

        if (
            active_key is not None
            and key < active_key
        ):
            raise _fail(
                "Benchmark persistence-diagram rows are not ordered "
                "by graph key."
            )

        if (
            active_key is not None
            and key != active_key
        ):
            yield (
                active_key,
                tuple(
                    atoms
                ),
            )

            atoms = []

        active_key = key

        atoms.append(
            PersistenceAtom(
                birth=_parse_float(
                    row[6],
                    "benchmark persistence birth",
                ),
                death=_parse_float(
                    row[7],
                    "benchmark persistence death",
                ),
                coefficient=_parse_float(
                    row[8],
                    "benchmark persistence coefficient",
                ),
            )
        )

    if active_key is not None:
        yield (
            active_key,
            tuple(
                atoms
            ),
        )


def _iter_benchmark_diagrams(
    graphs: BenchmarkGraphCollection,
    diagram_path: Path,
) -> Iterator[
    tuple[
        BenchmarkGraphKey,
        PersistenceDiagram,
    ]
]:
    atom_iterator = _iter_grouped_benchmark_atoms(
        diagram_path
    )

    try:
        atom_item = next(
            atom_iterator
        )
    except StopIteration:
        atom_item = None

    previous_graph_key: BenchmarkGraphKey | None = None

    for graph in graphs:
        graph_key = graph.key

        if (
            previous_graph_key is not None
            and graph_key <= previous_graph_key
        ):
            raise _fail(
                "Benchmark graph rows are not ordered by graph key."
            )

        previous_graph_key = graph_key

        if (
            atom_item is not None
            and atom_item[0] < graph_key
        ):
            raise _fail(
                "Benchmark persistence-diagram data contains a "
                "graph key absent from the benchmark graph stream."
            )

        if (
            atom_item is not None
            and atom_item[0] == graph_key
        ):
            yield (
                graph_key,
                PersistenceDiagram(
                    atoms=atom_item[1]
                ),
            )

            try:
                atom_item = next(
                    atom_iterator
                )
            except StopIteration:
                atom_item = None

            continue

        yield (
            graph_key,
            PersistenceDiagram(
                atoms=()
            ),
        )

    if atom_item is not None:
        raise _fail(
            "Benchmark persistence-diagram data contains a graph "
            "key absent from the benchmark graph stream."
        )


def _load_harmonic_feature_groups(
    path: Path,
    identity_columns: Sequence[str],
    key_parser: Any,
    *,
    seed_column: int | None = None,
) -> Mapping[
    tuple[Any, ...],
    HarmonicFeature,
]:
    header = tuple(
        identity_columns
    ) + (
        "character_index",
        "real",
        "imag",
    )

    grouped: dict[
        tuple[Any, ...],
        list[
            tuple[
                int,
                float,
                float,
            ]
        ],
    ] = {}

    seeds: dict[
        tuple[Any, ...],
        int | None,
    ] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        key = key_parser(
            row
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            (
                _parse_int(
                    row[
                        len(
                            identity_columns
                        )
                    ],
                    "character index",
                ),
                _parse_float(
                    row[
                        len(
                            identity_columns
                        )
                        + 1
                    ],
                    "harmonic real coordinate",
                ),
                _parse_float(
                    row[
                        len(
                            identity_columns
                        )
                        + 2
                    ],
                    "harmonic imaginary coordinate",
                ),
            )
        )

        if seed_column is not None:
            seeds[
                key
            ] = _parse_uint64(
                row[
                    seed_column
                ],
                "character seed",
            )

    features: dict[
        tuple[Any, ...],
        HarmonicFeature,
    ] = {}

    for key, coordinates in grouped.items():
        coordinates.sort(
            key=lambda item: item[0]
        )

        values: list[float] = []

        for (
            _,
            real,
            imag,
        ) in coordinates:
            values.extend(
                (
                    real,
                    imag,
                )
            )

        features[
            key
        ] = HarmonicFeature(
            key=key,
            character_seed=seeds.get(
                key
            ),
            values=_freeze_vector(
                values
            ),
        )

    return MappingProxyType(
        features
    )


def load_benchmark_catalog(
    common: CommonMetadata,
) -> BenchmarkCatalog:
    directory = (
        common.root
        / "benchmark"
    )

    (
        batches,
        batch_by_key,
    ) = _load_benchmark_batches(
        common
    )

    graph_collection = BenchmarkGraphCollection(
        path=directory / "graphs.csv",
        model_by_index=common.model_by_index,
    )

    harmonic_features = _load_harmonic_feature_groups(
        directory
        / "harmonic_aggregation.csv",
        (
            "schema_version",
            "model_index",
            "model_id",
            "parameter_condition_index",
            "batch_index",
            "character_seed",
        ),
        lambda row: (
            _parse_int(
                row[1],
                "benchmark harmonic model index",
            ),
            _parse_int(
                row[3],
                "benchmark harmonic parameter condition index",
            ),
            _parse_int(
                row[4],
                "benchmark harmonic batch index",
            ),
        ),
        seed_column=5,
    )

    return BenchmarkCatalog(
        batches=batches,
        batch_by_key=batch_by_key,
        graphs=graph_collection,
        graph_fields=GraphFieldCollection(
            directory
            / "graph_fields.csv"
        ),
        vertex_fields=VertexFieldCollection(
            directory
            / "vertex_fields.csv"
        ),
        edge_fields=EdgeFieldCollection(
            directory
            / "edge_fields.csv"
        ),
        diagrams=BenchmarkDiagramCollection(
            graph_collection=graph_collection,
            path=(
                directory
                / "persistence_diagrams.csv"
            ),
        ),
        harmonic_features=harmonic_features,
    )


def _load_ablation_conditions(
    common: CommonMetadata,
) -> Mapping[
    AblationKey,
    AblationCondition,
]:
    path = (
        common.root
        / "harmonic_aggregation_ablation"
        / "conditions.csv"
    )

    header = (
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "ablation_kind",
        "realization_index",
        "graphs_per_batch",
        "aggregation_order",
        "character_count",
        "character_seed",
        "ablation_seed",
    )

    conditions: dict[
        AblationKey,
        AblationCondition,
    ] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        condition = AblationCondition(
            model_index=_parse_int(
                row[1],
                "ablation model index",
            ),
            model_id=row[2],
            parameter_condition_index=_parse_int(
                row[3],
                "ablation parameter condition index",
            ),
            batch_index=_parse_int(
                row[4],
                "ablation batch index",
            ),
            ablation_kind=row[5],
            realization_index=_parse_int(
                row[6],
                "ablation realization index",
            ),
            graphs_per_batch=_parse_positive_int(
                row[7],
                "ablation graphs_per_batch",
            ),
            aggregation_order=_parse_positive_int(
                row[8],
                "ablation aggregation order",
            ),
            character_count=_parse_positive_int(
                row[9],
                "ablation character count",
            ),
            character_seed=_parse_uint64(
                row[10],
                "ablation character seed",
            ),
            ablation_seed=_parse_optional_uint64(
                row[11],
                "ablation seed",
            ),
        )

        if condition.key in conditions:
            raise _fail(
                "Ablation conditions contain a duplicate key."
            )

        conditions[
            condition.key
        ] = condition

    return MappingProxyType(
        conditions
    )


def load_ablation_catalog(
    common: CommonMetadata,
) -> AblationCatalog:
    directory = (
        common.root
        / "harmonic_aggregation_ablation"
    )

    conditions = _load_ablation_conditions(
        common
    )

    features = _load_harmonic_feature_groups(
        directory
        / "harmonic_aggregation.csv",
        (
            "schema_version",
            "model_index",
            "model_id",
            "parameter_condition_index",
            "batch_index",
            "ablation_kind",
            "realization_index",
        ),
        lambda row: (
            _parse_int(
                row[1],
                "ablation harmonic model index",
            ),
            _parse_int(
                row[3],
                "ablation harmonic parameter condition index",
            ),
            _parse_int(
                row[4],
                "ablation harmonic batch index",
            ),
            row[5],
            _parse_int(
                row[6],
                "ablation harmonic realization index",
            ),
        ),
    )

    return AblationCatalog(
        conditions=conditions,
        harmonic_features=features,
    )


def _load_batch_size_conditions(
    common: CommonMetadata,
) -> Mapping[
    BenchmarkSampleKey,
    BatchSizeCondition,
]:
    path = (
        common.root
        / "batch_size_sensitivity"
        / "conditions.csv"
    )

    header = (
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "maximum_graphs_per_batch",
        "character_seed",
        "condition_count",
    )

    conditions: dict[
        BenchmarkSampleKey,
        BatchSizeCondition,
    ] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        condition = BatchSizeCondition(
            model_index=_parse_int(
                row[1],
                "batch-size model index",
            ),
            model_id=row[2],
            parameter_condition_index=_parse_int(
                row[3],
                "batch-size parameter condition index",
            ),
            batch_index=_parse_int(
                row[4],
                "batch-size batch index",
            ),
            maximum_graphs_per_batch=_parse_positive_int(
                row[5],
                "batch-size maximum graphs_per_batch",
            ),
            character_seed=_parse_uint64(
                row[6],
                "batch-size character seed",
            ),
            condition_count=_parse_positive_int(
                row[7],
                "batch-size condition count",
            ),
        )

        if condition.sample_key in conditions:
            raise _fail(
                "Batch-size conditions contain a duplicate "
                "sample key."
            )

        conditions[
            condition.sample_key
        ] = condition

    return MappingProxyType(
        conditions
    )


def load_batch_size_catalog(
    common: CommonMetadata,
) -> BatchSizeCatalog:
    directory = (
        common.root
        / "batch_size_sensitivity"
    )

    conditions = _load_batch_size_conditions(
        common
    )

    features = _load_harmonic_feature_groups(
        directory
        / "harmonic_aggregation.csv",
        (
            "schema_version",
            "model_index",
            "model_id",
            "parameter_condition_index",
            "batch_index",
            "graphs_per_batch",
        ),
        lambda row: (
            _parse_int(
                row[1],
                "batch-size harmonic model index",
            ),
            _parse_int(
                row[3],
                "batch-size harmonic parameter condition index",
            ),
            _parse_int(
                row[4],
                "batch-size harmonic batch index",
            ),
            _parse_positive_int(
                row[5],
                "batch-size harmonic graphs_per_batch",
            ),
        ),
    )

    graph_counts = tuple(
        sorted(
            {
                int(
                    key[3]
                )
                for key in features
            }
        )
    )

    return BatchSizeCatalog(
        conditions=conditions,
        harmonic_features=features,
        graph_counts=graph_counts,
    )


def _load_rff_conditions(
    common: CommonMetadata,
) -> Mapping[
    tuple[
        int,
        int,
        int,
        int,
    ],
    RffCondition,
]:
    path = (
        common.root
        / "rff_sensitivity"
        / "conditions.csv"
    )

    header = (
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "replicate_index",
        "character_seed",
        "condition_count",
    )

    conditions: dict[
        tuple[
            int,
            int,
            int,
            int,
        ],
        RffCondition,
    ] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        condition = RffCondition(
            model_index=_parse_int(
                row[1],
                "RFF model index",
            ),
            model_id=row[2],
            parameter_condition_index=_parse_int(
                row[3],
                "RFF parameter condition index",
            ),
            batch_index=_parse_int(
                row[4],
                "RFF batch index",
            ),
            replicate_index=_parse_int(
                row[5],
                "RFF replicate index",
            ),
            character_seed=_parse_uint64(
                row[6],
                "RFF character seed",
            ),
            condition_count=_parse_positive_int(
                row[7],
                "RFF condition count",
            ),
        )

        if condition.key in conditions:
            raise _fail(
                "RFF conditions contain a duplicate key."
            )

        conditions[
            condition.key
        ] = condition

    return MappingProxyType(
        conditions
    )


def load_rff_catalog(
    common: CommonMetadata,
) -> RffCatalog:
    directory = (
        common.root
        / "rff_sensitivity"
    )

    conditions = _load_rff_conditions(
        common
    )

    features = _load_harmonic_feature_groups(
        directory
        / "harmonic_aggregation.csv",
        (
            "schema_version",
            "model_index",
            "model_id",
            "parameter_condition_index",
            "batch_index",
            "replicate_index",
            "character_count",
        ),
        lambda row: (
            _parse_int(
                row[1],
                "RFF harmonic model index",
            ),
            _parse_int(
                row[3],
                "RFF harmonic parameter condition index",
            ),
            _parse_int(
                row[4],
                "RFF harmonic batch index",
            ),
            _parse_int(
                row[5],
                "RFF harmonic replicate index",
            ),
            _parse_positive_int(
                row[6],
                "RFF harmonic character count",
            ),
        ),
    )

    character_counts = tuple(
        sorted(
            {
                int(
                    key[4]
                )
                for key in features
            }
        )
    )

    replicate_indices = tuple(
        sorted(
            {
                condition.replicate_index
                for condition in conditions.values()
            }
        )
    )

    return RffCatalog(
        conditions=conditions,
        harmonic_features=features,
        character_counts=character_counts,
        replicate_indices=replicate_indices,
    )


def _load_robustness_conditions(
    common: CommonMetadata,
) -> Mapping[
    RobustnessConditionKey,
    RobustnessCondition,
]:
    path = (
        common.root
        / "robustness"
        / "conditions.csv"
    )

    header = (
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "condition_index",
        "perturbation",
        "severity",
        "realization_index",
        "graphs_per_batch",
        "character_seed",
    )

    conditions: dict[
        RobustnessConditionKey,
        RobustnessCondition,
    ] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        condition = RobustnessCondition(
            model_index=_parse_int(
                row[1],
                "robustness model index",
            ),
            model_id=row[2],
            parameter_condition_index=_parse_int(
                row[3],
                "robustness parameter condition index",
            ),
            batch_index=_parse_int(
                row[4],
                "robustness batch index",
            ),
            condition_index=_parse_int(
                row[5],
                "robustness condition index",
            ),
            perturbation=row[6],
            severity=_parse_float(
                row[7],
                "robustness severity",
            ),
            realization_index=_parse_int(
                row[8],
                "robustness realization index",
            ),
            graphs_per_batch=_parse_positive_int(
                row[9],
                "robustness graphs_per_batch",
            ),
            character_seed=_parse_uint64(
                row[10],
                "robustness character seed",
            ),
        )

        if condition.key in conditions:
            raise _fail(
                "Robustness conditions contain a duplicate key."
            )

        conditions[
            condition.key
        ] = condition

    return MappingProxyType(
        conditions
    )


def _iter_robustness_graph_file(
    path: Path,
    model_by_index: Mapping[
        int,
        ModelMetadata,
    ],
    conditions: Mapping[
        RobustnessConditionKey,
        RobustnessCondition,
    ],
) -> Iterator[RobustnessGraph]:
    header = (
        "schema_version",
        "record_type",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "condition_index",
        "perturbation",
        "severity",
        "realization_index",
        "perturbation_seed",
        "vertex_count",
        "edge_count",
        "u",
        "v",
        "filtration_step",
        "filtration_time",
    )

    active: dict[str, Any] | None = None
    edges: list[
        SerializedEdge
    ] = []

    def finish(
    ) -> RobustnessGraph | None:
        nonlocal active
        nonlocal edges

        if active is None:
            return None

        graph = RobustnessGraph(
            model_index=active[
                "model_index"
            ],
            model_id=active[
                "model_id"
            ],
            parameter_condition_index=active[
                "parameter_condition_index"
            ],
            batch_index=active[
                "batch_index"
            ],
            graph_index=active[
                "graph_index"
            ],
            condition_index=active[
                "condition_index"
            ],
            perturbation=active[
                "perturbation"
            ],
            severity=active[
                "severity"
            ],
            realization_index=active[
                "realization_index"
            ],
            perturbation_seed=active[
                "perturbation_seed"
            ],
            vertex_count=active[
                "vertex_count"
            ],
            edge_count=active[
                "edge_count"
            ],
            edges=tuple(
                edges
            ),
        )

        if len(
            graph.edges
        ) != graph.edge_count:
            raise _fail(
                "Robustness graph edge count does not match "
                "its edge rows."
            )

        active = None
        edges = []

        return graph

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        model_index = _parse_int(
            row[2],
            "robustness graph model index",
        )

        _require_model_identity(
            model_by_index,
            model_index,
            row[3],
            "Robustness graph",
        )

        parameter_condition_index = _parse_int(
            row[4],
            "robustness graph parameter condition index",
        )

        batch_index = _parse_int(
            row[5],
            "robustness graph batch index",
        )

        graph_index = _parse_int(
            row[6],
            "robustness graph index",
        )

        condition_index = _parse_int(
            row[7],
            "robustness graph condition index",
        )

        realization_index = _parse_int(
            row[10],
            "robustness graph realization index",
        )

        condition_key: RobustnessConditionKey = (
            model_index,
            parameter_condition_index,
            batch_index,
            condition_index,
            realization_index,
        )

        try:
            condition = conditions[
                condition_key
            ]
        except KeyError as error:
            raise _fail(
                "Robustness graph row does not correspond to a "
                "loaded robustness condition."
            ) from error

        if (
            row[8]
            != condition.perturbation
            or _parse_float(
                row[9],
                "robustness graph severity",
            )
            != condition.severity
        ):
            raise _fail(
                "Robustness graph row disagrees with its "
                "robustness condition."
            )

        graph_key: RobustnessGraphKey = (
            model_index,
            parameter_condition_index,
            batch_index,
            graph_index,
            condition_index,
            realization_index,
        )

        record_type = row[1]

        if record_type == "graph":
            completed = finish()

            if completed is not None:
                yield completed

            active = {
                "model_index": model_index,
                "model_id": condition.model_id,
                "parameter_condition_index": (
                    parameter_condition_index
                ),
                "batch_index": batch_index,
                "graph_index": graph_index,
                "condition_index": condition_index,
                "perturbation": condition.perturbation,
                "severity": condition.severity,
                "realization_index": realization_index,
                "perturbation_seed": _parse_uint64(
                    row[11],
                    "robustness perturbation seed",
                ),
                "vertex_count": _parse_positive_int(
                    row[12],
                    "robustness vertex count",
                ),
                "edge_count": _parse_int(
                    row[13],
                    "robustness edge count",
                ),
                "key": graph_key,
            }

            edges = []

            continue

        if record_type != "edge":
            raise _fail(
                "robustness/graphs.csv contains an unknown "
                f"record type {record_type!r}."
            )

        if (
            active is None
            or active[
                "key"
            ]
            != graph_key
        ):
            raise _fail(
                "Robustness edge row appears outside its graph "
                "block."
            )

        edges.append(
            SerializedEdge(
                u=_parse_int(
                    row[14],
                    "robustness edge u",
                ),
                v=_parse_int(
                    row[15],
                    "robustness edge v",
                ),
                filtration_step=_parse_int(
                    row[16],
                    "robustness filtration step",
                ),
                filtration_time=_parse_float(
                    row[17],
                    "robustness filtration time",
                ),
            )
        )

    completed = finish()

    if completed is not None:
        yield completed


def _iter_grouped_robustness_atoms(
    path: Path,
    conditions: Mapping[
        RobustnessConditionKey,
        RobustnessCondition,
    ],
) -> Iterator[
    tuple[
        RobustnessGraphKey,
        tuple[
            PersistenceAtom,
            ...,
        ],
    ]
]:
    header = (
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "condition_index",
        "perturbation",
        "severity",
        "realization_index",
        "birth",
        "death",
        "coefficient",
    )

    active_key: RobustnessGraphKey | None = None
    atoms: list[
        PersistenceAtom
    ] = []

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        model_index = _parse_int(
            row[1],
            "robustness diagram model index",
        )

        parameter_condition_index = _parse_int(
            row[3],
            "robustness diagram parameter condition index",
        )

        batch_index = _parse_int(
            row[4],
            "robustness diagram batch index",
        )

        graph_index = _parse_int(
            row[5],
            "robustness diagram graph index",
        )

        condition_index = _parse_int(
            row[6],
            "robustness diagram condition index",
        )

        realization_index = _parse_int(
            row[9],
            "robustness diagram realization index",
        )

        condition_key: RobustnessConditionKey = (
            model_index,
            parameter_condition_index,
            batch_index,
            condition_index,
            realization_index,
        )

        try:
            condition = conditions[
                condition_key
            ]
        except KeyError as error:
            raise _fail(
                "Robustness persistence row does not correspond "
                "to a loaded robustness condition."
            ) from error

        if (
            row[2]
            != condition.model_id
            or row[7]
            != condition.perturbation
            or _parse_float(
                row[8],
                "robustness diagram severity",
            )
            != condition.severity
        ):
            raise _fail(
                "Robustness persistence row disagrees with its "
                "robustness condition."
            )

        key: RobustnessGraphKey = (
            model_index,
            parameter_condition_index,
            batch_index,
            graph_index,
            condition_index,
            realization_index,
        )

        if (
            active_key is not None
            and key < active_key
        ):
            raise _fail(
                "Robustness persistence-diagram rows are not "
                "ordered by graph key."
            )

        if (
            active_key is not None
            and key != active_key
        ):
            yield (
                active_key,
                tuple(
                    atoms
                ),
            )

            atoms = []

        active_key = key

        atoms.append(
            PersistenceAtom(
                birth=_parse_float(
                    row[10],
                    "robustness persistence birth",
                ),
                death=_parse_float(
                    row[11],
                    "robustness persistence death",
                ),
                coefficient=_parse_float(
                    row[12],
                    "robustness persistence coefficient",
                ),
            )
        )

    if active_key is not None:
        yield (
            active_key,
            tuple(
                atoms
            ),
        )


def _iter_robustness_diagrams(
    graphs: RobustnessGraphCollection,
    diagram_path: Path,
    conditions: Mapping[
        RobustnessConditionKey,
        RobustnessCondition,
    ],
) -> Iterator[
    RobustnessGraphDiagram
]:
    atom_iterator = _iter_grouped_robustness_atoms(
        diagram_path,
        conditions,
    )

    try:
        atom_item = next(
            atom_iterator
        )
    except StopIteration:
        atom_item = None

    previous_graph_key: RobustnessGraphKey | None = None

    for graph in graphs:
        graph_key = graph.key

        if (
            previous_graph_key is not None
            and graph_key <= previous_graph_key
        ):
            raise _fail(
                "Robustness graph rows are not ordered by graph key."
            )

        previous_graph_key = graph_key

        if (
            atom_item is not None
            and atom_item[0] < graph_key
        ):
            raise _fail(
                "Robustness persistence-diagram data contains a "
                "graph key absent from the robustness graph stream."
            )

        if (
            atom_item is not None
            and atom_item[0] == graph_key
        ):
            diagram = PersistenceDiagram(
                atoms=atom_item[1]
            )

            try:
                atom_item = next(
                    atom_iterator
                )
            except StopIteration:
                atom_item = None
        else:
            diagram = PersistenceDiagram(
                atoms=()
            )

        yield RobustnessGraphDiagram(
            graph=graph,
            diagram=diagram,
        )

    if atom_item is not None:
        raise _fail(
            "Robustness persistence-diagram data contains a graph "
            "key absent from the robustness graph stream."
        )


def load_robustness_catalog(
    common: CommonMetadata,
) -> RobustnessCatalog:
    directory = (
        common.root
        / "robustness"
    )

    conditions = _load_robustness_conditions(
        common
    )

    features = _load_harmonic_feature_groups(
        directory
        / "harmonic_aggregation.csv",
        (
            "schema_version",
            "model_index",
            "model_id",
            "parameter_condition_index",
            "batch_index",
            "condition_index",
            "realization_index",
            "character_seed",
        ),
        lambda row: (
            _parse_int(
                row[1],
                "robustness harmonic model index",
            ),
            _parse_int(
                row[3],
                "robustness harmonic parameter condition index",
            ),
            _parse_int(
                row[4],
                "robustness harmonic batch index",
            ),
            _parse_int(
                row[5],
                "robustness harmonic condition index",
            ),
            _parse_int(
                row[6],
                "robustness harmonic realization index",
            ),
        ),
        seed_column=7,
    )

    graph_collection = RobustnessGraphCollection(
        path=directory / "graphs.csv",
        model_by_index=common.model_by_index,
        conditions=conditions,
    )

    diagram_collection = RobustnessDiagramCollection(
        graph_collection=graph_collection,
        path=directory / "persistence_diagrams.csv",
        conditions=conditions,
    )

    return RobustnessCatalog(
        conditions=conditions,
        harmonic_features=features,
        graphs=graph_collection,
        diagrams=diagram_collection,
    )


def _load_scalability_conditions(
    common: CommonMetadata,
) -> Mapping[
    ScalabilityKey,
    ScalabilityCondition,
]:
    path = (
        common.root
        / "scalability"
        / "conditions.csv"
    )

    header = (
        "schema_version",
        "scalability_condition_index",
        "repeat_index",
        "graphs_per_batch",
        "input_support_size",
        "character_count",
        "sample_seed",
        "character_seed",
    )

    conditions: dict[
        ScalabilityKey,
        ScalabilityCondition,
    ] = {}

    for row in _iter_csv_rows(
        path,
        header,
    ):
        _require_output_schema(
            row,
            path,
        )

        condition = ScalabilityCondition(
            scalability_condition_index=_parse_int(
                row[1],
                "scalability condition index",
            ),
            repeat_index=_parse_int(
                row[2],
                "scalability repeat index",
            ),
            graphs_per_batch=_parse_positive_int(
                row[3],
                "scalability graphs_per_batch",
            ),
            input_support_size=_parse_positive_int(
                row[4],
                "scalability input support size",
            ),
            character_count=_parse_positive_int(
                row[5],
                "scalability character count",
            ),
            sample_seed=_parse_uint64(
                row[6],
                "scalability sample seed",
            ),
            character_seed=_parse_uint64(
                row[7],
                "scalability character seed",
            ),
        )

        key = (
            condition.scalability_condition_index
        )

        if key in conditions:
            raise _fail(
                "Scalability conditions contain a duplicate "
                "condition index."
            )

        conditions[
            key
        ] = condition

    return MappingProxyType(
        conditions
    )


def load_scalability_catalog(
    common: CommonMetadata,
) -> ScalabilityCatalog:
    directory = (
        common.root
        / "scalability"
    )

    conditions = _load_scalability_conditions(
        common
    )

    timing_path = (
        directory
        / "timings.csv"
    )

    timing_header = (
        "schema_version",
        "scalability_condition_index",
        "representation_seconds",
    )

    timings: dict[
        ScalabilityKey,
        float,
    ] = {}

    for row in _iter_csv_rows(
        timing_path,
        timing_header,
    ):
        _require_output_schema(
            row,
            timing_path,
        )

        key = _parse_int(
            row[1],
            "scalability timing condition index",
        )

        seconds = _parse_float(
            row[2],
            "scalability representation time",
        )

        if seconds <= 0.0:
            raise _fail(
                "Scalability representation time must be positive."
            )

        if key in timings:
            raise _fail(
                "Scalability timings contain a duplicate "
                f"condition index {key}."
            )

        timings[
            key
        ] = seconds

    measurements: dict[
        ScalabilityKey,
        ScalabilityMeasurement,
    ] = {}

    for key, condition in conditions.items():
        try:
            representation_seconds = timings[
                key
            ]
        except KeyError as error:
            raise _fail(
                "Scalability timing is absent for condition "
                f"index {key}."
            ) from error

        measurements[
            key
        ] = ScalabilityMeasurement(
            condition=condition,
            representation_seconds=representation_seconds,
        )

    return ScalabilityCatalog(
        measurements=MappingProxyType(
            measurements
        )
    )


def load_random_graph_classification_data(
    root: str | Path,
) -> RandomGraphClassificationData:
    common = load_common_metadata(
        root
    )

    benchmark = load_benchmark_catalog(
        common
    )

    ablation = load_ablation_catalog(
        common
    )

    batch_size = load_batch_size_catalog(
        common
    )

    rff = load_rff_catalog(
        common
    )

    robustness = load_robustness_catalog(
        common
    )

    scalability = load_scalability_catalog(
        common
    )

    return RandomGraphClassificationData(
        common=common,
        benchmark=benchmark,
        ablation=ablation,
        batch_size=batch_size,
        rff=rff,
        robustness=robustness,
        scalability=scalability,
    )


__all__ = [
    "AblationCatalog",
    "AblationCondition",
    "AblationKey",
    "BatchSizeCatalog",
    "BatchSizeCondition",
    "BatchSizeKey",
    "BenchmarkBatch",
    "BenchmarkCatalog",
    "BenchmarkDiagramCollection",
    "BenchmarkGraphCollection",
    "BenchmarkGraphKey",
    "BenchmarkSampleKey",
    "BuildMetadata",
    "CommonMetadata",
    "EdgeFieldCollection",
    "EdgeFieldRecord",
    "GenerationConfig",
    "GraphFieldCollection",
    "GraphFieldRecord",
    "HarmonicFeature",
    "ManifestMetadata",
    "METADATA_SCHEMA_VERSION",
    "ModelMetadata",
    "ModelParameter",
    "OUTPUT_SCHEMA_VERSION",
    "PersistenceAtom",
    "PersistenceDiagram",
    "RandomGraphClassificationData",
    "RandomGraphClassificationDataError",
    "RffCatalog",
    "RffCondition",
    "RffKey",
    "RobustnessCatalog",
    "RobustnessCondition",
    "RobustnessConditionKey",
    "RobustnessDiagramCollection",
    "RobustnessFeatureKey",
    "RobustnessGraph",
    "RobustnessGraphCollection",
    "RobustnessGraphDiagram",
    "RobustnessGraphKey",
    "ScalabilityCatalog",
    "ScalabilityCondition",
    "ScalabilityKey",
    "ScalabilityMeasurement",
    "SerializedEdge",
    "SerializedGraph",
    "VertexFieldCollection",
    "VertexFieldRecord",
    "load_ablation_catalog",
    "load_batch_size_catalog",
    "load_benchmark_catalog",
    "load_common_metadata",
    "load_random_graph_classification_data",
    "load_rff_catalog",
    "load_robustness_catalog",
    "load_scalability_catalog",
]