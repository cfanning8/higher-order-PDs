from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .configuration import (
    BENCHMARK_METHODS,
    EXPERIMENT_NAME,
    REAL_WORLD_DATA_ANALYSIS_SCHEMA_VERSION,
    RealWorldDataConfig,
)
from .data import RealWorldData
from .experiment import RealWorldExperimentResult


CONFIGURATION_FILENAME: Final[str] = "configuration.json"
OUTER_FOLDS_FILENAME: Final[str] = "outer_folds.csv"
PREDICTIONS_FILENAME: Final[str] = "predictions.csv"
SUMMARY_FILENAME: Final[str] = "summary.csv"


OUTER_FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "dataset",
    "method",
    "bag_size",
    "outer_fold",
    "train_graph_count",
    "test_graph_count",
    "train_bag_count",
    "test_bag_count",
    "selected_c",
    "selected_num_landscapes",
    "selected_landscape_resolution",
    "selected_image_resolution",
    "selected_image_bandwidth",
    "accuracy",
    "macro_f1",
)

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "dataset",
    "method",
    "bag_size",
    "outer_fold",
    "bag_index",
    "label",
    "prediction",
    "graph_indices",
)

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "dataset",
    "method",
    "bag_size",
    "test_bag_count",
    "outer_fold_count",
    "pooled_accuracy",
    "pooled_macro_f1",
    "confidence_level",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "fold_mean_accuracy",
    "fold_sd_accuracy",
    "fold_mean_macro_f1",
    "fold_sd_macro_f1",
)


@dataclass(
    frozen=True,
    slots=True,
)
class AnalysisOutputLayout:
    root: Path

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "root",
            Path(
                self.root
            ).expanduser().resolve(),
        )

    def configuration_json(
        self,
    ) -> Path:
        return (
            self.root
            /
            CONFIGURATION_FILENAME
        )

    def outer_folds_csv(
        self,
    ) -> Path:
        return (
            self.root
            /
            OUTER_FOLDS_FILENAME
        )

    def predictions_csv(
        self,
    ) -> Path:
        return (
            self.root
            /
            PREDICTIONS_FILENAME
        )

    def summary_csv(
        self,
    ) -> Path:
        return (
            self.root
            /
            SUMMARY_FILENAME
        )


def _csv_scalar(
    value: Any,
) -> str:
    if value is None:
        return ""

    if type(value) is bool:
        return (
            "true"
            if value
            else "false"
        )

    if type(value) is int:
        return str(
            value
        )

    if type(value) is float:
        if not math.isfinite(
            value
        ):
            raise ValueError(
                "CSV output requires finite floating-point values."
            )

        return format(
            0.0
            if value == 0.0
            else value,
            ".17g",
        )

    if isinstance(
        value,
        str,
    ):
        return value

    raise TypeError(
        "CSV output supports None, bool, int, float, and str."
    )


def _json_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if type(value) is bool:
        return value

    if type(value) is int:
        return value

    if type(value) is float:
        if not math.isfinite(
            value
        ):
            raise ValueError(
                "JSON output requires finite floating-point values."
            )

        return (
            0.0
            if value == 0.0
            else value
        )

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        tuple,
    ):
        return [
            _json_value(
                element
            )
            for element in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            _json_value(
                element
            )
            for element in value
        ]

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(
                key
            ): _json_value(
                element
            )
            for key, element in value.items()
        }

    raise TypeError(
        "JSON output contains unsupported type "
        f"{type(value).__name__}."
    )


def _atomic_write_text(
    path: Path,
    text: str,
) -> None:
    normalized = Path(
        path
    ).expanduser().resolve()

    normalized.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{normalized.name}.",
        suffix=".tmp",
        dir=normalized.parent,
    )

    temporary = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            stream.write(
                text
            )

            stream.flush()

            os.fsync(
                stream.fileno()
            )

        os.replace(
            temporary,
            normalized,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(
    path: Path,
    *,
    columns: Sequence[str],
    rows: Iterable[
        Mapping[str, Any]
    ],
    key_columns: Sequence[str],
    sort_key: Callable[
        [
            Mapping[str, Any]
        ],
        tuple[Any, ...],
    ],
) -> None:
    data_columns = tuple(
        columns
    )[
        1:
    ]

    expected = set(
        data_columns
    )

    normalized_rows: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[Any, ...]
    ] = set()

    for source_row in rows:
        if set(source_row) != expected:
            raise ValueError(
                "Output row columns do not match the declared "
                "schema."
            )

        row = {
            "schema_version": (
                REAL_WORLD_DATA_ANALYSIS_SCHEMA_VERSION
            ),
            **{
                column: source_row[
                    column
                ]
                for column in data_columns
            },
        }

        key = tuple(
            row[
                column
            ]
            for column in key_columns
        )

        if key in seen:
            raise ValueError(
                f"Output contains duplicate key {key!r}."
            )

        seen.add(
            key
        )

        normalized_rows.append(
            row
        )

    normalized_rows.sort(
        key=sort_key
    )

    buffer = io.StringIO(
        newline=""
    )

    writer = csv.DictWriter(
        buffer,
        fieldnames=list(
            columns
        ),
        extrasaction="raise",
        lineterminator="\n",
    )

    writer.writeheader()

    for row in normalized_rows:
        writer.writerow(
            {
                column: _csv_scalar(
                    row[
                        column
                    ]
                )
                for column in columns
            }
        )

    _atomic_write_text(
        path,
        buffer.getvalue(),
    )


def write_analysis_configuration(
    layout: AnalysisOutputLayout,
    data: RealWorldData,
    config: RealWorldDataConfig,
) -> None:
    payload = {
        "schema_version": (
            REAL_WORLD_DATA_ANALYSIS_SCHEMA_VERSION
        ),
        "experiment": EXPERIMENT_NAME,
        "raw_experiment": {
            "root": data.root,
            "tu_root": data.tu_root,
            "datasets": tuple(
                data.datasets
            ),
        },
        "methods": BENCHMARK_METHODS,
        "mil_design": {
            "graph_split_precedes_bag_construction": True,
            "bags_are_class_homogeneous": True,
            "bags_are_disjoint_within_partition": True,
            "incomplete_class_remainders_are_discarded": True,
            "class_permutation_shared_across_methods": True,
            "class_permutation_shared_across_bag_sizes": True,
            "graph_representations_are_aggregated_by_coordinatewise_mean": True,
            "perslay_training_is_partition_local": True,
        },
        "analysis_settings": asdict(
            config
        ),
    }

    normalized = _json_value(
        payload
    )

    _atomic_write_text(
        layout.configuration_json(),
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        +
        "\n",
    )


def _outer_fold_rows(
    result: RealWorldExperimentResult,
) -> list[
    dict[str, Any]
]:
    rows: list[
        dict[str, Any]
    ] = []

    for dataset_result in result.datasets.values():
        for method, bag_size_results in (
            dataset_result.methods.items()
        ):
            for bag_size_result in (
                bag_size_results.values()
            ):
                for fold in (
                    bag_size_result.outer_folds
                ):
                    rows.append(
                        {
                            "dataset": fold.dataset,
                            "method": method,
                            "bag_size": fold.bag_size,
                            "outer_fold": fold.outer_fold,
                            "train_graph_count": (
                                fold.train_graph_count
                            ),
                            "test_graph_count": (
                                fold.test_graph_count
                            ),
                            "train_bag_count": (
                                fold.train_bag_count
                            ),
                            "test_bag_count": (
                                fold.test_bag_count
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
                            "accuracy": fold.accuracy,
                            "macro_f1": fold.macro_f1,
                        }
                    )

    return rows


def _prediction_rows(
    result: RealWorldExperimentResult,
) -> list[
    dict[str, Any]
]:
    rows: list[
        dict[str, Any]
    ] = []

    for dataset_result in result.datasets.values():
        for method, bag_size_results in (
            dataset_result.methods.items()
        ):
            for bag_size_result in (
                bag_size_results.values()
            ):
                for prediction in (
                    bag_size_result.predictions
                ):
                    rows.append(
                        {
                            "dataset": prediction.dataset,
                            "method": method,
                            "bag_size": (
                                prediction.bag_size
                            ),
                            "outer_fold": (
                                prediction.outer_fold
                            ),
                            "bag_index": (
                                prediction.bag_index
                            ),
                            "label": prediction.label,
                            "prediction": (
                                prediction.prediction
                            ),
                            "graph_indices": "|".join(
                                str(
                                    graph_index
                                )
                                for graph_index in (
                                    prediction.graph_indices
                                )
                            ),
                        }
                    )

    return rows


def _summary_rows(
    result: RealWorldExperimentResult,
) -> list[
    dict[str, Any]
]:
    rows: list[
        dict[str, Any]
    ] = []

    for dataset_result in result.datasets.values():
        for method, bag_size_results in (
            dataset_result.methods.items()
        ):
            for bag_size_result in (
                bag_size_results.values()
            ):
                evaluation = (
                    bag_size_result.evaluation
                )

                pooled = (
                    evaluation.pooled_metrics
                )

                fold_summary = (
                    evaluation.fold_summary
                )

                accuracy_bootstrap = (
                    bag_size_result.bootstrap
                    .metrics
                    .accuracy
                )

                macro_f1_bootstrap = (
                    bag_size_result.bootstrap
                    .metrics
                    .macro_f1
                )

                rows.append(
                    {
                        "dataset": (
                            bag_size_result.dataset
                        ),
                        "method": method,
                        "bag_size": (
                            bag_size_result.bag_size
                        ),
                        "test_bag_count": (
                            evaluation.sample_count
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


def write_real_world_result(
    layout: AnalysisOutputLayout,
    result: RealWorldExperimentResult,
) -> None:
    layout.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_csv(
        layout.outer_folds_csv(),
        columns=OUTER_FOLD_COLUMNS,
        rows=_outer_fold_rows(
            result
        ),
        key_columns=(
            "dataset",
            "method",
            "bag_size",
            "outer_fold",
        ),
        sort_key=lambda row: (
            row[
                "dataset"
            ],
            row[
                "method"
            ],
            row[
                "bag_size"
            ],
            row[
                "outer_fold"
            ],
        ),
    )

    _write_csv(
        layout.predictions_csv(),
        columns=PREDICTION_COLUMNS,
        rows=_prediction_rows(
            result
        ),
        key_columns=(
            "dataset",
            "method",
            "bag_size",
            "outer_fold",
            "bag_index",
        ),
        sort_key=lambda row: (
            row[
                "dataset"
            ],
            row[
                "method"
            ],
            row[
                "bag_size"
            ],
            row[
                "outer_fold"
            ],
            row[
                "bag_index"
            ],
        ),
    )

    _write_csv(
        layout.summary_csv(),
        columns=SUMMARY_COLUMNS,
        rows=_summary_rows(
            result
        ),
        key_columns=(
            "dataset",
            "method",
            "bag_size",
        ),
        sort_key=lambda row: (
            row[
                "dataset"
            ],
            row[
                "method"
            ],
            row[
                "bag_size"
            ],
        ),
    )


__all__ = [
    "AnalysisOutputLayout",
    "write_analysis_configuration",
    "write_real_world_result",
]