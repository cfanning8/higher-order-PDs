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
    BATCH_SIZE_SENSITIVITY_EXPERIMENT,
    BENCHMARK_EXPERIMENT,
    BENCHMARK_METHODS,
    HARMONIC_AGGREGATION_ABLATION_EXPERIMENT,
    HARMONIC_AGGREGATION_ABLATION_KINDS,
    RFF_SENSITIVITY_EXPERIMENT,
    ROBUSTNESS_EXPERIMENT,
    SCALABILITY_EXPERIMENT,
    SUPPORTED_EXPERIMENTS,
    RandomGraphClassificationConfig,
)
from .data import (
    METADATA_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    CommonMetadata,
)


RANDOM_GRAPH_CLASSIFICATION_ANALYSIS_SCHEMA_VERSION: Final[int] = 2

ANALYSIS_DIRECTORY_NAME: Final[str] = "random_graph_classification"
ANALYSIS_CONFIGURATION_FILENAME: Final[str] = "configuration.json"

OUTER_FOLDS_FILENAME: Final[str] = "outer_folds.csv"
SUMMARY_FILENAME: Final[str] = "summary.csv"
ABLATION_KIND_SUMMARY_FILENAME: Final[str] = "kind_summary.csv"
RFF_REPLICATE_SUMMARY_FILENAME: Final[str] = "replicate_summary.csv"
SCALABILITY_MEASUREMENTS_FILENAME: Final[str] = "measurements.csv"


BENCHMARK_OUTER_FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "method",
    "outer_fold",
    "train_sample_count",
    "test_sample_count",
    "selected_c",
    "selected_num_landscapes",
    "selected_landscape_resolution",
    "selected_image_resolution",
    "selected_image_bandwidth",
    "accuracy",
    "macro_f1",
)

BENCHMARK_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "method",
    "sample_count",
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

ABLATION_OUTER_FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "ablation_kind",
    "realization_index",
    "outer_fold",
    "train_sample_count",
    "test_sample_count",
    "selected_c",
    "accuracy",
    "macro_f1",
    "baseline_accuracy",
    "baseline_macro_f1",
    "accuracy_difference",
    "macro_f1_difference",
)

ABLATION_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "ablation_kind",
    "realization_index",
    "outer_fold_count",
    "pooled_accuracy",
    "pooled_macro_f1",
    "accuracy_difference",
    "macro_f1_difference",
    "confidence_level",
    "accuracy_difference_ci_low",
    "accuracy_difference_ci_high",
    "macro_f1_difference_ci_low",
    "macro_f1_difference_ci_high",
    "fold_mean_accuracy",
    "fold_sd_accuracy",
    "fold_mean_macro_f1",
    "fold_sd_macro_f1",
)

ABLATION_KIND_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "ablation_kind",
    "realization_count",
    "mean_accuracy",
    "mean_macro_f1",
    "accuracy_difference",
    "macro_f1_difference",
    "confidence_level",
    "accuracy_difference_ci_low",
    "accuracy_difference_ci_high",
    "macro_f1_difference_ci_low",
    "macro_f1_difference_ci_high",
    "uncertainty_method",
)

BATCH_SIZE_OUTER_FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "graphs_per_batch",
    "outer_fold",
    "train_sample_count",
    "test_sample_count",
    "selected_c",
    "accuracy",
    "macro_f1",
)

BATCH_SIZE_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "graphs_per_batch",
    "is_reference",
    "outer_fold_count",
    "pooled_accuracy",
    "pooled_macro_f1",
    "confidence_level",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "accuracy_difference_from_reference",
    "accuracy_difference_ci_low",
    "accuracy_difference_ci_high",
    "macro_f1_difference_from_reference",
    "macro_f1_difference_ci_low",
    "macro_f1_difference_ci_high",
    "fold_mean_accuracy",
    "fold_sd_accuracy",
    "fold_mean_macro_f1",
    "fold_sd_macro_f1",
)

RFF_OUTER_FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "character_count",
    "replicate_index",
    "outer_fold",
    "train_sample_count",
    "test_sample_count",
    "selected_c",
    "accuracy",
    "macro_f1",
)

RFF_REPLICATE_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "character_count",
    "replicate_index",
    "outer_fold_count",
    "pooled_accuracy",
    "pooled_macro_f1",
    "fold_mean_accuracy",
    "fold_sd_accuracy",
    "fold_mean_macro_f1",
    "fold_sd_macro_f1",
)

RFF_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "character_count",
    "replicate_count",
    "mean_accuracy",
    "mean_macro_f1",
    "confidence_level",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "accuracy_difference_from_baseline",
    "accuracy_difference_ci_low",
    "accuracy_difference_ci_high",
    "macro_f1_difference_from_baseline",
    "macro_f1_difference_ci_low",
    "macro_f1_difference_ci_high",
    "uncertainty_method",
)

ROBUSTNESS_OUTER_FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "condition_index",
    "perturbation",
    "severity",
    "realization_index",
    "outer_fold",
    "train_sample_count",
    "test_sample_count",
    "selected_c",
    "accuracy",
    "macro_f1",
    "baseline_accuracy",
    "baseline_macro_f1",
    "accuracy_difference",
    "macro_f1_difference",
)

ROBUSTNESS_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "condition_index",
    "perturbation",
    "severity",
    "realization_count",
    "mean_accuracy",
    "mean_macro_f1",
    "confidence_level",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "accuracy_difference",
    "accuracy_difference_ci_low",
    "accuracy_difference_ci_high",
    "macro_f1_difference",
    "macro_f1_difference_ci_low",
    "macro_f1_difference_ci_high",
    "uncertainty_method",
)

SCALABILITY_MEASUREMENT_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "scalability_condition_index",
    "graphs_per_batch",
    "input_support_size",
    "character_count",
    "repeat_index",
    "sample_seed",
    "character_seed",
    "representation_seconds",
)

SCALABILITY_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "graphs_per_batch",
    "input_support_size",
    "character_count",
    "repeat_count",
    "mean_representation_seconds",
    "median_representation_seconds",
    "standard_deviation_representation_seconds",
)


def _require_experiment(
    experiment: str,
) -> str:
    if experiment not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            f"Unsupported experiment {experiment!r}."
        )

    return experiment


def _normalized_path(
    path: str | Path,
) -> Path:
    return Path(
        path
    ).expanduser().resolve()


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
            _normalized_path(
                self.root
            ),
        )

    def configuration_json(
        self,
    ) -> Path:
        return (
            self.root
            / ANALYSIS_CONFIGURATION_FILENAME
        )

    def experiment_directory(
        self,
        experiment: str,
    ) -> Path:
        return (
            self.root
            / _require_experiment(
                experiment
            )
        )

    def outer_folds_csv(
        self,
        experiment: str,
    ) -> Path:
        return (
            self.experiment_directory(
                experiment
            )
            / OUTER_FOLDS_FILENAME
        )

    def summary_csv(
        self,
        experiment: str,
    ) -> Path:
        return (
            self.experiment_directory(
                experiment
            )
            / SUMMARY_FILENAME
        )

    def ablation_kind_summary_csv(
        self,
    ) -> Path:
        return (
            self.experiment_directory(
                HARMONIC_AGGREGATION_ABLATION_EXPERIMENT
            )
            / ABLATION_KIND_SUMMARY_FILENAME
        )

    def rff_replicate_summary_csv(
        self,
    ) -> Path:
        return (
            self.experiment_directory(
                RFF_SENSITIVITY_EXPERIMENT
            )
            / RFF_REPLICATE_SUMMARY_FILENAME
        )

    def scalability_measurements_csv(
        self,
    ) -> Path:
        return (
            self.experiment_directory(
                SCALABILITY_EXPERIMENT
            )
            / SCALABILITY_MEASUREMENTS_FILENAME
        )

    def create_directories(
        self,
        enabled_experiments: Sequence[str],
    ) -> None:
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for experiment in dict.fromkeys(
            enabled_experiments
        ):
            self.experiment_directory(
                experiment
            ).mkdir(
                parents=True,
                exist_ok=True,
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
        "JSON output contains an unsupported value of type "
        f"{type(value).__name__}."
    )


def _atomic_write_text(
    path: Path,
    text: str,
) -> None:
    normalized_path = _normalized_path(
        path
    )

    normalized_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{normalized_path.name}.",
        suffix=".tmp",
        dir=normalized_path.parent,
    )

    temporary_path = Path(
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
            temporary_path,
            normalized_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _rows_with_schema_version(
    rows: Iterable[
        Mapping[str, Any]
    ],
    *,
    data_columns: Sequence[str],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    expected_columns = set(
        data_columns
    )

    normalized_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for row in rows:
        if set(
            row
        ) != expected_columns:
            raise ValueError(
                "Output row columns do not match the declared "
                "output schema."
            )

        normalized_rows.append(
            {
                "schema_version": (
                    RANDOM_GRAPH_CLASSIFICATION_ANALYSIS_SCHEMA_VERSION
                ),
                **{
                    column: row[
                        column
                    ]
                    for column in data_columns
                },
            }
        )

    return normalized_rows


def _reject_duplicate_keys(
    rows: Sequence[
        Mapping[str, Any]
    ],
    *,
    key_columns: Sequence[str],
    description: str,
) -> None:
    seen: set[
        tuple[
            Any,
            ...,
        ]
    ] = set()

    for row in rows:
        key = tuple(
            row[
                column
            ]
            for column in key_columns
        )

        if key in seen:
            raise ValueError(
                f"{description} contains duplicate key {key!r}."
            )

        seen.add(
            key
        )


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Iterable[
        Mapping[str, Any]
    ],
    sort_key: Callable[
        [
            Mapping[str, Any]
        ],
        tuple[
            Any,
            ...,
        ],
    ],
) -> None:
    fields = tuple(
        fieldnames
    )

    materialized = [
        dict(
            row
        )
        for row in rows
    ]

    materialized.sort(
        key=sort_key
    )

    buffer = io.StringIO(
        newline=""
    )

    writer = csv.DictWriter(
        buffer,
        fieldnames=list(
            fields
        ),
        extrasaction="raise",
        lineterminator="\n",
    )

    writer.writeheader()

    for row in materialized:
        writer.writerow(
            {
                field: _csv_scalar(
                    row[
                        field
                    ]
                )
                for field in fields
            }
        )

    _atomic_write_text(
        path,
        buffer.getvalue(),
    )


def _write_study_csv(
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
        tuple[
            Any,
            ...,
        ],
    ],
    description: str,
) -> None:
    normalized = _rows_with_schema_version(
        rows,
        data_columns=tuple(
            columns
        )[
            1:
        ],
    )

    _reject_duplicate_keys(
        normalized,
        key_columns=key_columns,
        description=description,
    )

    _write_csv(
        path,
        fieldnames=columns,
        rows=normalized,
        sort_key=sort_key,
    )


def _benchmark_method_rank(
    method: Any,
) -> int:
    return BENCHMARK_METHODS.index(
        str(
            method
        )
    )


def _ablation_kind_rank(
    ablation_kind: Any,
) -> int:
    return (
        HARMONIC_AGGREGATION_ABLATION_KINDS.index(
            str(
                ablation_kind
            )
        )
    )


def _configuration_payload(
    common: CommonMetadata,
    config: RandomGraphClassificationConfig,
) -> dict[
    str,
    Any,
]:
    return {
        "schema_version": (
            RANDOM_GRAPH_CLASSIFICATION_ANALYSIS_SCHEMA_VERSION
        ),
        "experiment": ANALYSIS_DIRECTORY_NAME,
        "raw_experiment": {
            "root": common.root,
            "metadata_schema_version": (
                METADATA_SCHEMA_VERSION
            ),
            "output_schema_version": (
                OUTPUT_SCHEMA_VERSION
            ),
            "manifest_created_utc": (
                common.manifest.created_utc
            ),
            "manifest_task_count": (
                common.manifest.task_count
            ),
            "manifest_studies": (
                common.manifest.studies
            ),
            "build": asdict(
                common.build
            ),
        },
        "analysis_settings": asdict(
            config
        ),
    }


def write_analysis_configuration(
    layout: AnalysisOutputLayout,
    common: CommonMetadata,
    config: RandomGraphClassificationConfig,
) -> None:
    payload = _json_value(
        _configuration_payload(
            common,
            config,
        )
    )

    _atomic_write_text(
        layout.configuration_json(),
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def write_benchmark_outer_folds(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.outer_folds_csv(
            BENCHMARK_EXPERIMENT
        ),
        columns=BENCHMARK_OUTER_FOLD_COLUMNS,
        rows=rows,
        key_columns=(
            "method",
            "outer_fold",
        ),
        sort_key=lambda row: (
            _benchmark_method_rank(
                row[
                    "method"
                ]
            ),
            row[
                "outer_fold"
            ],
        ),
        description="Benchmark outer-fold output",
    )


def write_benchmark_summary(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.summary_csv(
            BENCHMARK_EXPERIMENT
        ),
        columns=BENCHMARK_SUMMARY_COLUMNS,
        rows=rows,
        key_columns=(
            "method",
        ),
        sort_key=lambda row: (
            _benchmark_method_rank(
                row[
                    "method"
                ]
            ),
        ),
        description="Benchmark summary output",
    )


def write_ablation_outer_folds(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.outer_folds_csv(
            HARMONIC_AGGREGATION_ABLATION_EXPERIMENT
        ),
        columns=ABLATION_OUTER_FOLD_COLUMNS,
        rows=rows,
        key_columns=(
            "ablation_kind",
            "realization_index",
            "outer_fold",
        ),
        sort_key=lambda row: (
            _ablation_kind_rank(
                row[
                    "ablation_kind"
                ]
            ),
            row[
                "realization_index"
            ],
            row[
                "outer_fold"
            ],
        ),
        description="Harmonic-aggregation ablation outer-fold output",
    )


def write_ablation_summary(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.summary_csv(
            HARMONIC_AGGREGATION_ABLATION_EXPERIMENT
        ),
        columns=ABLATION_SUMMARY_COLUMNS,
        rows=rows,
        key_columns=(
            "ablation_kind",
            "realization_index",
        ),
        sort_key=lambda row: (
            _ablation_kind_rank(
                row[
                    "ablation_kind"
                ]
            ),
            row[
                "realization_index"
            ],
        ),
        description="Harmonic-aggregation ablation summary output",
    )


def write_ablation_kind_summary(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.ablation_kind_summary_csv(),
        columns=ABLATION_KIND_SUMMARY_COLUMNS,
        rows=rows,
        key_columns=(
            "ablation_kind",
        ),
        sort_key=lambda row: (
            _ablation_kind_rank(
                row[
                    "ablation_kind"
                ]
            ),
        ),
        description="Harmonic-aggregation ablation kind summary output",
    )


def write_batch_size_outer_folds(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.outer_folds_csv(
            BATCH_SIZE_SENSITIVITY_EXPERIMENT
        ),
        columns=BATCH_SIZE_OUTER_FOLD_COLUMNS,
        rows=rows,
        key_columns=(
            "graphs_per_batch",
            "outer_fold",
        ),
        sort_key=lambda row: (
            row[
                "graphs_per_batch"
            ],
            row[
                "outer_fold"
            ],
        ),
        description="Batch-size sensitivity outer-fold output",
    )


def write_batch_size_summary(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.summary_csv(
            BATCH_SIZE_SENSITIVITY_EXPERIMENT
        ),
        columns=BATCH_SIZE_SUMMARY_COLUMNS,
        rows=rows,
        key_columns=(
            "graphs_per_batch",
        ),
        sort_key=lambda row: (
            row[
                "graphs_per_batch"
            ],
        ),
        description="Batch-size sensitivity summary output",
    )


def write_rff_outer_folds(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.outer_folds_csv(
            RFF_SENSITIVITY_EXPERIMENT
        ),
        columns=RFF_OUTER_FOLD_COLUMNS,
        rows=rows,
        key_columns=(
            "character_count",
            "replicate_index",
            "outer_fold",
        ),
        sort_key=lambda row: (
            row[
                "character_count"
            ],
            row[
                "replicate_index"
            ],
            row[
                "outer_fold"
            ],
        ),
        description="RFF sensitivity outer-fold output",
    )


def write_rff_replicate_summary(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.rff_replicate_summary_csv(),
        columns=RFF_REPLICATE_SUMMARY_COLUMNS,
        rows=rows,
        key_columns=(
            "character_count",
            "replicate_index",
        ),
        sort_key=lambda row: (
            row[
                "character_count"
            ],
            row[
                "replicate_index"
            ],
        ),
        description="RFF sensitivity replicate summary output",
    )


def write_rff_summary(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.summary_csv(
            RFF_SENSITIVITY_EXPERIMENT
        ),
        columns=RFF_SUMMARY_COLUMNS,
        rows=rows,
        key_columns=(
            "character_count",
        ),
        sort_key=lambda row: (
            row[
                "character_count"
            ],
        ),
        description="RFF sensitivity summary output",
    )


def write_robustness_outer_folds(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.outer_folds_csv(
            ROBUSTNESS_EXPERIMENT
        ),
        columns=ROBUSTNESS_OUTER_FOLD_COLUMNS,
        rows=rows,
        key_columns=(
            "condition_index",
            "realization_index",
            "outer_fold",
        ),
        sort_key=lambda row: (
            row[
                "condition_index"
            ],
            row[
                "realization_index"
            ],
            row[
                "outer_fold"
            ],
        ),
        description="Robustness outer-fold output",
    )


def write_robustness_summary(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.summary_csv(
            ROBUSTNESS_EXPERIMENT
        ),
        columns=ROBUSTNESS_SUMMARY_COLUMNS,
        rows=rows,
        key_columns=(
            "condition_index",
        ),
        sort_key=lambda row: (
            row[
                "condition_index"
            ],
        ),
        description="Robustness summary output",
    )


def write_scalability_measurements(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.scalability_measurements_csv(),
        columns=SCALABILITY_MEASUREMENT_COLUMNS,
        rows=rows,
        key_columns=(
            "scalability_condition_index",
        ),
        sort_key=lambda row: (
            row[
                "scalability_condition_index"
            ],
        ),
        description="Scalability measurement output",
    )


def write_scalability_summary(
    layout: AnalysisOutputLayout,
    rows: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    _write_study_csv(
        layout.summary_csv(
            SCALABILITY_EXPERIMENT
        ),
        columns=SCALABILITY_SUMMARY_COLUMNS,
        rows=rows,
        key_columns=(
            "graphs_per_batch",
            "input_support_size",
            "character_count",
        ),
        sort_key=lambda row: (
            row[
                "input_support_size"
            ],
            row[
                "character_count"
            ],
            row[
                "graphs_per_batch"
            ],
        ),
        description="Scalability summary output",
    )


__all__ = [
    "ANALYSIS_DIRECTORY_NAME",
    "AnalysisOutputLayout",
    "RANDOM_GRAPH_CLASSIFICATION_ANALYSIS_SCHEMA_VERSION",
    "write_ablation_kind_summary",
    "write_ablation_outer_folds",
    "write_ablation_summary",
    "write_analysis_configuration",
    "write_batch_size_outer_folds",
    "write_batch_size_summary",
    "write_benchmark_outer_folds",
    "write_benchmark_summary",
    "write_rff_outer_folds",
    "write_rff_replicate_summary",
    "write_rff_summary",
    "write_robustness_outer_folds",
    "write_robustness_summary",
    "write_scalability_measurements",
    "write_scalability_summary",
]