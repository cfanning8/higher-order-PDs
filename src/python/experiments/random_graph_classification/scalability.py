from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import numpy as np

from .data import ScalabilityCatalog
from .output import (
    AnalysisOutputLayout,
    write_scalability_measurements,
    write_scalability_summary,
)


ScalabilityConditionKey: TypeAlias = tuple[
    int,
    int,
    int,
]


@dataclass(
    frozen=True,
    slots=True,
)
class ScalabilityMeasurement:
    scalability_condition_index: int
    graphs_per_batch: int
    input_support_size: int
    character_count: int
    repeat_index: int
    sample_seed: int
    character_seed: int
    representation_seconds: float


@dataclass(
    frozen=True,
    slots=True,
)
class ScalabilityConditionSummary:
    graphs_per_batch: int
    input_support_size: int
    character_count: int
    repeat_count: int
    mean_representation_seconds: float
    median_representation_seconds: float
    standard_deviation_representation_seconds: float


@dataclass(
    frozen=True,
    slots=True,
)
class ScalabilityResult:
    measurements: tuple[
        ScalabilityMeasurement,
        ...,
    ]
    summaries: Mapping[
        ScalabilityConditionKey,
        ScalabilityConditionSummary,
    ]


def _scalability_measurements(
    catalog: ScalabilityCatalog,
) -> tuple[
    ScalabilityMeasurement,
    ...,
]:
    measurements = [
        ScalabilityMeasurement(
            scalability_condition_index=(
                measurement
                .condition
                .scalability_condition_index
            ),
            graphs_per_batch=(
                measurement
                .condition
                .graphs_per_batch
            ),
            input_support_size=(
                measurement
                .condition
                .input_support_size
            ),
            character_count=(
                measurement
                .condition
                .character_count
            ),
            repeat_index=(
                measurement
                .condition
                .repeat_index
            ),
            sample_seed=(
                measurement
                .condition
                .sample_seed
            ),
            character_seed=(
                measurement
                .condition
                .character_seed
            ),
            representation_seconds=(
                measurement
                .representation_seconds
            ),
        )
        for measurement in (
            catalog.measurements.values()
        )
    ]

    measurements.sort(
        key=lambda measurement: (
            measurement.graphs_per_batch,
            measurement.input_support_size,
            measurement.character_count,
            measurement.repeat_index,
        )
    )

    return tuple(
        measurements
    )


def _summarize_scalability_condition(
    measurements: tuple[
        ScalabilityMeasurement,
        ...,
    ],
) -> ScalabilityConditionSummary:
    repeat_count = len(
        measurements
    )

    timing_values = np.asarray(
        tuple(
            measurement.representation_seconds
            for measurement in measurements
        ),
        dtype=np.float64,
    )

    first = measurements[
        0
    ]

    return ScalabilityConditionSummary(
        graphs_per_batch=(
            first.graphs_per_batch
        ),
        input_support_size=(
            first.input_support_size
        ),
        character_count=(
            first.character_count
        ),
        repeat_count=repeat_count,
        mean_representation_seconds=(
            math.fsum(
                measurement.representation_seconds
                for measurement in measurements
            )
            /
            repeat_count
        ),
        median_representation_seconds=float(
            np.median(
                timing_values
            )
        ),
        standard_deviation_representation_seconds=float(
            np.std(
                timing_values,
                ddof=1,
            )
        ),
    )


def _scalability_summaries(
    measurements: tuple[
        ScalabilityMeasurement,
        ...,
    ],
) -> Mapping[
    ScalabilityConditionKey,
    ScalabilityConditionSummary,
]:
    grouped: dict[
        ScalabilityConditionKey,
        list[
            ScalabilityMeasurement
        ],
    ] = {}

    for measurement in measurements:
        key: ScalabilityConditionKey = (
            measurement.graphs_per_batch,
            measurement.input_support_size,
            measurement.character_count,
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            measurement
        )

    return MappingProxyType(
        {
            key: _summarize_scalability_condition(
                tuple(
                    grouped[
                        key
                    ]
                )
            )
            for key in sorted(
                grouped
            )
        }
    )


def run_scalability(
    catalog: ScalabilityCatalog,
) -> ScalabilityResult:
    measurements = _scalability_measurements(
        catalog
    )

    summaries = _scalability_summaries(
        measurements
    )

    return ScalabilityResult(
        measurements=measurements,
        summaries=summaries,
    )


def scalability_measurement_rows(
    result: ScalabilityResult,
) -> list[
    dict[
        str,
        object,
    ]
]:
    return [
        {
            "scalability_condition_index": (
                measurement
                .scalability_condition_index
            ),
            "graphs_per_batch": (
                measurement.graphs_per_batch
            ),
            "input_support_size": (
                measurement.input_support_size
            ),
            "character_count": (
                measurement.character_count
            ),
            "repeat_index": (
                measurement.repeat_index
            ),
            "sample_seed": (
                measurement.sample_seed
            ),
            "character_seed": (
                measurement.character_seed
            ),
            "representation_seconds": (
                measurement.representation_seconds
            ),
        }
        for measurement in result.measurements
    ]


def scalability_summary_rows(
    result: ScalabilityResult,
) -> list[
    dict[
        str,
        object,
    ]
]:
    return [
        {
            "graphs_per_batch": (
                summary.graphs_per_batch
            ),
            "input_support_size": (
                summary.input_support_size
            ),
            "character_count": (
                summary.character_count
            ),
            "repeat_count": (
                summary.repeat_count
            ),
            "mean_representation_seconds": (
                summary.mean_representation_seconds
            ),
            "median_representation_seconds": (
                summary.median_representation_seconds
            ),
            "standard_deviation_representation_seconds": (
                summary
                .standard_deviation_representation_seconds
            ),
        }
        for summary in result.summaries.values()
    ]


def write_scalability_result(
    layout: AnalysisOutputLayout,
    result: ScalabilityResult,
) -> None:
    write_scalability_measurements(
        layout,
        scalability_measurement_rows(
            result
        ),
    )

    write_scalability_summary(
        layout,
        scalability_summary_rows(
            result
        ),
    )


__all__ = [
    "ScalabilityConditionSummary",
    "ScalabilityMeasurement",
    "ScalabilityResult",
    "run_scalability",
    "scalability_measurement_rows",
    "scalability_summary_rows",
    "write_scalability_result",
]