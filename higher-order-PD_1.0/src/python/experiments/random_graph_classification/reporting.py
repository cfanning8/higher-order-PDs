from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from core.plotting import (
    COLOR_BLUE,
    COLOR_GRAY,
    DEFAULT_BAND_ALPHA,
    DEFAULT_COLOR_CYCLE,
    DEFAULT_MARKERS,
    DOUBLE_COLUMN_WIDTH_INCHES,
    SINGLE_COLUMN_WIDTH_INCHES,
    add_panel_label,
    add_reference_line,
    create_figure,
    create_subplots,
    plot_interval_band,
    save_figure,
    style_axis,
)

from .batch_size_sensitivity import (
    BatchSizeSensitivityResult,
    batch_size_summary_rows,
)
from .benchmark import (
    BenchmarkResult,
    benchmark_summary_rows,
)
from .configuration import (
    CROSS_OBSERVATION_ABLATION,
    HARMONIC_AGGREGATION_ABLATION_KINDS,
    LINEAR_ABLATION,
    NO_PREORDER_ABLATION,
)
from .harmonic_aggregation_ablation import (
    HarmonicAggregationAblationResult,
    ablation_kind_summary_rows,
)
from .rff_sensitivity import (
    RffSensitivityResult,
    rff_summary_rows,
)
from .robustness import (
    RobustnessResult,
    robustness_summary_rows,
)
from .scalability import (
    ScalabilityResult,
    scalability_summary_rows,
)


_BENCHMARK_FIGURE_FILENAME: Final[str] = (
    "benchmark_accuracy.pdf"
)
_ABLATION_FIGURE_FILENAME: Final[str] = (
    "harmonic_aggregation_ablation.pdf"
)
_BATCH_SIZE_FIGURE_FILENAME: Final[str] = (
    "batch_size_sensitivity.pdf"
)
_RFF_FIGURE_FILENAME: Final[str] = (
    "rff_sensitivity.pdf"
)
_ROBUSTNESS_FIGURE_FILENAME: Final[str] = (
    "robustness.pdf"
)
_SCALABILITY_FIGURE_FILENAME: Final[str] = (
    "scalability.pdf"
)

_BENCHMARK_METHOD_LABELS: Final[
    Mapping[str, str]
] = {
    "graph_statistics": "Graph statistics",
    "persistence_landscape": "Persistence landscape",
    "persistence_image": "Persistence image",
    "harmonic_aggregation": "Harmonic aggregation",
}

_ABLATION_KIND_LABELS: Final[
    Mapping[str, str]
] = {
    CROSS_OBSERVATION_ABLATION: "Cross-observation",
    LINEAR_ABLATION: "Linear",
    NO_PREORDER_ABLATION: "No preorder",
}

_ROBUSTNESS_PERTURBATIONS: Final[
    tuple[
        str,
        ...,
    ]
] = (
    "edge_deletion",
    "edge_insertion",
    "degree_preserving_rewiring",
    "filtration_noise",
)

_ROBUSTNESS_PANEL_LABELS: Final[
    Mapping[
        str,
        str,
    ]
] = {
    "edge_deletion": "Deleted edge fraction",
    "edge_insertion": "Inserted edge fraction",
    "degree_preserving_rewiring": "Rewired edge fraction",
    "filtration_noise": "Filtration-noise scale",
}


def _float_values(
    rows: Sequence[
        Mapping[
            str,
            object,
        ]
    ],
    column: str,
) -> np.ndarray:
    return np.asarray(
        tuple(
            float(
                row[
                    column
                ]
            )
            for row in rows
        ),
        dtype=np.float64,
    )


def _plot_interval_series(
    axis: Axes,
    rows: Sequence[
        Mapping[
            str,
            object,
        ]
    ],
    *,
    x_column: str,
    center_column: str,
    lower_column: str,
    upper_column: str,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
    label: str | None = None,
) -> None:
    x = _float_values(
        rows,
        x_column,
    )

    center = _float_values(
        rows,
        center_column,
    )

    lower = _float_values(
        rows,
        lower_column,
    )

    upper = _float_values(
        rows,
        upper_column,
    )

    plot_interval_band(
        axis,
        x,
        lower,
        upper,
        color=color,
        alpha=DEFAULT_BAND_ALPHA,
        zorder=1.0,
    )

    axis.plot(
        x,
        center,
        color=color,
        marker=marker,
        linestyle=linestyle,
        label=label,
        zorder=2.0,
    )


def _plot_discrete_intervals(
    axis: Axes,
    x: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    color: str,
) -> None:
    axis.vlines(
        x,
        lower,
        upper,
        color=color,
        linewidth=1.1,
        zorder=1.0,
    )

    axis.scatter(
        x,
        center,
        color=color,
        marker="o",
        s=25.0,
        zorder=2.0,
    )


def _save_reporting_figure(
    figure: Figure,
    figure_directory: str | Path,
    filename: str,
    *,
    overwrite: bool,
) -> Path:
    directory = Path(
        figure_directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return save_figure(
        figure,
        directory
        / filename,
        overwrite=overwrite,
        close=True,
    )


def write_benchmark_figure(
    figure_directory: str | Path,
    result: BenchmarkResult,
    *,
    overwrite: bool = False,
) -> Path:
    rows = benchmark_summary_rows(
        result
    )

    figure, axis = create_figure(
        width=SINGLE_COLUMN_WIDTH_INCHES,
        aspect_ratio=1.25,
    )

    x = np.arange(
        len(
            rows
        ),
        dtype=np.float64,
    )

    _plot_discrete_intervals(
        axis,
        x,
        _float_values(
            rows,
            "pooled_accuracy",
        ),
        _float_values(
            rows,
            "accuracy_ci_low",
        ),
        _float_values(
            rows,
            "accuracy_ci_high",
        ),
        color=COLOR_BLUE,
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            _BENCHMARK_METHOD_LABELS[
                str(
                    row[
                        "method"
                    ]
                )
            ]
            for row in rows
        ],
        rotation=22.5,
        ha="right",
    )

    chance_accuracy = (
        1.0
        /
        np.unique(
            result.labels
        ).size
    )

    add_reference_line(
        axis,
        chance_accuracy,
        orientation="horizontal",
        color=COLOR_GRAY,
        linestyle="--",
        linewidth=1.0,
    )

    style_axis(
        axis,
        ylabel="Accuracy",
        ygrid=True,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    return _save_reporting_figure(
        figure,
        figure_directory,
        _BENCHMARK_FIGURE_FILENAME,
        overwrite=overwrite,
    )


def write_ablation_figure(
    figure_directory: str | Path,
    result: HarmonicAggregationAblationResult,
    *,
    overwrite: bool = False,
) -> Path:
    rows_by_kind = {
        str(
            row[
                "ablation_kind"
            ]
        ): row
        for row in ablation_kind_summary_rows(
            result
        )
    }

    rows = tuple(
        rows_by_kind[
            ablation_kind
        ]
        for ablation_kind in (
            HARMONIC_AGGREGATION_ABLATION_KINDS
        )
    )

    figure, axis = create_figure(
        width=SINGLE_COLUMN_WIDTH_INCHES,
        aspect_ratio=1.25,
    )

    x = np.arange(
        len(
            rows
        ),
        dtype=np.float64,
    )

    _plot_discrete_intervals(
        axis,
        x,
        _float_values(
            rows,
            "accuracy_difference",
        ),
        _float_values(
            rows,
            "accuracy_difference_ci_low",
        ),
        _float_values(
            rows,
            "accuracy_difference_ci_high",
        ),
        color=COLOR_BLUE,
    )

    add_reference_line(
        axis,
        0.0,
        orientation="horizontal",
        color=COLOR_GRAY,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            _ABLATION_KIND_LABELS[
                ablation_kind
            ]
            for ablation_kind in (
                HARMONIC_AGGREGATION_ABLATION_KINDS
            )
        ],
        rotation=15.0,
        ha="right",
    )

    style_axis(
        axis,
        xlabel="Harmonic-aggregation ablation",
        ylabel=(
            "Accuracy difference "
            "(ablation - baseline)"
        ),
        ygrid=True,
    )

    return _save_reporting_figure(
        figure,
        figure_directory,
        _ABLATION_FIGURE_FILENAME,
        overwrite=overwrite,
    )


def write_batch_size_figure(
    figure_directory: str | Path,
    result: BatchSizeSensitivityResult,
    *,
    overwrite: bool = False,
) -> Path:
    rows = batch_size_summary_rows(
        result
    )

    reference_graphs_per_batch = (
        result.reference_graphs_per_batch
    )

    comparison_rows = tuple(
        row
        for row in rows
        if int(
            row[
                "graphs_per_batch"
            ]
        )
        != reference_graphs_per_batch
    )

    figure, axis = create_figure(
        width=SINGLE_COLUMN_WIDTH_INCHES,
    )

    if comparison_rows:
        _plot_interval_series(
            axis,
            comparison_rows,
            x_column="graphs_per_batch",
            center_column=(
                "accuracy_difference_from_reference"
            ),
            lower_column="accuracy_difference_ci_low",
            upper_column="accuracy_difference_ci_high",
            color=COLOR_BLUE,
        )

    axis.scatter(
        np.asarray(
            [
                reference_graphs_per_batch
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                0.0
            ],
            dtype=np.float64,
        ),
        color=COLOR_BLUE,
        marker="o",
        s=25.0,
        zorder=3.0,
    )

    add_reference_line(
        axis,
        0.0,
        orientation="horizontal",
        color=COLOR_GRAY,
        linestyle="--",
        linewidth=1.0,
    )

    style_axis(
        axis,
        xlabel="Graphs per batch",
        ylabel="Accuracy difference from reference",
        ygrid=True,
    )

    axis.set_xticks(
        _float_values(
            rows,
            "graphs_per_batch",
        )
    )

    return _save_reporting_figure(
        figure,
        figure_directory,
        _BATCH_SIZE_FIGURE_FILENAME,
        overwrite=overwrite,
    )


def write_rff_sensitivity_figure(
    figure_directory: str | Path,
    result: RffSensitivityResult,
    *,
    overwrite: bool = False,
) -> Path:
    rows = rff_summary_rows(
        result
    )

    figure, axes = create_subplots(
        1,
        2,
        width=DOUBLE_COLUMN_WIDTH_INCHES,
        aspect_ratio=2.0,
        constrained_layout=True,
    )

    flat_axes = np.asarray(
        axes,
        dtype=object,
    ).reshape(
        -1
    )

    absolute_axis = flat_axes[
        0
    ]

    difference_axis = flat_axes[
        1
    ]

    _plot_interval_series(
        absolute_axis,
        rows,
        x_column="character_count",
        center_column="mean_accuracy",
        lower_column="accuracy_ci_low",
        upper_column="accuracy_ci_high",
        color=COLOR_BLUE,
    )

    add_reference_line(
        absolute_axis,
        (
            result.baseline
            .evaluation
            .pooled_metrics
            .accuracy
        ),
        orientation="horizontal",
        color=COLOR_GRAY,
        linestyle="--",
        linewidth=1.0,
    )

    _plot_interval_series(
        difference_axis,
        rows,
        x_column="character_count",
        center_column="accuracy_difference_from_baseline",
        lower_column="accuracy_difference_ci_low",
        upper_column="accuracy_difference_ci_high",
        color=COLOR_BLUE,
    )

    add_reference_line(
        difference_axis,
        0.0,
        orientation="horizontal",
        color=COLOR_GRAY,
        linestyle="--",
        linewidth=1.0,
    )

    character_counts = _float_values(
        rows,
        "character_count",
    )

    for axis in (
        absolute_axis,
        difference_axis,
    ):
        axis.set_xscale(
            "log",
            base=2,
        )

        axis.set_xticks(
            character_counts
        )

        axis.set_xticklabels(
            [
                str(
                    int(
                        value
                    )
                )
                for value in (
                    character_counts
                )
            ]
        )

    style_axis(
        absolute_axis,
        xlabel="Random Fourier feature count",
        ylabel="Mean accuracy",
        ygrid=True,
    )

    absolute_axis.set_ylim(
        0.0,
        1.0,
    )

    style_axis(
        difference_axis,
        xlabel="Random Fourier feature count",
        ylabel=(
            "Accuracy difference "
            "from baseline"
        ),
        ygrid=True,
    )

    add_panel_label(
        absolute_axis,
        "(a)",
    )

    add_panel_label(
        difference_axis,
        "(b)",
    )

    return _save_reporting_figure(
        figure,
        figure_directory,
        _RFF_FIGURE_FILENAME,
        overwrite=overwrite,
    )


def write_robustness_figure(
    figure_directory: str | Path,
    result: RobustnessResult,
    *,
    overwrite: bool = False,
) -> Path:
    rows = robustness_summary_rows(
        result
    )

    figure, axes = create_subplots(
        2,
        2,
        width=DOUBLE_COLUMN_WIDTH_INCHES,
        aspect_ratio=1.6,
        constrained_layout=True,
    )

    flat_axes = np.asarray(
        axes,
        dtype=object,
    ).reshape(
        -1
    )

    panel_labels = (
        "(a)",
        "(b)",
        "(c)",
        "(d)",
    )

    for (
        panel_index,
        perturbation,
    ) in enumerate(
        _ROBUSTNESS_PERTURBATIONS
    ):
        axis = flat_axes[
            panel_index
        ]

        perturbation_rows = tuple(
            sorted(
                (
                    row
                    for row in rows
                    if str(
                        row[
                            "perturbation"
                        ]
                    )
                    == perturbation
                ),
                key=lambda row: float(
                    row[
                        "severity"
                    ]
                ),
            )
        )

        _plot_interval_series(
            axis,
            perturbation_rows,
            x_column="severity",
            center_column="accuracy_difference",
            lower_column="accuracy_difference_ci_low",
            upper_column="accuracy_difference_ci_high",
            color=COLOR_BLUE,
        )

        add_reference_line(
            axis,
            0.0,
            orientation="horizontal",
            color=COLOR_GRAY,
            linestyle="--",
            linewidth=1.0,
        )

        style_axis(
            axis,
            xlabel=_ROBUSTNESS_PANEL_LABELS[
                perturbation
            ],
            ylabel=(
                "Accuracy difference "
                "from clean baseline"
            ),
            ygrid=True,
        )

        axis.set_xticks(
            _float_values(
                perturbation_rows,
                "severity",
            )
        )

        add_panel_label(
            axis,
            panel_labels[
                panel_index
            ],
        )

    return _save_reporting_figure(
        figure,
        figure_directory,
        _ROBUSTNESS_FIGURE_FILENAME,
        overwrite=overwrite,
    )


def write_scalability_figure(
    figure_directory: str | Path,
    result: ScalabilityResult,
    *,
    overwrite: bool = False,
) -> Path:
    rows = scalability_summary_rows(
        result
    )

    support_sizes = tuple(
        sorted(
            {
                int(
                    row[
                        "input_support_size"
                    ]
                )
                for row in rows
            }
        )
    )

    character_counts = tuple(
        sorted(
            {
                int(
                    row[
                        "character_count"
                    ]
                )
                for row in rows
            }
        )
    )

    figure, axes = create_subplots(
        1,
        len(
            support_sizes
        ),
        width=DOUBLE_COLUMN_WIDTH_INCHES,
        aspect_ratio=2.4,
        sharey=True,
        constrained_layout=True,
    )

    flat_axes = np.asarray(
        axes,
        dtype=object,
    ).reshape(
        -1
    )

    for support_position, (
        support_size,
        axis,
    ) in enumerate(
        zip(
            support_sizes,
            flat_axes,
            strict=True,
        )
    ):
        support_rows = tuple(
            row
            for row in rows
            if int(
                row[
                    "input_support_size"
                ]
            )
            == support_size
        )

        for (
            character_position,
            character_count,
        ) in enumerate(
            character_counts
        ):
            series_rows = tuple(
                sorted(
                    (
                        row
                        for row in support_rows
                        if int(
                            row[
                                "character_count"
                            ]
                        )
                        == character_count
                    ),
                    key=lambda row: int(
                        row[
                            "graphs_per_batch"
                        ]
                    ),
                )
            )

            color = DEFAULT_COLOR_CYCLE[
                character_position
                %
                len(
                    DEFAULT_COLOR_CYCLE
                )
            ]

            marker = DEFAULT_MARKERS[
                character_position
                %
                len(
                    DEFAULT_MARKERS
                )
            ]

            axis.plot(
                _float_values(
                    series_rows,
                    "graphs_per_batch",
                ),
                _float_values(
                    series_rows,
                    "mean_representation_seconds",
                ),
                color=color,
                marker=marker,
                linestyle="-",
                label=str(
                    character_count
                ),
            )

        axis.set_xscale(
            "log",
            base=2,
        )

        graph_counts = tuple(
            sorted(
                {
                    int(
                        row[
                            "graphs_per_batch"
                        ]
                    )
                    for row in support_rows
                }
            )
        )

        axis.set_xticks(
            graph_counts
        )

        axis.set_xticklabels(
            [
                str(
                    graph_count
                )
                for graph_count in (
                    graph_counts
                )
            ]
        )

        style_axis(
            axis,
            xlabel="Graphs per batch",
            ylabel=(
                "Representation time (s)"
                if support_position == 0
                else None
            ),
            ygrid=True,
        )

        axis.set_title(
            f"Support size = {support_size}"
        )

        add_panel_label(
            axis,
            f"({chr(ord('a') + support_position)})",
        )

    handles, labels = (
        flat_axes[
            0
        ].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        title="Characters",
        loc="outside lower center",
        ncols=len(
            character_counts
        ),
        frameon=False,
    )

    return _save_reporting_figure(
        figure,
        figure_directory,
        _SCALABILITY_FIGURE_FILENAME,
        overwrite=overwrite,
    )


def write_all_figures(
    figure_directory: str | Path,
    *,
    benchmark: BenchmarkResult,
    ablation: HarmonicAggregationAblationResult,
    batch_size: BatchSizeSensitivityResult,
    rff: RffSensitivityResult,
    robustness: RobustnessResult,
    scalability: ScalabilityResult,
    overwrite: bool = False,
) -> tuple[
    Path,
    ...,
]:
    return (
        write_benchmark_figure(
            figure_directory,
            benchmark,
            overwrite=overwrite,
        ),
        write_ablation_figure(
            figure_directory,
            ablation,
            overwrite=overwrite,
        ),
        write_batch_size_figure(
            figure_directory,
            batch_size,
            overwrite=overwrite,
        ),
        write_rff_sensitivity_figure(
            figure_directory,
            rff,
            overwrite=overwrite,
        ),
        write_robustness_figure(
            figure_directory,
            robustness,
            overwrite=overwrite,
        ),
        write_scalability_figure(
            figure_directory,
            scalability,
            overwrite=overwrite,
        ),
    )


__all__ = [
    "write_ablation_figure",
    "write_all_figures",
    "write_batch_size_figure",
    "write_benchmark_figure",
    "write_rff_sensitivity_figure",
    "write_robustness_figure",
    "write_scalability_figure",
]