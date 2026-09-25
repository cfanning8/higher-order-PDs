from __future__ import annotations

import time

from core.paths import default_project_paths

from .batch_size_sensitivity import (
    BatchSizeSensitivityResult,
    run_batch_size_sensitivity,
    write_batch_size_sensitivity_result,
)
from .benchmark import (
    BenchmarkResult,
    run_benchmark,
    write_benchmark_result,
)
from .configuration import (
    default_random_graph_classification_config,
)
from .data import (
    load_ablation_catalog,
    load_batch_size_catalog,
    load_benchmark_catalog,
    load_common_metadata,
    load_rff_catalog,
    load_robustness_catalog,
    load_scalability_catalog,
)
from .harmonic_aggregation_ablation import (
    HarmonicAggregationAblationResult,
    run_harmonic_aggregation_ablation,
    write_harmonic_aggregation_ablation_result,
)
from .output import (
    AnalysisOutputLayout,
    write_analysis_configuration,
)
from .reporting import (
    write_ablation_figure,
    write_batch_size_figure,
    write_benchmark_figure,
    write_rff_sensitivity_figure,
    write_robustness_figure,
    write_scalability_figure,
)
from .rff_sensitivity import (
    RffSensitivityResult,
    run_rff_sensitivity,
    write_rff_sensitivity_result,
)
from .robustness import (
    RobustnessResult,
    run_robustness,
    write_robustness_result,
)
from .scalability import (
    ScalabilityResult,
    run_scalability,
    write_scalability_result,
)


def _print_progress(
    step: int,
    total: int,
    message: str,
) -> None:
    print(
        f"[{step}/{total}] {message}",
        flush=True,
    )


def _print_completed(
    step: int,
    total: int,
    message: str,
    started: float,
) -> None:
    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        f"[{step}/{total}] {message} ({elapsed:.1f} s)",
        flush=True,
    )


def main() -> None:
    analysis_started = time.perf_counter()

    config = default_random_graph_classification_config()

    benchmark_required = (
        config.selection.benchmark
        or config.selection.harmonic_aggregation_ablation
        or config.selection.batch_size_sensitivity
        or config.selection.rff_sensitivity
        or config.selection.robustness
    )

    total_steps = 2

    if benchmark_required:
        total_steps += 1

    if config.selection.benchmark:
        total_steps += 2

    if config.selection.harmonic_aggregation_ablation:
        total_steps += 3

    if config.selection.batch_size_sensitivity:
        total_steps += 3

    if config.selection.rff_sensitivity:
        total_steps += 3

    if config.selection.robustness:
        total_steps += 3

    if config.selection.scalability:
        total_steps += 3

    paths = default_project_paths()

    step = 1
    _print_progress(
        step,
        total_steps,
        "Loading common metadata",
    )
    started = time.perf_counter()

    common = load_common_metadata(
        paths.random_graph_classification_raw_root
    )

    _print_completed(
        step,
        total_steps,
        "Common metadata loaded",
        started,
    )

    step += 1
    _print_progress(
        step,
        total_steps,
        "Preparing analysis output",
    )
    started = time.perf_counter()

    paths.create_output_directories(
        experiments=(
            "random_graph_classification",
        ),
    )

    layout = AnalysisOutputLayout(
        paths.random_graph_classification_processed_root
    )

    write_analysis_configuration(
        layout,
        common,
        config,
    )

    _print_completed(
        step,
        total_steps,
        "Analysis output prepared",
        started,
    )

    benchmark_result: BenchmarkResult | None = None
    ablation_result: HarmonicAggregationAblationResult | None = None
    batch_size_result: BatchSizeSensitivityResult | None = None
    rff_result: RffSensitivityResult | None = None
    robustness_result: RobustnessResult | None = None
    scalability_result: ScalabilityResult | None = None

    if benchmark_required:
        step += 1
        _print_progress(
            step,
            total_steps,
            "Running benchmark",
        )
        started = time.perf_counter()

        benchmark = load_benchmark_catalog(
            common
        )

        benchmark_result = run_benchmark(
            common,
            benchmark,
            config,
        )

        _print_completed(
            step,
            total_steps,
            "Benchmark complete",
            started,
        )

        if config.selection.benchmark:
            step += 1
            _print_progress(
                step,
                total_steps,
                "Writing benchmark results",
            )
            started = time.perf_counter()

            write_benchmark_result(
                layout,
                benchmark_result,
            )

            _print_completed(
                step,
                total_steps,
                "Benchmark results written",
                started,
            )

        if config.selection.harmonic_aggregation_ablation:
            step += 1
            _print_progress(
                step,
                total_steps,
                "Running harmonic-aggregation ablation",
            )
            started = time.perf_counter()

            ablation = load_ablation_catalog(
                common
            )

            ablation_result = run_harmonic_aggregation_ablation(
                ablation,
                benchmark_result,
                config,
            )

            _print_completed(
                step,
                total_steps,
                "Harmonic-aggregation ablation complete",
                started,
            )

            step += 1
            _print_progress(
                step,
                total_steps,
                "Writing harmonic-aggregation ablation results",
            )
            started = time.perf_counter()

            write_harmonic_aggregation_ablation_result(
                layout,
                ablation_result,
            )

            _print_completed(
                step,
                total_steps,
                "Harmonic-aggregation ablation results written",
                started,
            )

        if config.selection.batch_size_sensitivity:
            step += 1
            _print_progress(
                step,
                total_steps,
                "Running batch-size sensitivity",
            )
            started = time.perf_counter()

            batch_size = load_batch_size_catalog(
                common
            )

            batch_size_result = run_batch_size_sensitivity(
                common,
                batch_size,
                benchmark_result,
                config,
            )

            _print_completed(
                step,
                total_steps,
                "Batch-size sensitivity complete",
                started,
            )

            step += 1
            _print_progress(
                step,
                total_steps,
                "Writing batch-size sensitivity results",
            )
            started = time.perf_counter()

            write_batch_size_sensitivity_result(
                layout,
                batch_size_result,
            )

            _print_completed(
                step,
                total_steps,
                "Batch-size sensitivity results written",
                started,
            )

        if config.selection.rff_sensitivity:
            step += 1
            _print_progress(
                step,
                total_steps,
                "Running RFF sensitivity",
            )
            started = time.perf_counter()

            rff = load_rff_catalog(
                common
            )

            rff_result = run_rff_sensitivity(
                rff,
                benchmark_result,
                config,
            )

            _print_completed(
                step,
                total_steps,
                "RFF sensitivity complete",
                started,
            )

            step += 1
            _print_progress(
                step,
                total_steps,
                "Writing RFF sensitivity results",
            )
            started = time.perf_counter()

            write_rff_sensitivity_result(
                layout,
                rff_result,
            )

            _print_completed(
                step,
                total_steps,
                "RFF sensitivity results written",
                started,
            )

        if config.selection.robustness:
            step += 1
            _print_progress(
                step,
                total_steps,
                "Running robustness study",
            )
            started = time.perf_counter()

            robustness = load_robustness_catalog(
                common
            )

            robustness_result = run_robustness(
                robustness,
                benchmark_result,
                config,
            )

            _print_completed(
                step,
                total_steps,
                "Robustness study complete",
                started,
            )

            step += 1
            _print_progress(
                step,
                total_steps,
                "Writing robustness results",
            )
            started = time.perf_counter()

            write_robustness_result(
                layout,
                robustness_result,
            )

            _print_completed(
                step,
                total_steps,
                "Robustness results written",
                started,
            )

    if config.selection.scalability:
        step += 1
        _print_progress(
            step,
            total_steps,
            "Running scalability analysis",
        )
        started = time.perf_counter()

        scalability = load_scalability_catalog(
            common
        )

        scalability_result = run_scalability(
            scalability,
        )

        _print_completed(
            step,
            total_steps,
            "Scalability analysis complete",
            started,
        )

        step += 1
        _print_progress(
            step,
            total_steps,
            "Writing scalability results",
        )
        started = time.perf_counter()

        write_scalability_result(
            layout,
            scalability_result,
        )

        _print_completed(
            step,
            total_steps,
            "Scalability results written",
            started,
        )

    figure_directory = (
        paths.random_graph_classification_figures_root
    )

    if (
        config.selection.benchmark
        and benchmark_result is not None
    ):
        step += 1
        _print_progress(
            step,
            total_steps,
            "Writing benchmark figure",
        )
        started = time.perf_counter()

        write_benchmark_figure(
            figure_directory,
            benchmark_result,
            overwrite=True,
        )

        _print_completed(
            step,
            total_steps,
            "Benchmark figure written",
            started,
        )

    if ablation_result is not None:
        step += 1
        _print_progress(
            step,
            total_steps,
            "Writing harmonic-aggregation ablation figure",
        )
        started = time.perf_counter()

        write_ablation_figure(
            figure_directory,
            ablation_result,
            overwrite=True,
        )

        _print_completed(
            step,
            total_steps,
            "Harmonic-aggregation ablation figure written",
            started,
        )

    if batch_size_result is not None:
        step += 1
        _print_progress(
            step,
            total_steps,
            "Writing batch-size sensitivity figure",
        )
        started = time.perf_counter()

        write_batch_size_figure(
            figure_directory,
            batch_size_result,
            overwrite=True,
        )

        _print_completed(
            step,
            total_steps,
            "Batch-size sensitivity figure written",
            started,
        )

    if rff_result is not None:
        step += 1
        _print_progress(
            step,
            total_steps,
            "Writing RFF sensitivity figure",
        )
        started = time.perf_counter()

        write_rff_sensitivity_figure(
            figure_directory,
            rff_result,
            overwrite=True,
        )

        _print_completed(
            step,
            total_steps,
            "RFF sensitivity figure written",
            started,
        )

    if robustness_result is not None:
        step += 1
        _print_progress(
            step,
            total_steps,
            "Writing robustness figure",
        )
        started = time.perf_counter()

        write_robustness_figure(
            figure_directory,
            robustness_result,
            overwrite=True,
        )

        _print_completed(
            step,
            total_steps,
            "Robustness figure written",
            started,
        )

    if scalability_result is not None:
        step += 1
        _print_progress(
            step,
            total_steps,
            "Writing scalability figure",
        )
        started = time.perf_counter()

        write_scalability_figure(
            figure_directory,
            scalability_result,
            overwrite=True,
        )

        _print_completed(
            step,
            total_steps,
            "Scalability figure written",
            started,
        )

    elapsed = (
        time.perf_counter()
        - analysis_started
    )

    print(
        f"Analysis complete ({elapsed:.1f} s).",
        flush=True,
    )


__all__ = [
    "main",
]


if __name__ == "__main__":
    main()