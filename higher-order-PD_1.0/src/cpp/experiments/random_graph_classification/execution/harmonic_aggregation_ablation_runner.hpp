#ifndef GRAPH_BENCHMARK_HARMONIC_AGGREGATION_ABLATION_RUNNER_HPP
#define GRAPH_BENCHMARK_HARMONIC_AGGREGATION_ABLATION_RUNNER_HPP

#include "../configuration/experiment_config.hpp"
#include "../experiment_design/benchmark.hpp"
#include "../experiment_design/harmonic_aggregation_ablation.hpp"
#include "benchmark.hpp"
#include "execution_plan_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

namespace vpd {

struct HarmonicAggregationAblationBatch {
    BenchmarkBatch benchmark_batch;

    HarmonicAggregationAblationKind ablation_kind{
        HarmonicAggregationAblationKind::CrossObservation
    };

    int realization_index{};

    std::uint64_t character_seed{};

    std::optional<std::uint64_t>
        ablation_seed;

    HarmonicValues harmonic_values;

    std::size_t graph_count() const;
};

using HarmonicAggregationAblationConsumer =
    std::function<void(
        HarmonicAggregationAblationBatch&&
    )>;

HarmonicAggregationAblationBatch
generate_harmonic_aggregation_ablation_batch(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task
);

void run_harmonic_aggregation_ablation_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const HarmonicAggregationAblationConsumer&
        consume_batch
);

}  // namespace vpd

#endif