#ifndef GRAPH_BENCHMARK_SCALABILITY_RUNNER_HPP
#define GRAPH_BENCHMARK_SCALABILITY_RUNNER_HPP

#include "../configuration/experiment_config.hpp"
#include "../experiment_design/scalability.hpp"
#include "execution_plan_runner.hpp"

#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/aggregation.hpp"

#include <cstdint>
#include <functional>

namespace vpd {

struct ScalabilityComputationMetrics {
    double representation_seconds{};
};

struct ScalabilityResult {
    ScalabilityCondition condition;

    std::uint64_t sample_seed{};

    std::uint64_t character_seed{};

    HarmonicValues harmonic_values;

    ScalabilityComputationMetrics metrics;
};

using ScalabilityResultConsumer =
    std::function<void(
        ScalabilityResult&&
    )>;

ScalabilityResult execute_scalability_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionTask& task
);

void run_scalability_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionTask& task,
    const ScalabilityResultConsumer& consume_result
);

}  // namespace vpd

#endif