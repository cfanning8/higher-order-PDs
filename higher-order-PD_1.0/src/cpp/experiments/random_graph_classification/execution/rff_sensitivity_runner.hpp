#ifndef GRAPH_BENCHMARK_RFF_SENSITIVITY_RUNNER_HPP
#define GRAPH_BENCHMARK_RFF_SENSITIVITY_RUNNER_HPP

#include "../configuration/experiment_config.hpp"
#include "../experiment_design/benchmark.hpp"
#include "benchmark.hpp"
#include "execution_plan_runner.hpp"

#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/aggregation.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace vpd {

struct RffSensitivityConditionResult {
    int character_count{};

    HarmonicValues harmonic_values;
};

struct RffSensitivityResult {
    BenchmarkBatch benchmark_batch;

    int replicate_index{-1};

    std::uint64_t character_seed{};

    std::vector<
        RffSensitivityConditionResult
    > conditions;
};

using RffSensitivityConsumer =
    std::function<void(
        RffSensitivityResult&&
    )>;

void run_rff_sensitivity_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const RffSensitivityConsumer& consume_result
);

}  // namespace vpd

#endif