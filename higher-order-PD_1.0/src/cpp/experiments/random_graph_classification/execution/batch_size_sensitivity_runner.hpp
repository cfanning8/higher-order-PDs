#ifndef GRAPH_BENCHMARK_BATCH_SIZE_SENSITIVITY_RUNNER_HPP
#define GRAPH_BENCHMARK_BATCH_SIZE_SENSITIVITY_RUNNER_HPP

#include "../configuration/experiment_config.hpp"
#include "../experiment_design/benchmark.hpp"
#include "benchmark.hpp"
#include "execution_plan_runner.hpp"

#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/aggregation.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace vpd {

struct BatchSizeSensitivityConditionResult {
    int graphs_per_batch{};

    HarmonicValues harmonic_values;
};

struct BatchSizeSensitivityResult {
    BenchmarkBatch benchmark_batch;

    std::uint64_t character_seed{};

    std::vector<
        BatchSizeSensitivityConditionResult
    > conditions;
};

using BatchSizeSensitivityConsumer =
    std::function<void(
        BatchSizeSensitivityResult&&
    )>;

void run_batch_size_sensitivity_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const BatchSizeSensitivityConsumer& consume_result
);

}  // namespace vpd

#endif