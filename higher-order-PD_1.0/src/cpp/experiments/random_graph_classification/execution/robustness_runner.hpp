#ifndef GRAPH_BENCHMARK_ROBUSTNESS_RUNNER_HPP
#define GRAPH_BENCHMARK_ROBUSTNESS_RUNNER_HPP

#include "../configuration/experiment_config.hpp"
#include "../experiment_design/benchmark.hpp"
#include "benchmark.hpp"
#include "execution_plan_runner.hpp"

#include "../../../core/graph/graph.hpp"
#include "../../../methods/TDA/persistent_homology/diagrams.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace vpd {

struct RobustnessGraphRecord {
    GraphKey key;

    std::uint64_t perturbation_seed{};

    Graph perturbed_graph;

    VPD1 persistence_diagram;
};

struct RobustnessBatch {
    BenchmarkBatch clean_batch;

    int condition_index{-1};

    RobustnessCondition condition;

    int realization_index{-1};

    std::vector<RobustnessGraphRecord>
        perturbed_graphs;

    std::size_t graph_count() const;
};

using RobustnessBatchConsumer =
    std::function<void(
        RobustnessBatch&&
    )>;

RobustnessBatch generate_robustness_batch(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task
);

void run_robustness_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const RobustnessBatchConsumer& consume_batch
);

}  // namespace vpd

#endif