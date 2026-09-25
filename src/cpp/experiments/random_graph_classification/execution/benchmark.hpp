#ifndef GRAPH_BENCHMARK_EXECUTION_BENCHMARK_HPP
#define GRAPH_BENCHMARK_EXECUTION_BENCHMARK_HPP

#include "../experiment_design/benchmark.hpp"
#include "execution_plan_runner.hpp"

#include "../../../methods/TDA/persistent_homology/diagrams.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace vpd {

struct GraphKey {
    int model_index{-1};

    int parameter_condition_index{-1};

    int batch_index{-1};

    int graph_index{-1};
};

struct BenchmarkGraphRecord {
    GraphKey key;

    std::uint64_t latent_seed{};

    std::uint64_t sampling_seed{};

    GeneratedNetwork network;

    VPD1 persistence_diagram;
};

struct BenchmarkBatch {
    int model_index{-1};

    int parameter_condition_index{-1};

    int batch_index{-1};

    RandomNetworkModel descriptor;

    std::vector<BenchmarkGraphRecord>
        graphs;

    std::size_t graph_count() const;
};

using BenchmarkBatchConsumer =
    std::function<void(
        BenchmarkBatch&&
    )>;

BenchmarkBatch generate_benchmark_batch(
    std::uint64_t root_seed,
    const BenchmarkSpecification& specification,
    int model_index,
    int batch_index,
    int graphs_per_batch
);

void run_benchmark_task(
    std::uint64_t root_seed,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const BenchmarkBatchConsumer& consume_batch
);

}  // namespace vpd

#endif