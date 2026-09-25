#ifndef GRAPH_BENCHMARK_BENCHMARK_HPP
#define GRAPH_BENCHMARK_BENCHMARK_HPP

#include "../configuration/experiment_config.hpp"
#include "../../../core/random_network_models/random_network_models.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpd {

struct BenchmarkModel {
    int model_index{};

    RandomNetworkModel descriptor;
};

struct BenchmarkSpecification {
    GraphSizeSpecification graph_size;

    int batches_per_model{};

    int graphs_per_batch{};

    ModelParameterSpecification
        model_parameters;

    std::vector<BenchmarkModel>
        models;

    std::size_t model_count() const;

    std::uint64_t batch_count() const;

    std::uint64_t graph_count() const;
};

std::size_t parameter_condition_count(
    const BenchmarkSpecification& specification,
    RandomNetworkFamily family
);

std::size_t parameter_condition_index(
    const BenchmarkSpecification& specification,
    const BenchmarkModel& model,
    int batch_index
);

const BenchmarkModel& benchmark_model_for_index(
    const BenchmarkSpecification& specification,
    int model_index
);

BenchmarkSpecification make_benchmark_specification(
    const CanonicalExperimentSpecification& canonical
);

}  // namespace vpd

#endif