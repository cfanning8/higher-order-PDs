#include "benchmark.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vpd {

std::size_t
BenchmarkSpecification::model_count() const {
    return models.size();
}

std::uint64_t
BenchmarkSpecification::batch_count() const {
    return
        static_cast<std::uint64_t>(
            models.size()
        ) *
        static_cast<std::uint64_t>(
            batches_per_model
        );
}

std::uint64_t
BenchmarkSpecification::graph_count() const {
    return
        batch_count() *
        static_cast<std::uint64_t>(
            graphs_per_batch
        );
}

std::size_t parameter_condition_count(
    const BenchmarkSpecification& specification,
    RandomNetworkFamily family
) {
    switch (family) {
        case RandomNetworkFamily::ErdosRenyi:
            return 1U;

        case RandomNetworkFamily::WattsStrogatz:
            return
                specification.
                    model_parameters.
                    watts_strogatz.size();

        case RandomNetworkFamily::BarabasiAlbert:
            return
                specification.
                    model_parameters.
                    barabasi_albert.size();

        case RandomNetworkFamily::ConfigurationModel:
            return
                specification.
                    model_parameters.
                    configuration_model.size();

        case RandomNetworkFamily::StochasticBlockModel:
            return
                specification.
                    model_parameters.
                    stochastic_block_model.size();

        case RandomNetworkFamily::ChungLu:
            return
                specification.
                    model_parameters.
                    chung_lu.size();

        case RandomNetworkFamily::Kleinberg:
            return
                specification.
                    model_parameters.
                    kleinberg.size();

        case RandomNetworkFamily::Girg:
            return
                specification.
                    model_parameters.
                    girg.size();

        case RandomNetworkFamily::
                HyperbolicRandomGraph:
            return
                specification.
                    model_parameters.
                    hyperbolic_random_graph.size();

        case RandomNetworkFamily::
                ExponentialRandomGraphModel:
            return
                specification.
                    model_parameters.
                    exponential_random_graph_model.size();
    }

    throw std::logic_error(
        "Unknown random-network family."
    );
}

std::size_t parameter_condition_index(
    const BenchmarkSpecification& specification,
    const BenchmarkModel& model,
    int batch_index
) {
    const std::size_t condition_count =
        parameter_condition_count(
            specification,
            model.descriptor.family
        );

    const int batches_per_condition =
        specification.batches_per_model /
        static_cast<int>(
            condition_count
        );

    return static_cast<std::size_t>(
        batch_index /
        batches_per_condition
    );
}

const BenchmarkModel& benchmark_model_for_index(
    const BenchmarkSpecification& specification,
    int model_index
) {
    for (const BenchmarkModel& model :
         specification.models) {
        if (
            model.model_index ==
            model_index
        ) {
            return model;
        }
    }

    throw std::out_of_range(
        "The requested benchmark model index is invalid."
    );
}

BenchmarkSpecification make_benchmark_specification(
    const CanonicalExperimentSpecification& canonical
) {
    BenchmarkSpecification specification;

    specification.graph_size =
        canonical.benchmark.graph_size;

    specification.batches_per_model =
        canonical.benchmark.batches_per_model;

    specification.graphs_per_batch =
        canonical.benchmark.graphs_per_batch;

    specification.model_parameters =
        canonical.benchmark.model_parameters;

    const std::vector<RandomNetworkModel>& registry =
        all_random_network_models();

    specification.models.reserve(
        registry.size()
    );

    for (std::size_t model_index = 0U;
         model_index < registry.size();
         ++model_index) {
        specification.models.push_back(
            BenchmarkModel{
                static_cast<int>(
                    model_index
                ),
                registry[
                    model_index
                ]
            }
        );
    }

    return specification;
}

}  // namespace vpd