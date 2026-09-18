#include "benchmark.hpp"

#include "../../../core/utils/seed_derivation.hpp"
#include "../../../methods/TDA/persistent_homology/persistence.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace vpd {

namespace {

VPD1 compute_persistence_diagram(
    const Graph& graph
) {
    VPD1 diagram =
        compute_H1_persistence(
            graph
        );

    clean_diagram(
        diagram
    );

    return diagram;
}

GeneratedNetwork generate_network(
    const BenchmarkSpecification& specification,
    const BenchmarkModel& model,
    std::size_t condition_index,
    const NetworkGenerationRequest& request
) {
    switch (model.descriptor.family) {
        case RandomNetworkFamily::ErdosRenyi:
            return generate_erdos_renyi(
                request
            );

        case RandomNetworkFamily::WattsStrogatz:
            return generate_watts_strogatz(
                request,
                specification.
                    model_parameters.
                    watts_strogatz[
                        condition_index
                    ]
            );

        case RandomNetworkFamily::BarabasiAlbert:
            return generate_barabasi_albert(
                request,
                specification.
                    model_parameters.
                    barabasi_albert[
                        condition_index
                    ]
            );

        case RandomNetworkFamily::ConfigurationModel:
            return generate_configuration_model(
                request,
                specification.
                    model_parameters.
                    configuration_model[
                        condition_index
                    ]
            );

        case RandomNetworkFamily::StochasticBlockModel:
            return generate_stochastic_block_model(
                request,
                specification.
                    model_parameters.
                    stochastic_block_model[
                        condition_index
                    ]
            );

        case RandomNetworkFamily::ChungLu:
            return generate_chung_lu(
                request,
                specification.
                    model_parameters.
                    chung_lu[
                        condition_index
                    ]
            );

        case RandomNetworkFamily::Kleinberg:
            return generate_kleinberg(
                request,
                specification.
                    model_parameters.
                    kleinberg[
                        condition_index
                    ]
            );

        case RandomNetworkFamily::Girg:
            return generate_girg(
                request,
                specification.
                    model_parameters.
                    girg[
                        condition_index
                    ]
            );

        case RandomNetworkFamily::
                HyperbolicRandomGraph:
            return
                generate_hyperbolic_random_graph(
                    request,
                    specification.
                        model_parameters.
                        hyperbolic_random_graph[
                            condition_index
                        ]
                );

        case RandomNetworkFamily::
                ExponentialRandomGraphModel:
            return
                generate_exponential_random_graph_model(
                    request,
                    specification.
                        model_parameters.
                        exponential_random_graph_model[
                            condition_index
                        ]
                );
    }

    throw std::logic_error(
        "Unknown random-network family."
    );
}

BenchmarkGraphRecord generate_benchmark_graph(
    std::uint64_t root_seed,
    const BenchmarkSpecification& specification,
    const BenchmarkModel& model,
    std::size_t condition_index,
    int batch_index,
    int graph_index,
    std::uint64_t latent_seed
) {
    const std::uint64_t sampling_seed =
        derive_seed(
            root_seed,
            SeedDomain::GraphSampling,
            {
                static_cast<std::uint64_t>(
                    model.model_index
                ),
                static_cast<std::uint64_t>(
                    condition_index
                ),
                static_cast<std::uint64_t>(
                    batch_index
                ),
                static_cast<std::uint64_t>(
                    graph_index
                )
            }
        );

    const NetworkGenerationRequest request{
        specification.graph_size.vertex_count,
        specification.graph_size.edge_count,
        latent_seed,
        sampling_seed
    };

    GeneratedNetwork network =
        generate_network(
            specification,
            model,
            condition_index,
            request
        );

    VPD1 persistence_diagram =
        compute_persistence_diagram(
            network.graph
        );

    return BenchmarkGraphRecord{
        GraphKey{
            model.model_index,
            static_cast<int>(
                condition_index
            ),
            batch_index,
            graph_index
        },
        latent_seed,
        sampling_seed,
        std::move(
            network
        ),
        std::move(
            persistence_diagram
        )
    };
}

}  // namespace

std::size_t BenchmarkBatch::graph_count() const {
    return graphs.size();
}

BenchmarkBatch generate_benchmark_batch(
    std::uint64_t root_seed,
    const BenchmarkSpecification& specification,
    int model_index,
    int batch_index,
    int graphs_per_batch
) {
    const BenchmarkModel& model =
        benchmark_model_for_index(
            specification,
            model_index
        );

    const std::size_t condition_index =
        parameter_condition_index(
            specification,
            model,
            batch_index
        );

    const std::uint64_t latent_seed =
        derive_seed(
            root_seed,
            SeedDomain::GraphLatent,
            {
                static_cast<std::uint64_t>(
                    model_index
                ),
                static_cast<std::uint64_t>(
                    condition_index
                ),
                static_cast<std::uint64_t>(
                    batch_index
                )
            }
        );

    BenchmarkBatch batch;

    batch.model_index =
        model.model_index;

    batch.parameter_condition_index =
        static_cast<int>(
            condition_index
        );

    batch.batch_index =
        batch_index;

    batch.descriptor =
        model.descriptor;

    batch.graphs.reserve(
        static_cast<std::size_t>(
            graphs_per_batch
        )
    );

    for (int graph_index = 0;
         graph_index <
             graphs_per_batch;
         ++graph_index) {
        batch.graphs.push_back(
            generate_benchmark_graph(
                root_seed,
                specification,
                model,
                condition_index,
                batch_index,
                graph_index,
                latent_seed
            )
        );
    }

    return batch;
}

void run_benchmark_task(
    std::uint64_t root_seed,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const BenchmarkBatchConsumer& consume_batch
) {
    consume_batch(
        generate_benchmark_batch(
            root_seed,
            specification,
            task.model_index,
            task.batch_index,
            specification.graphs_per_batch
        )
    );
}

}  // namespace vpd