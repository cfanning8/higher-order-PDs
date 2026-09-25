#include "random_network_models.hpp"

namespace vpd {

const std::vector<RandomNetworkModel>&
all_random_network_models() {
    static const std::vector<RandomNetworkModel>
        models{
            {
                RandomNetworkFamily::ErdosRenyi,
                "erdos_renyi",
                "Erdos-Renyi",
                "gnm"
            },
            {
                RandomNetworkFamily::WattsStrogatz,
                "watts_strogatz",
                "Watts-Strogatz",
                "clockwise_rewiring"
            },
            {
                RandomNetworkFamily::BarabasiAlbert,
                "barabasi_albert",
                "Barabasi-Albert",
                "initial_clique_preferential_attachment"
            },
            {
                RandomNetworkFamily::ConfigurationModel,
                "configuration_model",
                "Configuration Model",
                "simple_regular_configuration"
            },
            {
                RandomNetworkFamily::StochasticBlockModel,
                "stochastic_block_model",
                "Stochastic Block Model",
                "balanced_sbm_conditioned_edge_count"
            },
            {
                RandomNetworkFamily::ChungLu,
                "chung_lu",
                "Chung-Lu",
                "chung_lu_conditioned_edge_count"
            },
            {
                RandomNetworkFamily::Kleinberg,
                "kleinberg",
                "Kleinberg Small-World",
                "undirected_fixed_edge_kleinberg"
            },
            {
                RandomNetworkFamily::Girg,
                "girg",
                "Geometric Inhomogeneous Random Graph",
                "girg_conditioned_edge_count"
            },
            {
                RandomNetworkFamily::
                    HyperbolicRandomGraph,
                "hyperbolic_random_graph",
                "Hyperbolic Random Graph",
                "soft_hyperbolic_conditioned_edge_count"
            },
            {
                RandomNetworkFamily::
                    ExponentialRandomGraphModel,
                "exponential_random_graph_model",
                "Exponential Random Graph Model",
                "fixed_edge_gwesp_ergm"
            }
        };

    return models;
}

}  // namespace vpd