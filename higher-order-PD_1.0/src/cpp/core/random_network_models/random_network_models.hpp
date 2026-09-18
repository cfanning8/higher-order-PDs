#ifndef GRAPH_BENCHMARK_RANDOM_NETWORK_MODELS_HPP
#define GRAPH_BENCHMARK_RANDOM_NETWORK_MODELS_HPP

#include "../graph/graph.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vpd {

enum class RandomNetworkFamily {
    ErdosRenyi,
    WattsStrogatz,
    BarabasiAlbert,
    ConfigurationModel,
    StochasticBlockModel,
    ChungLu,
    Kleinberg,
    Girg,
    HyperbolicRandomGraph,
    ExponentialRandomGraphModel
};

struct RandomNetworkModel {
    RandomNetworkFamily family{};
    std::string stable_id;
    std::string display_name;
    std::string variant;
};

struct NetworkGenerationRequest {
    int vertex_count{};
    int edge_count{};
    std::uint64_t latent_seed{};
    std::uint64_t sampling_seed{};
};

struct ScalarField {
    std::string name;
    double value{};
};

struct VertexField {
    std::string name;
    std::vector<double> values;
};

struct EdgeField {
    std::string name;
    std::vector<double> values;
};

struct GeneratedNetwork {
    Graph graph;
    std::vector<ScalarField> graph_fields;
    std::vector<VertexField> vertex_fields;
    std::vector<EdgeField> edge_fields;
};

struct WattsStrogatzParameters {
    int degree{};
    double rewiring_probability{};
};

struct BarabasiAlbertParameters {
    int attachment_count{};
    int initial_clique_size{};
};

struct ConfigurationModelParameters {
    int degree{};
};

struct StochasticBlockModelParameters {
    int block_count{};
    double log_odds_contrast{};
};

struct ChungLuParameters {
    double degree_exponent{};
};

struct KleinbergParameters {
    int rows{};
    int columns{};
    int long_range_edge_count{};
    double distance_exponent{};
};

struct GirgParameters {
    double degree_exponent{};
    double connection_exponent{};
};

struct HyperbolicRandomGraphParameters {
    double degree_exponent{};
    double temperature{};
    double curvature{};
    double radius{};
};

struct ExponentialRandomGraphModelParameters {
    double gwesp_coefficient{};
    double gwesp_decay{};
    int burn_in_sweeps{};
    int sampling_sweeps{};
};

GeneratedNetwork generate_erdos_renyi(
    const NetworkGenerationRequest& request
);

GeneratedNetwork generate_watts_strogatz(
    const NetworkGenerationRequest& request,
    const WattsStrogatzParameters& parameters
);

GeneratedNetwork generate_barabasi_albert(
    const NetworkGenerationRequest& request,
    const BarabasiAlbertParameters& parameters
);

GeneratedNetwork generate_configuration_model(
    const NetworkGenerationRequest& request,
    const ConfigurationModelParameters& parameters
);

GeneratedNetwork generate_stochastic_block_model(
    const NetworkGenerationRequest& request,
    const StochasticBlockModelParameters& parameters
);

GeneratedNetwork generate_chung_lu(
    const NetworkGenerationRequest& request,
    const ChungLuParameters& parameters
);

GeneratedNetwork generate_kleinberg(
    const NetworkGenerationRequest& request,
    const KleinbergParameters& parameters
);

GeneratedNetwork generate_girg(
    const NetworkGenerationRequest& request,
    const GirgParameters& parameters
);

GeneratedNetwork generate_hyperbolic_random_graph(
    const NetworkGenerationRequest& request,
    const HyperbolicRandomGraphParameters& parameters
);

GeneratedNetwork generate_exponential_random_graph_model(
    const NetworkGenerationRequest& request,
    const ExponentialRandomGraphModelParameters& parameters
);

const std::vector<RandomNetworkModel>&
all_random_network_models();

}  // namespace vpd

#endif