#ifndef GRAPH_BENCHMARK_RANDOM_NETWORK_MODELS_COMMON_HPP
#define GRAPH_BENCHMARK_RANDOM_NETWORK_MODELS_COMMON_HPP

#include "random_network_models.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace vpd::random_network_models_detail {

using EdgePair =
    std::pair<int, int>;

struct GeneratedEdge {
    int u{};
    int v{};
    double support{};
    std::uint64_t formation_index{};
};

struct DyadProbability {
    int u{};
    int v{};
    double probability{};
};

struct FiltrationResult {
    Graph graph;
    std::vector<std::size_t> source_indices;
};

EdgePair canonical_pair(
    int u,
    int v
);

std::size_t sample_weighted_index(
    std::mt19937_64& generator,
    const std::vector<double>& weights
);

double logistic(
    double value
);

FiltrationResult build_filtration_graph(
    int vertex_count,
    const std::vector<GeneratedEdge>& edges
);

std::vector<GeneratedEdge>
sample_conditional_bernoulli_edges(
    std::vector<DyadProbability> dyads,
    int edge_count,
    std::uint64_t seed
);

std::vector<double>
power_law_weight_profile(
    int vertex_count,
    double exponent
);

std::vector<double>
ordered_edge_field(
    const std::vector<GeneratedEdge>& edges,
    const FiltrationResult& filtration,
    bool support
);

}  // namespace vpd::random_network_models_detail

#endif