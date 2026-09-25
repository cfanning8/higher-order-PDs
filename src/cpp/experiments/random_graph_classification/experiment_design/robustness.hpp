#ifndef GRAPH_BENCHMARK_ROBUSTNESS_HPP
#define GRAPH_BENCHMARK_ROBUSTNESS_HPP

#include "../../../core/graph/graph.hpp"

#include <cstdint>

namespace vpd {

Graph apply_edge_deletion(
    const Graph& graph,
    double severity,
    std::uint64_t seed
);

Graph apply_edge_insertion(
    const Graph& graph,
    double severity,
    std::uint64_t seed
);

Graph apply_degree_preserving_rewiring(
    const Graph& graph,
    double severity,
    std::uint64_t seed
);

Graph apply_filtration_noise(
    const Graph& graph,
    double severity,
    std::uint64_t seed
);

}  // namespace vpd

#endif