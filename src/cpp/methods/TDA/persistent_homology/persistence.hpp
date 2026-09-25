#ifndef GRAPH_BENCHMARK_PERSISTENCE_HPP
#define GRAPH_BENCHMARK_PERSISTENCE_HPP

#include "diagrams.hpp"
#include "../../../core/graph/graph.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace vpd {

struct Simplex {
    std::vector<int> vertices;
    double filtration{};
    int dimension{};

    bool operator<(
        const Simplex& other
    ) const;

    bool operator==(
        const Simplex& other
    ) const;
};

std::vector<Simplex> build_clique_filtration(
    const Graph& graph
);

VPD1 compute_H1_persistence(
    const Graph& graph
);

VPD1 compute_H1_persistence(
    const Graph& graph,
    const std::vector<
        std::array<std::size_t, 3U>
    >& triangles
);

}  // namespace vpd

#endif