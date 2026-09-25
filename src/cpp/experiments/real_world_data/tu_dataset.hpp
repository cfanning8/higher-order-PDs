#ifndef GRAPH_BENCHMARK_TU_DATASET_HPP
#define GRAPH_BENCHMARK_TU_DATASET_HPP

#include "../../core/graph/graph.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace vpd {

struct TuGraph {
    Graph graph;
    int label{};
    std::size_t graph_index{};
};

struct TuDataset {
    std::string name;
    std::vector<TuGraph> graphs;
};

TuDataset load_tu_dataset(
    const std::filesystem::path& directory,
    const std::string& name
);

}  // namespace vpd

#endif