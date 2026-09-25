#include "graph.hpp"

#include <utility>

namespace vpd {

Graph::Graph(int vertex_count)
    : vertex_count_(
          vertex_count
      ) {
}

int Graph::num_vertices() const {
    return vertex_count_;
}

int Graph::num_edges() const {
    return static_cast<int>(
        edges_.size()
    );
}

void Graph::reserve_edges(
    std::size_t expected_edges
) {
    edges_.reserve(
        expected_edges
    );
}

void Graph::clear() {
    edges_.clear();
    edge_set_.clear();
}

std::pair<int, int> Graph::canonical_edge(
    int u,
    int v
) {
    if (u > v) {
        std::swap(
            u,
            v
        );
    }

    return {
        u,
        v
    };
}

bool Graph::has_edge(
    int u,
    int v
) const {
    return
        edge_set_.find(
            canonical_edge(
                u,
                v
            )
        ) !=
        edge_set_.end();
}

bool Graph::add_edge(
    int u,
    int v,
    int step,
    double time
) {
    const std::pair<int, int> edge =
        canonical_edge(
            u,
            v
        );

    if (
        edge.first ==
        edge.second
    ) {
        return false;
    }

    if (
        !edge_set_.insert(
            edge
        ).second
    ) {
        return false;
    }

    edges_.push_back(
        Edge{
            edge.first,
            edge.second,
            step,
            time
        }
    );

    return true;
}

const std::vector<Edge>&
Graph::get_edges() const {
    return edges_;
}

}  // namespace vpd