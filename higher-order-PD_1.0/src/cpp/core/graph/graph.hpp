#ifndef VPD_GRAPH_HPP
#define VPD_GRAPH_HPP

#include <cstddef>
#include <set>
#include <utility>
#include <vector>

namespace vpd {

struct Edge {
    int u{};
    int v{};
    int step{};
    double time{};
};

class Graph {
public:
    explicit Graph(int vertex_count = 0);

    int num_vertices() const;
    int num_edges() const;

    bool has_edge(int u, int v) const;

    bool add_edge(
        int u,
        int v,
        int step = 0,
        double time = 0.0
    );

    void reserve_edges(std::size_t expected_edges);
    void clear();

    const std::vector<Edge>& get_edges() const;

private:
    int vertex_count_{};
    std::vector<Edge> edges_;
    std::set<std::pair<int, int>> edge_set_;

    static std::pair<int, int> canonical_edge(
        int u,
        int v
    );
};

}  // namespace vpd

#endif