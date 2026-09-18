#include "robustness.hpp"

#include "../../../core/utils/portable_random.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace vpd {

namespace {

using EdgePair =
    std::pair<int, int>;

struct EdgeSwap {
    std::size_t first_index{};
    std::size_t second_index{};
    EdgePair first_replacement;
    EdgePair second_replacement;
};

EdgePair canonical_pair(
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

std::size_t perturbation_count(
    std::size_t edge_count,
    double severity
) {
    return static_cast<std::size_t>(
        std::floor(
            static_cast<long double>(
                severity
            ) *
            static_cast<long double>(
                edge_count
            )
        )
    );
}

Graph graph_from_edges(
    int vertex_count,
    const std::vector<Edge>& edges
) {
    Graph result(
        vertex_count
    );

    result.reserve_edges(
        edges.size()
    );

    for (const Edge& edge :
         edges) {
        result.add_edge(
            edge.u,
            edge.v,
            edge.step,
            edge.time
        );
    }

    return result;
}

std::set<EdgePair> edge_membership(
    const std::vector<Edge>& edges
) {
    std::set<EdgePair> membership;

    for (const Edge& edge :
         edges) {
        membership.insert(
            canonical_pair(
                edge.u,
                edge.v
            )
        );
    }

    return membership;
}

std::vector<EdgePair> nonedges(
    int vertex_count,
    const std::set<EdgePair>& membership
) {
    std::vector<EdgePair> candidates;

    for (int u = 0;
         u < vertex_count;
         ++u) {
        for (int v = u + 1;
             v < vertex_count;
             ++v) {
            const EdgePair edge{
                u,
                v
            };

            if (
                membership.find(
                    edge
                ) ==
                membership.end()
            ) {
                candidates.push_back(
                    edge
                );
            }
        }
    }

    return candidates;
}

void assign_filtration_steps(
    std::vector<Edge>& edges
) {
    std::vector<std::size_t> order(
        edges.size()
    );

    for (std::size_t index = 0U;
         index < order.size();
         ++index) {
        order[index] =
            index;
    }

    std::sort(
        order.begin(),
        order.end(),
        [&edges](
            std::size_t first_index,
            std::size_t second_index
        ) {
            const Edge& first =
                edges[
                    first_index
                ];

            const Edge& second =
                edges[
                    second_index
                ];

            if (
                first.time !=
                second.time
            ) {
                return
                    first.time <
                    second.time;
            }

            return
                canonical_pair(
                    first.u,
                    first.v
                ) <
                canonical_pair(
                    second.u,
                    second.v
                );
        }
    );

    int step =
        0;

    double previous_time =
        0.0;

    for (std::size_t position = 0U;
         position < order.size();
         ++position) {
        Edge& edge =
            edges[
                order[
                    position
                ]
            ];

        if (
            position == 0U ||
            edge.time !=
                previous_time
        ) {
            ++step;

            previous_time =
                edge.time;
        }

        edge.step =
            step;
    }
}

bool four_distinct_endpoints(
    const EdgePair& first,
    const EdgePair& second
) {
    return
        first.first !=
            second.first &&
        first.first !=
            second.second &&
        first.second !=
            second.first &&
        first.second !=
            second.second;
}

bool replacement_available(
    const EdgePair& replacement,
    const EdgePair& first_original,
    const EdgePair& second_original,
    const std::set<EdgePair>& membership
) {
    if (
        replacement ==
            first_original ||
        replacement ==
            second_original
    ) {
        return true;
    }

    return
        membership.find(
            replacement
        ) ==
        membership.end();
}

void append_swap_candidate(
    std::vector<EdgeSwap>& candidates,
    std::size_t first_index,
    std::size_t second_index,
    const EdgePair& first_original,
    const EdgePair& second_original,
    const EdgePair& first_replacement,
    const EdgePair& second_replacement,
    const std::set<EdgePair>& membership
) {
    if (
        first_replacement ==
        second_replacement
    ) {
        return;
    }

    if (
        !replacement_available(
            first_replacement,
            first_original,
            second_original,
            membership
        ) ||
        !replacement_available(
            second_replacement,
            first_original,
            second_original,
            membership
        )
    ) {
        return;
    }

    candidates.push_back(
        EdgeSwap{
            first_index,
            second_index,
            first_replacement,
            second_replacement
        }
    );
}

std::vector<EdgeSwap> admissible_swaps(
    const std::vector<Edge>& edges,
    const std::set<EdgePair>& membership
) {
    std::vector<EdgeSwap> candidates;

    for (std::size_t first_index = 0U;
         first_index < edges.size();
         ++first_index) {
        const EdgePair first =
            canonical_pair(
                edges[
                    first_index
                ].u,
                edges[
                    first_index
                ].v
            );

        for (
            std::size_t second_index =
                first_index + 1U;
            second_index <
                edges.size();
            ++second_index
        ) {
            const EdgePair second =
                canonical_pair(
                    edges[
                        second_index
                    ].u,
                    edges[
                        second_index
                    ].v
                );

            if (
                !four_distinct_endpoints(
                    first,
                    second
                )
            ) {
                continue;
            }

            append_swap_candidate(
                candidates,
                first_index,
                second_index,
                first,
                second,
                canonical_pair(
                    first.first,
                    second.first
                ),
                canonical_pair(
                    first.second,
                    second.second
                ),
                membership
            );

            append_swap_candidate(
                candidates,
                first_index,
                second_index,
                first,
                second,
                canonical_pair(
                    first.first,
                    second.second
                ),
                canonical_pair(
                    first.second,
                    second.first
                ),
                membership
            );
        }
    }

    return candidates;
}

}  // namespace

Graph apply_edge_deletion(
    const Graph& graph,
    double severity,
    std::uint64_t seed
) {
    const std::vector<Edge>& source =
        graph.get_edges();

    const std::size_t deletion_count =
        perturbation_count(
            source.size(),
            severity
        );

    std::vector<std::size_t> order(
        source.size()
    );

    for (std::size_t index = 0U;
         index < order.size();
         ++index) {
        order[index] =
            index;
    }

    std::mt19937_64 generator(
        seed
    );

    portable_shuffle(
        order,
        generator
    );

    std::vector<bool> deleted(
        source.size(),
        false
    );

    for (std::size_t index = 0U;
         index < deletion_count;
         ++index) {
        deleted[
            order[
                index
            ]
        ] =
            true;
    }

    std::vector<Edge> edges;

    edges.reserve(
        source.size() -
        deletion_count
    );

    for (std::size_t index = 0U;
         index < source.size();
         ++index) {
        if (!deleted[index]) {
            edges.push_back(
                source[
                    index
                ]
            );
        }
    }

    return graph_from_edges(
        graph.num_vertices(),
        edges
    );
}

Graph apply_edge_insertion(
    const Graph& graph,
    double severity,
    std::uint64_t seed
) {
    std::vector<Edge> edges =
        graph.get_edges();

    const std::size_t requested_count =
        perturbation_count(
            edges.size(),
            severity
        );

    const std::set<EdgePair> membership =
        edge_membership(
            edges
        );

    std::vector<EdgePair> candidates =
        nonedges(
            graph.num_vertices(),
            membership
        );

    std::vector<double> filtration_times;

    filtration_times.reserve(
        edges.size()
    );

    for (const Edge& edge :
         edges) {
        filtration_times.push_back(
            edge.time
        );
    }

    std::mt19937_64 generator(
        seed
    );

    portable_shuffle(
        candidates,
        generator
    );

    portable_shuffle(
        filtration_times,
        generator
    );

    const std::size_t insertion_count =
        std::min(
            requested_count,
            candidates.size()
        );

    edges.reserve(
        edges.size() +
        insertion_count
    );

    for (std::size_t index = 0U;
         index < insertion_count;
         ++index) {
        edges.push_back(
            Edge{
                candidates[
                    index
                ].first,
                candidates[
                    index
                ].second,
                0,
                filtration_times[
                    index
                ]
            }
        );
    }

    assign_filtration_steps(
        edges
    );

    return graph_from_edges(
        graph.num_vertices(),
        edges
    );
}

Graph apply_degree_preserving_rewiring(
    const Graph& graph,
    double severity,
    std::uint64_t seed
) {
    std::vector<Edge> edges =
        graph.get_edges();

    const std::size_t requested_swap_count =
        perturbation_count(
            edges.size(),
            severity
        );

    std::set<EdgePair> membership =
        edge_membership(
            edges
        );

    std::mt19937_64 generator(
        seed
    );

    for (std::size_t swap_index = 0U;
         swap_index <
             requested_swap_count;
         ++swap_index) {
        const std::vector<EdgeSwap> candidates =
            admissible_swaps(
                edges,
                membership
            );

        if (candidates.empty()) {
            break;
        }

        const EdgeSwap& selected =
            candidates[
                uniform_index(
                    generator,
                    candidates.size()
                )
            ];

        const Edge first =
            edges[
                selected.first_index
            ];

        const Edge second =
            edges[
                selected.second_index
            ];

        membership.erase(
            canonical_pair(
                first.u,
                first.v
            )
        );

        membership.erase(
            canonical_pair(
                second.u,
                second.v
            )
        );

        membership.insert(
            selected.first_replacement
        );

        membership.insert(
            selected.second_replacement
        );

        edges[
            selected.first_index
        ] =
            Edge{
                selected.
                    first_replacement.first,
                selected.
                    first_replacement.second,
                first.step,
                first.time
            };

        edges[
            selected.second_index
        ] =
            Edge{
                selected.
                    second_replacement.first,
                selected.
                    second_replacement.second,
                second.step,
                second.time
            };
    }

    return graph_from_edges(
        graph.num_vertices(),
        edges
    );
}

Graph apply_filtration_noise(
    const Graph& graph,
    double severity,
    std::uint64_t seed
) {
    std::vector<Edge> edges =
        graph.get_edges();

    std::mt19937_64 generator(
        seed
    );

    NormalGenerator normal(
        generator
    );

    for (Edge& edge :
         edges) {
        edge.time =
            std::clamp(
                edge.time +
                    severity *
                    normal.sample(),
                0.0,
                1.0
            );
    }

    assign_filtration_steps(
        edges
    );

    return graph_from_edges(
        graph.num_vertices(),
        edges
    );
}

}  // namespace vpd