#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vpd {

using random_network_models_detail::EdgePair;
using random_network_models_detail::FiltrationResult;
using random_network_models_detail::GeneratedEdge;
using random_network_models_detail::build_filtration_graph;
using random_network_models_detail::canonical_pair;
using random_network_models_detail::ordered_edge_field;
using random_network_models_detail::sample_weighted_index;

namespace {

struct LongRangeCandidate {
    EdgePair edge;
    double weight{};
    int distance{};
};

int grid_distance(
    int first,
    int second,
    int columns
) {
    const int first_row =
        first /
        columns;

    const int first_column =
        first %
        columns;

    const int second_row =
        second /
        columns;

    const int second_column =
        second %
        columns;

    return
        std::abs(
            first_row -
            second_row
        ) +
        std::abs(
            first_column -
            second_column
        );
}

}  // namespace

GeneratedNetwork generate_kleinberg(
    const NetworkGenerationRequest& request,
    const KleinbergParameters& parameters
) {
    if (
        parameters.rows <= 0 ||
        parameters.columns <= 0
    ) {
        throw std::invalid_argument(
            "Kleinberg grid dimensions must be positive."
        );
    }

    const std::int64_t grid_vertex_count =
        static_cast<std::int64_t>(
            parameters.rows
        ) *
        static_cast<std::int64_t>(
            parameters.columns
        );

    if (
        grid_vertex_count !=
        static_cast<std::int64_t>(
            request.vertex_count
        )
    ) {
        throw std::invalid_argument(
            "Kleinberg grid dimensions are incompatible "
            "with the requested vertex count."
        );
    }

    if (
        parameters.long_range_edge_count < 0
    ) {
        throw std::invalid_argument(
            "Kleinberg long-range edge count must be "
            "nonnegative."
        );
    }

    if (
        parameters.distance_exponent <= 0.0
    ) {
        throw std::invalid_argument(
            "Kleinberg distance exponent must be positive."
        );
    }

    const std::int64_t local_edge_count =
        static_cast<std::int64_t>(
            parameters.rows
        ) *
            static_cast<std::int64_t>(
                parameters.columns -
                1
            ) +
        static_cast<std::int64_t>(
            parameters.rows -
            1
        ) *
            static_cast<std::int64_t>(
                parameters.columns
            );

    const std::int64_t dyad_count =
        grid_vertex_count *
            (
                grid_vertex_count -
                1
            ) /
        2;

    const std::int64_t available_long_range_edges =
        dyad_count -
        local_edge_count;

    if (
        static_cast<std::int64_t>(
            parameters.
                long_range_edge_count
        ) >
        available_long_range_edges
    ) {
        throw std::invalid_argument(
            "Kleinberg long-range edge count exceeds "
            "the number of nonlocal dyads."
        );
    }

    const std::int64_t expected_edge_count =
        local_edge_count +
        static_cast<std::int64_t>(
            parameters.
                long_range_edge_count
        );

    if (
        expected_edge_count !=
        static_cast<std::int64_t>(
            request.edge_count
        )
    ) {
        throw std::invalid_argument(
            "Kleinberg parameters are incompatible "
            "with the requested edge count."
        );
    }

    std::vector<GeneratedEdge> edges;

    edges.reserve(
        static_cast<std::size_t>(
            expected_edge_count
        )
    );

    std::set<EdgePair> local_edges;

    std::map<EdgePair, double>
        local_indicator;

    std::map<EdgePair, double>
        distances;

    std::uint64_t formation_index =
        0U;

    const auto vertex_index =
        [&](int row, int column) {
            return
                row *
                    parameters.columns +
                column;
        };

    for (int row = 0;
         row <
             parameters.rows;
         ++row) {
        for (int column = 0;
             column <
                 parameters.columns;
             ++column) {
            const int vertex =
                vertex_index(
                    row,
                    column
                );

            if (
                column + 1 <
                parameters.columns
            ) {
                const EdgePair edge =
                    canonical_pair(
                        vertex,
                        vertex_index(
                            row,
                            column + 1
                        )
                    );

                local_edges.insert(
                    edge
                );

                edges.push_back(
                    GeneratedEdge{
                        edge.first,
                        edge.second,
                        1.0,
                        formation_index
                    }
                );

                ++formation_index;

                local_indicator[
                    edge
                ] =
                    1.0;

                distances[
                    edge
                ] =
                    1.0;
            }

            if (
                row + 1 <
                parameters.rows
            ) {
                const EdgePair edge =
                    canonical_pair(
                        vertex,
                        vertex_index(
                            row + 1,
                            column
                        )
                    );

                local_edges.insert(
                    edge
                );

                edges.push_back(
                    GeneratedEdge{
                        edge.first,
                        edge.second,
                        1.0,
                        formation_index
                    }
                );

                ++formation_index;

                local_indicator[
                    edge
                ] =
                    1.0;

                distances[
                    edge
                ] =
                    1.0;
            }
        }
    }

    std::vector<LongRangeCandidate>
        candidates;

    candidates.reserve(
        static_cast<std::size_t>(
            available_long_range_edges
        )
    );

    for (int u = 0;
         u <
             request.vertex_count;
         ++u) {
        for (int v = u + 1;
             v <
                 request.vertex_count;
             ++v) {
            const EdgePair edge{
                u,
                v
            };

            if (
                local_edges.find(
                    edge
                ) !=
                local_edges.end()
            ) {
                continue;
            }

            const int distance =
                grid_distance(
                    u,
                    v,
                    parameters.columns
                );

            candidates.push_back(
                LongRangeCandidate{
                    edge,
                    std::pow(
                        static_cast<double>(
                            distance
                        ),
                        -parameters.
                            distance_exponent
                    ),
                    distance
                }
            );
        }
    }

    std::mt19937_64 generator(
        request.sampling_seed
    );

    for (
        int draw = 0;
        draw <
            parameters.
                long_range_edge_count;
        ++draw
    ) {
        std::vector<double> weights;

        weights.reserve(
            candidates.size()
        );

        double total_weight =
            0.0;

        for (
            const LongRangeCandidate& candidate :
            candidates
        ) {
            weights.push_back(
                candidate.weight
            );

            total_weight +=
                candidate.weight;
        }

        const std::size_t selected_index =
            sample_weighted_index(
                generator,
                weights
            );

        const LongRangeCandidate selected =
            candidates[
                selected_index
            ];

        edges.push_back(
            GeneratedEdge{
                selected.edge.first,
                selected.edge.second,
                selected.weight /
                    total_weight,
                formation_index
            }
        );

        ++formation_index;

        local_indicator[
            selected.edge
        ] =
            0.0;

        distances[
            selected.edge
        ] =
            static_cast<double>(
                selected.distance
            );

        candidates.erase(
            candidates.begin() +
            static_cast<std::ptrdiff_t>(
                selected_index
            )
        );
    }

    FiltrationResult filtration =
        build_filtration_graph(
            request.vertex_count,
            edges
        );

    std::vector<double>
        ordered_local_indicator;

    std::vector<double>
        ordered_distances;

    ordered_local_indicator.reserve(
        edges.size()
    );

    ordered_distances.reserve(
        edges.size()
    );

    for (std::size_t source_index :
         filtration.source_indices) {
        const GeneratedEdge& edge =
            edges[
                source_index
            ];

        const EdgePair pair =
            canonical_pair(
                edge.u,
                edge.v
            );

        ordered_local_indicator.
            push_back(
                local_indicator[
                    pair
                ]
            );

        ordered_distances.
            push_back(
                distances[
                    pair
                ]
            );
    }

    return GeneratedNetwork{
        std::move(
            filtration.graph
        ),
        {
            {
                "rows",
                static_cast<double>(
                    parameters.rows
                )
            },
            {
                "columns",
                static_cast<double>(
                    parameters.columns
                )
            },
            {
                "long_range_edge_count",
                static_cast<double>(
                    parameters.
                        long_range_edge_count
                )
            },
            {
                "distance_exponent",
                parameters.
                    distance_exponent
            }
        },
        {},
        {
            {
                "support",
                ordered_edge_field(
                    edges,
                    filtration,
                    true
                )
            },
            {
                "formation_index",
                ordered_edge_field(
                    edges,
                    filtration,
                    false
                )
            },
            {
                "local_edge",
                std::move(
                    ordered_local_indicator
                )
            },
            {
                "grid_distance",
                std::move(
                    ordered_distances
                )
            }
        }
    };
}

}  // namespace vpd