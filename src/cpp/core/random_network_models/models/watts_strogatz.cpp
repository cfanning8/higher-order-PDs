#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include "../../utils/portable_random.hpp"

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

namespace {

struct OrientedLatticeEdge {
    int source{};
    int target{};
};

}  // namespace

GeneratedNetwork generate_watts_strogatz(
    const NetworkGenerationRequest& request,
    const WattsStrogatzParameters& parameters
) {
    if (
        parameters.degree < 2 ||
        parameters.degree >=
            request.vertex_count ||
        parameters.degree % 2 != 0
    ) {
        throw std::invalid_argument(
            "Watts-Strogatz degree must be even and "
            "lie in [2, vertex_count)."
        );
    }

    if (
        parameters.rewiring_probability < 0.0 ||
        parameters.rewiring_probability > 1.0
    ) {
        throw std::invalid_argument(
            "Watts-Strogatz rewiring probability "
            "must lie in [0,1]."
        );
    }

    const std::int64_t expected_edge_count =
        static_cast<std::int64_t>(
            request.vertex_count
        ) *
        static_cast<std::int64_t>(
            parameters.degree
        ) /
        2;

    if (
        expected_edge_count !=
        static_cast<std::int64_t>(
            request.edge_count
        )
    ) {
        throw std::invalid_argument(
            "Watts-Strogatz parameters are incompatible "
            "with the requested edge count."
        );
    }

    const int half_degree =
        parameters.degree /
        2;

    std::vector<OrientedLatticeEdge>
        lattice_edges;

    lattice_edges.reserve(
        static_cast<std::size_t>(
            expected_edge_count
        )
    );

    std::set<EdgePair> current_edges;

    for (int offset = 1;
         offset <= half_degree;
         ++offset) {
        for (int source = 0;
             source <
                 request.vertex_count;
             ++source) {
            const int target =
                (
                    source +
                    offset
                ) %
                request.vertex_count;

            lattice_edges.push_back(
                OrientedLatticeEdge{
                    source,
                    target
                }
            );

            current_edges.insert(
                canonical_pair(
                    source,
                    target
                )
            );
        }
    }

    std::mt19937_64 generator(
        request.sampling_seed
    );

    std::vector<GeneratedEdge> edges;

    edges.reserve(
        lattice_edges.size()
    );

    std::map<EdgePair, double>
        rewired_by_edge;

    for (std::size_t index = 0U;
         index <
             lattice_edges.size();
         ++index) {
        const OrientedLatticeEdge& original =
            lattice_edges[
                index
            ];

        const EdgePair original_pair =
            canonical_pair(
                original.source,
                original.target
            );

        EdgePair final_pair =
            original_pair;

        double support =
            1.0 -
            parameters.
                rewiring_probability;

        double rewired =
            0.0;

        if (
            uniform_open_01(
                generator
            ) <
            parameters.
                rewiring_probability
        ) {
            current_edges.erase(
                original_pair
            );

            std::vector<int>
                admissible_targets;

            admissible_targets.reserve(
                static_cast<std::size_t>(
                    request.vertex_count -
                    1
                )
            );

            for (int target = 0;
                 target <
                     request.vertex_count;
                 ++target) {
                if (
                    target ==
                    original.source
                ) {
                    continue;
                }

                const EdgePair candidate =
                    canonical_pair(
                        original.source,
                        target
                    );

                if (
                    current_edges.find(
                        candidate
                    ) ==
                    current_edges.end()
                ) {
                    admissible_targets.
                        push_back(
                            target
                        );
                }
            }

            const int target =
                admissible_targets[
                    uniform_index(
                        generator,
                        admissible_targets.
                            size()
                    )
                ];

            final_pair =
                canonical_pair(
                    original.source,
                    target
                );

            support =
                parameters.
                    rewiring_probability /
                static_cast<double>(
                    admissible_targets.
                        size()
                );

            rewired =
                1.0;

            current_edges.insert(
                final_pair
            );
        }

        edges.push_back(
            GeneratedEdge{
                final_pair.first,
                final_pair.second,
                support,
                static_cast<std::uint64_t>(
                    index
                )
            }
        );

        rewired_by_edge[
            final_pair
        ] =
            rewired;
    }

    FiltrationResult filtration =
        build_filtration_graph(
            request.vertex_count,
            edges
        );

    std::vector<double> rewired_values;

    rewired_values.reserve(
        edges.size()
    );

    for (std::size_t source_index :
         filtration.source_indices) {
        const GeneratedEdge& edge =
            edges[
                source_index
            ];

        rewired_values.push_back(
            rewired_by_edge[
                canonical_pair(
                    edge.u,
                    edge.v
                )
            ]
        );
    }

    return GeneratedNetwork{
        std::move(
            filtration.graph
        ),
        {
            {
                "degree",
                static_cast<double>(
                    parameters.degree
                )
            },
            {
                "rewiring_probability",
                parameters.
                    rewiring_probability
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
                "rewired",
                std::move(
                    rewired_values
                )
            }
        }
    };
}

}  // namespace vpd