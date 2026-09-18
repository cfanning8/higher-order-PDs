#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vpd {

using random_network_models_detail::FiltrationResult;
using random_network_models_detail::GeneratedEdge;
using random_network_models_detail::build_filtration_graph;
using random_network_models_detail::ordered_edge_field;
using random_network_models_detail::sample_weighted_index;

GeneratedNetwork generate_barabasi_albert(
    const NetworkGenerationRequest& request,
    const BarabasiAlbertParameters& parameters
) {
    if (
        parameters.initial_clique_size < 2 ||
        parameters.initial_clique_size >
            request.vertex_count
    ) {
        throw std::invalid_argument(
            "Barabasi-Albert initial clique size must "
            "lie in [2, vertex_count]."
        );
    }

    if (
        parameters.attachment_count < 1 ||
        parameters.attachment_count >
            parameters.initial_clique_size
    ) {
        throw std::invalid_argument(
            "Barabasi-Albert attachment count must "
            "lie in [1, initial_clique_size]."
        );
    }

    const std::int64_t clique_size =
        static_cast<std::int64_t>(
            parameters.
                initial_clique_size
        );

    const std::int64_t attachment_count =
        static_cast<std::int64_t>(
            parameters.
                attachment_count
        );

    const std::int64_t vertex_count =
        static_cast<std::int64_t>(
            request.vertex_count
        );

    const std::int64_t expected_edge_count =
        clique_size *
            (
                clique_size -
                1
            ) /
            2 +
        attachment_count *
            (
                vertex_count -
                clique_size
            );

    if (
        expected_edge_count !=
        static_cast<std::int64_t>(
            request.edge_count
        )
    ) {
        throw std::invalid_argument(
            "Barabasi-Albert parameters are incompatible "
            "with the requested edge count."
        );
    }

    std::mt19937_64 generator(
        request.sampling_seed
    );

    std::vector<int> degrees(
        static_cast<std::size_t>(
            request.vertex_count
        ),
        0
    );

    std::vector<double> arrival_index(
        static_cast<std::size_t>(
            request.vertex_count
        )
    );

    for (int vertex = 0;
         vertex <
             request.vertex_count;
         ++vertex) {
        arrival_index[
            static_cast<std::size_t>(
                vertex
            )
        ] =
            static_cast<double>(
                vertex
            );
    }

    std::vector<GeneratedEdge> edges;

    edges.reserve(
        static_cast<std::size_t>(
            expected_edge_count
        )
    );

    std::uint64_t formation_index =
        0U;

    for (int u = 0;
         u <
             parameters.
                 initial_clique_size;
         ++u) {
        for (
            int v = u + 1;
            v <
                parameters.
                    initial_clique_size;
            ++v
        ) {
            edges.push_back(
                GeneratedEdge{
                    u,
                    v,
                    1.0,
                    formation_index
                }
            );

            ++formation_index;

            ++degrees[
                static_cast<std::size_t>(
                    u
                )
            ];

            ++degrees[
                static_cast<std::size_t>(
                    v
                )
            ];
        }
    }

    for (
        int new_vertex =
            parameters.
                initial_clique_size;
        new_vertex <
            request.vertex_count;
        ++new_vertex
    ) {
        std::vector<int> candidates(
            static_cast<std::size_t>(
                new_vertex
            )
        );

        std::iota(
            candidates.begin(),
            candidates.end(),
            0
        );

        for (
            int attachment = 0;
            attachment <
                parameters.
                    attachment_count;
            ++attachment
        ) {
            std::vector<double> weights;

            weights.reserve(
                candidates.size()
            );

            for (int candidate :
                 candidates) {
                weights.push_back(
                    static_cast<double>(
                        degrees[
                            static_cast<
                                std::size_t
                            >(
                                candidate
                            )
                        ]
                    )
                );
            }

            const double total_weight =
                std::accumulate(
                    weights.begin(),
                    weights.end(),
                    0.0
                );

            const std::size_t selected_index =
                sample_weighted_index(
                    generator,
                    weights
                );

            const int target =
                candidates[
                    selected_index
                ];

            const double probability =
                weights[
                    selected_index
                ] /
                total_weight;

            edges.push_back(
                GeneratedEdge{
                    new_vertex,
                    target,
                    probability,
                    formation_index
                }
            );

            ++formation_index;

            ++degrees[
                static_cast<std::size_t>(
                    new_vertex
                )
            ];

            ++degrees[
                static_cast<std::size_t>(
                    target
                )
            ];

            candidates.erase(
                candidates.begin() +
                static_cast<std::ptrdiff_t>(
                    selected_index
                )
            );
        }
    }

    FiltrationResult filtration =
        build_filtration_graph(
            request.vertex_count,
            edges
        );

    return GeneratedNetwork{
        std::move(
            filtration.graph
        ),
        {
            {
                "attachment_count",
                static_cast<double>(
                    parameters.
                        attachment_count
                )
            },
            {
                "initial_clique_size",
                static_cast<double>(
                    parameters.
                        initial_clique_size
                )
            }
        },
        {
            {
                "arrival_index",
                std::move(
                    arrival_index
                )
            }
        },
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
                "attachment_probability",
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
            }
        }
    };
}

}  // namespace vpd