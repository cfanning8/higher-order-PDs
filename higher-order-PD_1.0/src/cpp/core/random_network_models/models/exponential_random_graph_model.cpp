#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include "../../utils/portable_random.hpp"
#include "../../utils/seed_derivation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace vpd {

using random_network_models_detail::EdgePair;
using random_network_models_detail::FiltrationResult;
using random_network_models_detail::GeneratedEdge;
using random_network_models_detail::build_filtration_graph;
using random_network_models_detail::ordered_edge_field;

namespace {

constexpr std::uint64_t kInitialGraphDomain =
    0x6572676d5f696e69ULL;

constexpr std::uint64_t kChainDomain =
    0x6572676d5f636861ULL;

using AdjacencyMatrix =
    std::vector<std::vector<unsigned char>>;

using SharedPartnerMatrix =
    std::vector<std::vector<int>>;

double gwesp_weight(
    int shared_partners,
    double decay
) {
    if (shared_partners == 0) {
        return 0.0;
    }

    const double base =
        1.0 -
        std::exp(
            -decay
        );

    return
        std::exp(
            decay
        ) *
        (
            1.0 -
            std::pow(
                base,
                static_cast<double>(
                    shared_partners
                )
            )
        );
}

std::vector<double> make_gwesp_weights(
    int vertex_count,
    double decay
) {
    std::vector<double> weights(
        static_cast<std::size_t>(
            vertex_count
        )
    );

    for (int shared_partners = 0;
         shared_partners <
             vertex_count;
         ++shared_partners) {
        weights[
            static_cast<std::size_t>(
                shared_partners
            )
        ] =
            gwesp_weight(
                shared_partners,
                decay
            );
    }

    return weights;
}

SharedPartnerMatrix make_shared_partner_matrix(
    const AdjacencyMatrix& adjacency
) {
    const std::size_t vertex_count =
        adjacency.size();

    SharedPartnerMatrix shared_partners(
        vertex_count,
        std::vector<int>(
            vertex_count,
            0
        )
    );

    for (std::size_t u = 0U;
         u < vertex_count;
         ++u) {
        for (std::size_t v = u + 1U;
             v < vertex_count;
             ++v) {
            int count =
                0;

            for (std::size_t w = 0U;
                 w < vertex_count;
                 ++w) {
                if (
                    adjacency[u][w] != 0U &&
                    adjacency[v][w] != 0U
                ) {
                    ++count;
                }
            }

            shared_partners[u][v] =
                count;

            shared_partners[v][u] =
                count;
        }
    }

    return shared_partners;
}

double deletion_delta(
    const AdjacencyMatrix& adjacency,
    const SharedPartnerMatrix&
        shared_partners,
    const std::vector<double>&
        gwesp_weights,
    int u,
    int v
) {
    const std::size_t u_index =
        static_cast<std::size_t>(
            u
        );

    const std::size_t v_index =
        static_cast<std::size_t>(
            v
        );

    const int shared_uv =
        shared_partners[
            u_index
        ][
            v_index
        ];

    double delta =
        -gwesp_weights[
            static_cast<std::size_t>(
                shared_uv
            )
        ];

    for (std::size_t vertex = 0U;
         vertex < adjacency.size();
         ++vertex) {
        if (
            adjacency[
                u_index
            ][vertex] == 0U ||
            adjacency[
                v_index
            ][vertex] == 0U
        ) {
            continue;
        }

        const int shared_uw =
            shared_partners[
                u_index
            ][vertex];

        const int shared_vw =
            shared_partners[
                v_index
            ][vertex];

        delta +=
            gwesp_weights[
                static_cast<std::size_t>(
                    shared_uw - 1
                )
            ] -
            gwesp_weights[
                static_cast<std::size_t>(
                    shared_uw
                )
            ];

        delta +=
            gwesp_weights[
                static_cast<std::size_t>(
                    shared_vw - 1
                )
            ] -
            gwesp_weights[
                static_cast<std::size_t>(
                    shared_vw
                )
            ];
    }

    return delta;
}

double addition_delta(
    const AdjacencyMatrix& adjacency,
    const SharedPartnerMatrix&
        shared_partners,
    const std::vector<double>&
        gwesp_weights,
    int u,
    int v
) {
    const std::size_t u_index =
        static_cast<std::size_t>(
            u
        );

    const std::size_t v_index =
        static_cast<std::size_t>(
            v
        );

    const int shared_uv =
        shared_partners[
            u_index
        ][
            v_index
        ];

    double delta =
        gwesp_weights[
            static_cast<std::size_t>(
                shared_uv
            )
        ];

    for (std::size_t vertex = 0U;
         vertex < adjacency.size();
         ++vertex) {
        if (
            adjacency[
                u_index
            ][vertex] == 0U ||
            adjacency[
                v_index
            ][vertex] == 0U
        ) {
            continue;
        }

        const int shared_uw =
            shared_partners[
                u_index
            ][vertex];

        const int shared_vw =
            shared_partners[
                v_index
            ][vertex];

        delta +=
            gwesp_weights[
                static_cast<std::size_t>(
                    shared_uw + 1
                )
            ] -
            gwesp_weights[
                static_cast<std::size_t>(
                    shared_uw
                )
            ];

        delta +=
            gwesp_weights[
                static_cast<std::size_t>(
                    shared_vw + 1
                )
            ] -
            gwesp_weights[
                static_cast<std::size_t>(
                    shared_vw
                )
            ];
    }

    return delta;
}

void set_edge_state(
    AdjacencyMatrix& adjacency,
    SharedPartnerMatrix& shared_partners,
    const EdgePair& edge,
    bool present
) {
    const std::size_t u =
        static_cast<std::size_t>(
            edge.first
        );

    const std::size_t v =
        static_cast<std::size_t>(
            edge.second
        );

    if (present) {
        for (std::size_t w = 0U;
             w < adjacency.size();
             ++w) {
            if (
                w == u ||
                w == v
            ) {
                continue;
            }

            if (
                adjacency[u][w] != 0U
            ) {
                ++shared_partners[v][w];
                ++shared_partners[w][v];
            }

            if (
                adjacency[v][w] != 0U
            ) {
                ++shared_partners[u][w];
                ++shared_partners[w][u];
            }
        }

        adjacency[u][v] =
            1U;

        adjacency[v][u] =
            1U;

        return;
    }

    for (std::size_t w = 0U;
         w < adjacency.size();
         ++w) {
        if (
            w == u ||
            w == v
        ) {
            continue;
        }

        if (
            adjacency[u][w] != 0U
        ) {
            --shared_partners[v][w];
            --shared_partners[w][v];
        }

        if (
            adjacency[v][w] != 0U
        ) {
            --shared_partners[u][w];
            --shared_partners[w][u];
        }
    }

    adjacency[u][v] =
        0U;

    adjacency[v][u] =
        0U;
}

}  // namespace

GeneratedNetwork
generate_exponential_random_graph_model(
    const NetworkGenerationRequest& request,
    const ExponentialRandomGraphModelParameters&
        parameters
) {
    std::vector<EdgePair> dyads;

    dyads.reserve(
        static_cast<std::size_t>(
            request.vertex_count *
            (
                request.vertex_count -
                1
            ) /
            2
        )
    );

    for (int u = 0;
         u < request.vertex_count;
         ++u) {
        for (int v = u + 1;
             v < request.vertex_count;
             ++v) {
            dyads.emplace_back(
                u,
                v
            );
        }
    }

    std::mt19937_64 initial_generator(
        splitmix64(
            request.sampling_seed ^
            kInitialGraphDomain
        )
    );

    portable_shuffle(
        dyads,
        initial_generator
    );

    std::vector<EdgePair> present(
        dyads.begin(),
        dyads.begin() +
            static_cast<std::ptrdiff_t>(
                request.edge_count
            )
    );

    std::vector<EdgePair> absent(
        dyads.begin() +
            static_cast<std::ptrdiff_t>(
                request.edge_count
            ),
        dyads.end()
    );

    AdjacencyMatrix adjacency(
        static_cast<std::size_t>(
            request.vertex_count
        ),
        std::vector<unsigned char>(
            static_cast<std::size_t>(
                request.vertex_count
            ),
            0U
        )
    );

    std::map<EdgePair, std::uint64_t>
        insertion_index;

    std::uint64_t next_insertion_index =
        0U;

    for (const EdgePair& edge :
         present) {
        const std::size_t u =
            static_cast<std::size_t>(
                edge.first
            );

        const std::size_t v =
            static_cast<std::size_t>(
                edge.second
            );

        adjacency[u][v] =
            1U;

        adjacency[v][u] =
            1U;

        insertion_index[
            edge
        ] =
            next_insertion_index;

        ++next_insertion_index;
    }

    SharedPartnerMatrix shared_partners =
        make_shared_partner_matrix(
            adjacency
        );

    const std::vector<double> gwesp_weights =
        make_gwesp_weights(
            request.vertex_count,
            parameters.gwesp_decay
        );

    std::mt19937_64 chain_generator(
        splitmix64(
            request.sampling_seed ^
            kChainDomain
        )
    );

    const std::uint64_t total_sweeps =
        static_cast<std::uint64_t>(
            parameters.
                burn_in_sweeps +
            parameters.
                sampling_sweeps
        );

    const std::uint64_t proposal_count =
        total_sweeps *
        static_cast<std::uint64_t>(
            request.edge_count
        );

    std::uint64_t accepted_swaps =
        0U;

    for (std::uint64_t proposal = 0U;
         proposal < proposal_count;
         ++proposal) {
        const std::size_t present_index =
            uniform_index(
                chain_generator,
                present.size()
            );

        const std::size_t absent_index =
            uniform_index(
                chain_generator,
                absent.size()
            );

        const EdgePair removed =
            present[
                present_index
            ];

        const EdgePair added =
            absent[
                absent_index
            ];

        const double remove_change =
            deletion_delta(
                adjacency,
                shared_partners,
                gwesp_weights,
                removed.first,
                removed.second
            );

        set_edge_state(
            adjacency,
            shared_partners,
            removed,
            false
        );

        const double add_change =
            addition_delta(
                adjacency,
                shared_partners,
                gwesp_weights,
                added.first,
                added.second
            );

        const double statistic_change =
            remove_change +
            add_change;

        const double log_acceptance =
            parameters.
                gwesp_coefficient *
            statistic_change;

        const bool accept =
            log_acceptance >= 0.0 ||
            uniform_open_01(
                chain_generator
            ) <
                std::exp(
                    log_acceptance
                );

        if (!accept) {
            set_edge_state(
                adjacency,
                shared_partners,
                removed,
                true
            );

            continue;
        }

        set_edge_state(
            adjacency,
            shared_partners,
            added,
            true
        );

        present[
            present_index
        ] =
            added;

        absent[
            absent_index
        ] =
            removed;

        insertion_index.erase(
            removed
        );

        insertion_index[
            added
        ] =
            next_insertion_index;

        ++next_insertion_index;
        ++accepted_swaps;
    }

    std::vector<GeneratedEdge> edges;

    edges.reserve(
        present.size()
    );

    for (const EdgePair& edge :
         present) {
        const double deletion_change =
            deletion_delta(
                adjacency,
                shared_partners,
                gwesp_weights,
                edge.first,
                edge.second
            );

        const double support =
            -parameters.
                 gwesp_coefficient *
            deletion_change;

        edges.push_back(
            GeneratedEdge{
                edge.first,
                edge.second,
                support,
                insertion_index[
                    edge
                ]
            }
        );
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
                "gwesp_coefficient",
                parameters.
                    gwesp_coefficient
            },
            {
                "gwesp_decay",
                parameters.
                    gwesp_decay
            },
            {
                "burn_in_sweeps",
                static_cast<double>(
                    parameters.
                        burn_in_sweeps
                )
            },
            {
                "sampling_sweeps",
                static_cast<double>(
                    parameters.
                        sampling_sweeps
                )
            },
            {
                "accepted_swaps",
                static_cast<double>(
                    accepted_swaps
                )
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
            }
        }
    };
}

}  // namespace vpd