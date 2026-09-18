#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include "../../utils/portable_random.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vpd {

using random_network_models_detail::DyadProbability;
using random_network_models_detail::FiltrationResult;
using random_network_models_detail::GeneratedEdge;
using random_network_models_detail::build_filtration_graph;
using random_network_models_detail::ordered_edge_field;
using random_network_models_detail::power_law_weight_profile;
using random_network_models_detail::
    sample_conditional_bernoulli_edges;

namespace {

constexpr double kProbabilityCeiling =
    1.0 -
    1.0e-12;

double torus_distance(
    double first,
    double second
) {
    const double difference =
        std::abs(
            first -
            second
        );

    return
        std::min(
            difference,
            1.0 -
                difference
        );
}

double expected_edge_count(
    const std::vector<double>& weights,
    const std::vector<double>& positions,
    double connection_exponent,
    double scale
) {
    const double total_weight =
        std::accumulate(
            weights.begin(),
            weights.end(),
            0.0
        );

    double expected =
        0.0;

    for (std::size_t first = 0U;
         first <
             weights.size();
         ++first) {
        for (
            std::size_t second =
                first + 1U;
            second <
                weights.size();
            ++second
        ) {
            const double distance =
                torus_distance(
                    positions[
                        first
                    ],
                    positions[
                        second
                    ]
                );

            const double structural_term =
                std::pow(
                    weights[
                        first
                    ] *
                        weights[
                            second
                        ] /
                        (
                            total_weight *
                            distance
                        ),
                    connection_exponent
                );

            expected +=
                std::min(
                    kProbabilityCeiling,
                    scale *
                        structural_term
                );
        }
    }

    return expected;
}

}  // namespace

GeneratedNetwork generate_girg(
    const NetworkGenerationRequest& request,
    const GirgParameters& parameters
) {
    if (
        parameters.degree_exponent <= 1.0
    ) {
        throw std::invalid_argument(
            "GIRG degree exponent must exceed one."
        );
    }

    if (
        parameters.connection_exponent <= 0.0
    ) {
        throw std::invalid_argument(
            "GIRG connection exponent must be positive."
        );
    }

    std::vector<double> weights =
        power_law_weight_profile(
            request.vertex_count,
            parameters.degree_exponent
        );

    std::mt19937_64 latent_generator(
        request.latent_seed
    );

    portable_shuffle(
        weights,
        latent_generator
    );

    std::vector<double> positions(
        static_cast<std::size_t>(
            request.vertex_count
        )
    );

    for (double& position :
         positions) {
        position =
            uniform_open_01(
                latent_generator
            );
    }

    double lower =
        0.0;

    double upper =
        1.0;

    while (
        expected_edge_count(
            weights,
            positions,
            parameters.
                connection_exponent,
            upper
        ) <
        static_cast<double>(
            request.edge_count
        )
    ) {
        upper *=
            2.0;
    }

    for (int iteration = 0;
         iteration < 100;
         ++iteration) {
        const double middle =
            0.5 *
            (
                lower +
                upper
            );

        if (
            expected_edge_count(
                weights,
                positions,
                parameters.
                    connection_exponent,
                middle
            ) <
            static_cast<double>(
                request.edge_count
            )
        ) {
            lower =
                middle;
        } else {
            upper =
                middle;
        }
    }

    const double scale =
        0.5 *
        (
            lower +
            upper
        );

    const double total_weight =
        std::accumulate(
            weights.begin(),
            weights.end(),
            0.0
        );

    std::vector<DyadProbability> dyads;

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
         u <
             request.vertex_count;
         ++u) {
        for (int v = u + 1;
             v <
                 request.vertex_count;
             ++v) {
            const std::size_t first =
                static_cast<std::size_t>(
                    u
                );

            const std::size_t second =
                static_cast<std::size_t>(
                    v
                );

            const double distance =
                torus_distance(
                    positions[
                        first
                    ],
                    positions[
                        second
                    ]
                );

            const double probability =
                std::min(
                    kProbabilityCeiling,
                    scale *
                        std::pow(
                            weights[
                                first
                            ] *
                                weights[
                                    second
                                ] /
                                (
                                    total_weight *
                                    distance
                                ),
                            parameters.
                                connection_exponent
                        )
                );

            dyads.push_back(
                DyadProbability{
                    u,
                    v,
                    probability
                }
            );
        }
    }

    std::vector<GeneratedEdge> edges =
        sample_conditional_bernoulli_edges(
            std::move(
                dyads
            ),
            request.edge_count,
            request.sampling_seed
        );

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
                "degree_exponent",
                parameters.
                    degree_exponent
            },
            {
                "connection_exponent",
                parameters.
                    connection_exponent
            },
            {
                "connection_scale",
                scale
            }
        },
        {
            {
                "weight",
                std::move(
                    weights
                )
            },
            {
                "position",
                std::move(
                    positions
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