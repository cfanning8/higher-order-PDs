#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include "../../utils/portable_random.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace vpd {

using random_network_models_detail::DyadProbability;
using random_network_models_detail::FiltrationResult;
using random_network_models_detail::GeneratedEdge;
using random_network_models_detail::build_filtration_graph;
using random_network_models_detail::logistic;
using random_network_models_detail::ordered_edge_field;
using random_network_models_detail::
    sample_conditional_bernoulli_edges;

GeneratedNetwork generate_stochastic_block_model(
    const NetworkGenerationRequest& request,
    const StochasticBlockModelParameters& parameters
) {
    std::vector<int> ordering(
        static_cast<std::size_t>(
            request.vertex_count
        )
    );

    std::iota(
        ordering.begin(),
        ordering.end(),
        0
    );

    std::mt19937_64 latent_generator(
        request.latent_seed
    );

    portable_shuffle(
        ordering,
        latent_generator
    );

    std::vector<int> block(
        static_cast<std::size_t>(
            request.vertex_count
        )
    );

    for (int position = 0;
         position <
             request.vertex_count;
         ++position) {
        block[
            static_cast<std::size_t>(
                ordering[
                    static_cast<std::size_t>(
                        position
                    )
                ]
            )
        ] =
            position *
                parameters.block_count /
            request.vertex_count;
    }

    const auto expected_edge_count =
        [&](double intercept) {
            double expected =
                0.0;

            for (int u = 0;
                 u < request.vertex_count;
                 ++u) {
                for (int v = u + 1;
                     v <
                         request.vertex_count;
                     ++v) {
                    const bool same =
                        block[
                            static_cast<std::size_t>(
                                u
                            )
                        ] ==
                        block[
                            static_cast<std::size_t>(
                                v
                            )
                        ];

                    const double logit =
                        intercept +
                        (
                            same
                                ? 0.5
                                : -0.5
                        ) *
                            parameters.
                                log_odds_contrast;

                    expected +=
                        logistic(
                            logit
                        );
                }
            }

            return expected;
        };

    double lower =
        -30.0;

    double upper =
        30.0;

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

    const double intercept =
        0.5 *
        (
            lower +
            upper
        );

    const double p_in =
        logistic(
            intercept +
            0.5 *
                parameters.
                    log_odds_contrast
        );

    const double p_out =
        logistic(
            intercept -
            0.5 *
                parameters.
                    log_odds_contrast
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
         u < request.vertex_count;
         ++u) {
        for (int v = u + 1;
             v < request.vertex_count;
             ++v) {
            dyads.push_back(
                DyadProbability{
                    u,
                    v,
                    block[
                        static_cast<std::size_t>(
                            u
                        )
                    ] ==
                        block[
                            static_cast<std::size_t>(
                                v
                            )
                        ]
                        ? p_in
                        : p_out
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

    std::vector<double> block_values(
        block.size()
    );

    for (std::size_t index = 0U;
         index < block.size();
         ++index) {
        block_values[index] =
            static_cast<double>(
                block[index]
            );
    }

    return GeneratedNetwork{
        std::move(
            filtration.graph
        ),
        {
            {
                "block_count",
                static_cast<double>(
                    parameters.block_count
                )
            },
            {
                "log_odds_contrast",
                parameters.
                    log_odds_contrast
            },
            {
                "logit_intercept",
                intercept
            },
            {
                "within_block_probability",
                p_in
            },
            {
                "between_block_probability",
                p_out
            }
        },
        {
            {
                "block",
                std::move(
                    block_values
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