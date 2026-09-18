#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include "../../utils/portable_random.hpp"

#include <algorithm>
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
using random_network_models_detail::ordered_edge_field;
using random_network_models_detail::
    sample_conditional_bernoulli_edges;

namespace {

constexpr double kPi =
    3.141592653589793238462643383279502884;

constexpr double kProbabilityCeiling =
    1.0 -
    1.0e-12;

double angular_distance(
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
            2.0 *
                kPi -
                difference
        );
}

double hyperbolic_distance(
    double first_radius,
    double first_angle,
    double second_radius,
    double second_angle,
    double curvature_scale
) {
    const double scaled_first_radius =
        curvature_scale *
        first_radius;

    const double scaled_second_radius =
        curvature_scale *
        second_radius;

    const double argument =
        std::cosh(
            scaled_first_radius
        ) *
            std::cosh(
                scaled_second_radius
            ) -
        std::sinh(
            scaled_first_radius
        ) *
            std::sinh(
                scaled_second_radius
            ) *
            std::cos(
                angular_distance(
                    first_angle,
                    second_angle
                )
            );

    return
        std::acosh(
            std::max(
                1.0,
                argument
            )
        ) /
        curvature_scale;
}

}  // namespace

GeneratedNetwork generate_hyperbolic_random_graph(
    const NetworkGenerationRequest& request,
    const HyperbolicRandomGraphParameters& parameters
) {
    const double curvature_scale =
        std::sqrt(
            -parameters.curvature
        );

    const double radial_alpha =
        0.5 *
        curvature_scale *
        (
            parameters.
                degree_exponent -
            1.0
        );

    std::mt19937_64 generator(
        request.latent_seed
    );

    std::vector<double> radii(
        static_cast<std::size_t>(
            request.vertex_count
        )
    );

    std::vector<double> angles(
        static_cast<std::size_t>(
            request.vertex_count
        )
    );

    const double radial_normalizer =
        std::cosh(
            radial_alpha *
            parameters.radius
        ) -
        1.0;

    for (int vertex = 0;
         vertex < request.vertex_count;
         ++vertex) {
        const std::size_t index =
            static_cast<std::size_t>(
                vertex
            );

        const double radial_uniform =
            uniform_open_01(
                generator
            );

        radii[index] =
            std::acosh(
                1.0 +
                radial_uniform *
                    radial_normalizer
            ) /
            radial_alpha;

        angles[index] =
            2.0 *
            kPi *
            uniform_open_01(
                generator
            );
    }

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
            const std::size_t first =
                static_cast<std::size_t>(
                    u
                );

            const std::size_t second =
                static_cast<std::size_t>(
                    v
                );

            const double distance =
                hyperbolic_distance(
                    radii[first],
                    angles[first],
                    radii[second],
                    angles[second],
                    curvature_scale
                );

            const double exponent =
                curvature_scale *
                (
                    distance -
                    parameters.radius
                ) /
                (
                    2.0 *
                    parameters.temperature
                );

            const double probability =
                std::min(
                    kProbabilityCeiling,
                    1.0 /
                        (
                            1.0 +
                            std::exp(
                                exponent
                            )
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
                "temperature",
                parameters.temperature
            },
            {
                "curvature",
                parameters.curvature
            },
            {
                "radius",
                parameters.radius
            }
        },
        {
            {
                "radius",
                std::move(
                    radii
                )
            },
            {
                "angle",
                std::move(
                    angles
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