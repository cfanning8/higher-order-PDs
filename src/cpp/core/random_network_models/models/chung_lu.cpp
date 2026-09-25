#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include "../../utils/portable_random.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <random>
#include <tuple>
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

using ChungLuCalibrationKey =
    std::tuple<
        int,
        int,
        double
    >;

struct ChungLuCalibration {
    std::vector<double> base_weights;
    double base_total{};
    double scale{};
};

double expected_edge_count(
    const std::vector<double>& base_weights,
    double base_total,
    double scale
) {
    double expected =
        0.0;

    for (std::size_t first = 0U;
         first < base_weights.size();
         ++first) {
        for (std::size_t second =
                 first + 1U;
             second <
                 base_weights.size();
             ++second) {
            expected +=
                std::min(
                    kProbabilityCeiling,
                    scale *
                        base_weights[first] *
                        base_weights[second] /
                        base_total
                );
        }
    }

    return expected;
}

ChungLuCalibration make_chung_lu_calibration(
    int vertex_count,
    int edge_count,
    double degree_exponent
) {
    std::vector<double> base_weights =
        power_law_weight_profile(
            vertex_count,
            degree_exponent
        );

    const double base_total =
        std::accumulate(
            base_weights.begin(),
            base_weights.end(),
            0.0
        );

    const double target =
        static_cast<double>(
            edge_count
        );

    double lower =
        0.0;

    double upper =
        1.0;

    while (
        expected_edge_count(
            base_weights,
            base_total,
            upper
        ) <
        target
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
                base_weights,
                base_total,
                middle
            ) <
            target
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

    return ChungLuCalibration{
        std::move(
            base_weights
        ),
        base_total,
        scale
    };
}

const ChungLuCalibration& chung_lu_calibration(
    int vertex_count,
    int edge_count,
    double degree_exponent
) {
    thread_local std::map<
        ChungLuCalibrationKey,
        ChungLuCalibration
    > cache;

    const ChungLuCalibrationKey key{
        vertex_count,
        edge_count,
        degree_exponent
    };

    const auto found =
        cache.find(
            key
        );

    if (
        found !=
        cache.end()
    ) {
        return found->second;
    }

    auto inserted =
        cache.emplace(
            key,
            make_chung_lu_calibration(
                vertex_count,
                edge_count,
                degree_exponent
            )
        );

    return inserted.first->second;
}

}  // namespace

GeneratedNetwork generate_chung_lu(
    const NetworkGenerationRequest& request,
    const ChungLuParameters& parameters
) {
    const ChungLuCalibration& calibration =
        chung_lu_calibration(
            request.vertex_count,
            request.edge_count,
            parameters.degree_exponent
        );

    std::vector<double> weights =
        calibration.base_weights;

    std::mt19937_64 latent_generator(
        request.latent_seed
    );

    portable_shuffle(
        weights,
        latent_generator
    );

    for (double& weight :
         weights) {
        weight *=
            calibration.scale;
    }

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
         u < request.vertex_count;
         ++u) {
        for (int v = u + 1;
             v < request.vertex_count;
             ++v) {
            const double probability =
                std::min(
                    kProbabilityCeiling,
                    weights[
                        static_cast<std::size_t>(
                            u
                        )
                    ] *
                        weights[
                            static_cast<std::size_t>(
                                v
                            )
                        ] /
                        total_weight
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
                "weight_scale",
                calibration.scale
            }
        },
        {
            {
                "expected_degree_weight",
                std::move(
                    weights
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