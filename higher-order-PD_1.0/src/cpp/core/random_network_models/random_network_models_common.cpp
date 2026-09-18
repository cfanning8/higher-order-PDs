#include "random_network_models_common.hpp"

#include "../utils/portable_random.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace vpd::random_network_models_detail {

namespace {

double log_add_exp(
    double first,
    double second
) {
    if (
        first ==
        -std::numeric_limits<double>::infinity()
    ) {
        return second;
    }

    if (
        second ==
        -std::numeric_limits<double>::infinity()
    ) {
        return first;
    }

    const double maximum =
        std::max(
            first,
            second
        );

    return
        maximum +
        std::log(
            std::exp(
                first -
                maximum
            ) +
            std::exp(
                second -
                maximum
            )
        );
}

double log_odds(
    double probability
) {
    return
        std::log(
            probability
        ) -
        std::log1p(
            -probability
        );
}

}  // namespace

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

std::size_t sample_weighted_index(
    std::mt19937_64& generator,
    const std::vector<double>& weights
) {
    const double total =
        std::accumulate(
            weights.begin(),
            weights.end(),
            0.0
        );

    const double threshold =
        uniform_open_01(
            generator
        ) *
        total;

    double cumulative =
        0.0;

    for (std::size_t index = 0U;
         index < weights.size();
         ++index) {
        cumulative +=
            weights[index];

        if (
            threshold <
            cumulative
        ) {
            return index;
        }
    }

    return
        weights.size() -
        1U;
}

double logistic(
    double value
) {
    if (value >= 0.0) {
        const double exponential =
            std::exp(
                -value
            );

        return
            1.0 /
            (
                1.0 +
                exponential
            );
    }

    const double exponential =
        std::exp(
            value
        );

    return
        exponential /
        (
            1.0 +
            exponential
        );
}

FiltrationResult build_filtration_graph(
    int vertex_count,
    const std::vector<GeneratedEdge>& edges
) {
    std::vector<std::size_t> order(
        edges.size()
    );

    std::iota(
        order.begin(),
        order.end(),
        std::size_t{0}
    );

    std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t first_index,
            std::size_t second_index) {
            const GeneratedEdge& first =
                edges[first_index];

            const GeneratedEdge& second =
                edges[second_index];

            if (
                first.support !=
                second.support
            ) {
                return
                    first.support >
                    second.support;
            }

            if (
                first.formation_index !=
                second.formation_index
            ) {
                return
                    first.formation_index <
                    second.formation_index;
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

    Graph graph(
        vertex_count
    );

    graph.reserve_edges(
        edges.size()
    );

    const double denominator =
        edges.size() > 1U
            ? static_cast<double>(
                  edges.size() -
                  1U
              )
            : 1.0;

    for (std::size_t rank = 0U;
         rank < order.size();
         ++rank) {
        const GeneratedEdge& edge =
            edges[
                order[rank]
            ];

        graph.add_edge(
            edge.u,
            edge.v,
            static_cast<int>(
                rank + 1U
            ),
            edges.size() > 1U
                ? static_cast<double>(
                      rank
                  ) /
                      denominator
                : 0.0
        );
    }

    return FiltrationResult{
        std::move(
            graph
        ),
        std::move(
            order
        )
    };
}

std::vector<GeneratedEdge>
sample_conditional_bernoulli_edges(
    std::vector<DyadProbability> dyads,
    int edge_count,
    std::uint64_t seed
) {
    std::mt19937_64 generator(
        seed
    );

    portable_shuffle(
        dyads,
        generator
    );

    const std::size_t dyad_count =
        dyads.size();

    const std::size_t target =
        static_cast<std::size_t>(
            edge_count
        );

    const double negative_infinity =
        -std::numeric_limits<double>::infinity();

    std::vector<std::vector<double>>
        suffix(
            dyad_count + 1U,
            std::vector<double>(
                target + 1U,
                negative_infinity
            )
        );

    suffix[
        dyad_count
    ][0U] =
        0.0;

    for (std::size_t reverse = dyad_count;
         reverse > 0U;
         --reverse) {
        const std::size_t index =
            reverse -
            1U;

        const double odds =
            log_odds(
                dyads[index].
                    probability
            );

        suffix[index][0U] =
            0.0;

        const std::size_t maximum_selected =
            std::min(
                target,
                dyad_count -
                    index
            );

        for (std::size_t selected = 1U;
             selected <=
                 maximum_selected;
             ++selected) {
            suffix[index][selected] =
                log_add_exp(
                    suffix[
                        index + 1U
                    ][selected],
                    odds +
                        suffix[
                            index + 1U
                        ][
                            selected -
                            1U
                        ]
                );
        }
    }

    std::vector<GeneratedEdge> result;

    result.reserve(
        target
    );

    std::size_t remaining =
        target;

    std::uint64_t formation_index =
        0U;

    for (std::size_t index = 0U;
         index < dyad_count &&
             remaining > 0U;
         ++index) {
        const std::size_t available =
            dyad_count -
            index;

        bool include =
            remaining ==
            available;

        if (!include) {
            const double numerator =
                log_odds(
                    dyads[index].
                        probability
                ) +
                suffix[
                    index + 1U
                ][
                    remaining -
                    1U
                ];

            const double denominator =
                suffix[
                    index
                ][remaining];

            const double probability =
                std::exp(
                    numerator -
                    denominator
                );

            include =
                uniform_open_01(
                    generator
                ) <
                probability;
        }

        if (!include) {
            continue;
        }

        result.push_back(
            GeneratedEdge{
                dyads[index].u,
                dyads[index].v,
                dyads[index].
                    probability,
                formation_index
            }
        );

        ++formation_index;
        --remaining;
    }

    return result;
}

std::vector<double>
power_law_weight_profile(
    int vertex_count,
    double exponent
) {
    std::vector<double> weights(
        static_cast<std::size_t>(
            vertex_count
        )
    );

    const double reciprocal_exponent =
        1.0 /
        (
            exponent -
            1.0
        );

    for (int index = 0;
         index < vertex_count;
         ++index) {
        weights[
            static_cast<std::size_t>(
                index
            )
        ] =
            std::pow(
                static_cast<double>(
                    vertex_count
                ) /
                    (
                        static_cast<double>(
                            index
                        ) +
                        0.5
                    ),
                reciprocal_exponent
            );
    }

    return weights;
}

std::vector<double>
ordered_edge_field(
    const std::vector<GeneratedEdge>& edges,
    const FiltrationResult& filtration,
    bool support
) {
    std::vector<double> values;

    values.reserve(
        edges.size()
    );

    for (std::size_t source_index :
         filtration.source_indices) {
        const GeneratedEdge& edge =
            edges[
                source_index
            ];

        values.push_back(
            support
                ? edge.support
                : static_cast<double>(
                      edge.formation_index
                  )
        );
    }

    return values;
}

}  // namespace vpd::random_network_models_detail