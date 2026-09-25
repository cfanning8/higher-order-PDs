#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include "../../utils/portable_random.hpp"

#include <cstddef>
#include <cstdint>
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

GeneratedNetwork generate_configuration_model(
    const NetworkGenerationRequest& request,
    const ConfigurationModelParameters& parameters
) {
    if (
        parameters.degree <= 0 ||
        parameters.degree >=
            request.vertex_count
    ) {
        throw std::invalid_argument(
            "Configuration-model degree must lie in "
            "[1, vertex_count)."
        );
    }

    const std::int64_t stub_count =
        static_cast<std::int64_t>(
            request.vertex_count
        ) *
        static_cast<std::int64_t>(
            parameters.degree
        );

    if (
        stub_count % 2 != 0
    ) {
        throw std::invalid_argument(
            "A regular configuration model requires "
            "an even total stub count."
        );
    }

    const std::int64_t expected_edge_count =
        stub_count /
        2;

    if (
        expected_edge_count !=
        static_cast<std::int64_t>(
            request.edge_count
        )
    ) {
        throw std::invalid_argument(
            "Configuration-model degree is incompatible "
            "with the requested edge count."
        );
    }

    std::vector<int> base_stubs;

    base_stubs.reserve(
        static_cast<std::size_t>(
            stub_count
        )
    );

    for (int vertex = 0;
         vertex <
             request.vertex_count;
         ++vertex) {
        for (int copy = 0;
             copy <
                 parameters.degree;
             ++copy) {
            base_stubs.push_back(
                vertex
            );
        }
    }

    std::mt19937_64 generator(
        request.sampling_seed
    );

    std::uint64_t attempt =
        0U;

    std::vector<GeneratedEdge> edges;

    for (;;) {
        std::vector<int> stubs =
            base_stubs;

        portable_shuffle(
            stubs,
            generator
        );

        std::set<EdgePair> seen;

        edges.clear();

        edges.reserve(
            static_cast<std::size_t>(
                expected_edge_count
            )
        );

        bool simple =
            true;

        for (std::size_t index = 0U;
             index <
                 stubs.size();
             index += 2U) {
            const EdgePair edge =
                canonical_pair(
                    stubs[
                        index
                    ],
                    stubs[
                        index + 1U
                    ]
                );

            if (
                edge.first ==
                    edge.second ||
                !seen.insert(
                    edge
                ).second
            ) {
                simple =
                    false;

                break;
            }

            edges.push_back(
                GeneratedEdge{
                    edge.first,
                    edge.second,
                    static_cast<double>(
                        parameters.degree
                    ) /
                        static_cast<double>(
                            request.vertex_count -
                            1
                        ),
                    static_cast<std::uint64_t>(
                        index /
                        2U
                    )
                }
            );
        }

        if (simple) {
            break;
        }

        ++attempt;
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
                "degree",
                static_cast<double>(
                    parameters.degree
                )
            },
            {
                "rejection_attempt",
                static_cast<double>(
                    attempt
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