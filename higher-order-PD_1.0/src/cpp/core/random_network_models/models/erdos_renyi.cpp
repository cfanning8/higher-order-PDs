#include "../random_network_models.hpp"
#include "../random_network_models_common.hpp"

#include "../../utils/portable_random.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace vpd {

using random_network_models_detail::EdgePair;
using random_network_models_detail::FiltrationResult;
using random_network_models_detail::GeneratedEdge;
using random_network_models_detail::build_filtration_graph;
using random_network_models_detail::ordered_edge_field;

GeneratedNetwork generate_erdos_renyi(
    const NetworkGenerationRequest& request
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

    std::mt19937_64 generator(
        request.sampling_seed
    );

    portable_shuffle(
        dyads,
        generator
    );

    const double support =
        static_cast<double>(
            request.edge_count
        ) /
        static_cast<double>(
            dyads.size()
        );

    std::vector<GeneratedEdge> edges;

    edges.reserve(
        static_cast<std::size_t>(
            request.edge_count
        )
    );

    for (int index = 0;
         index < request.edge_count;
         ++index) {
        const EdgePair& edge =
            dyads[
                static_cast<std::size_t>(
                    index
                )
            ];

        edges.push_back(
            GeneratedEdge{
                edge.first,
                edge.second,
                support,
                static_cast<std::uint64_t>(
                    index
                )
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
                "gnm_edge_probability",
                support
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