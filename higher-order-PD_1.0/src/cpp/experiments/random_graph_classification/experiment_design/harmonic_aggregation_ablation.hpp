#ifndef GRAPH_BENCHMARK_HARMONIC_AGGREGATION_ABLATION_HPP
#define GRAPH_BENCHMARK_HARMONIC_AGGREGATION_ABLATION_HPP

#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/aggregation.hpp"
#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/random_fourier_features.hpp"
#include "../../../methods/TDA/persistent_homology/diagrams.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpd {

enum class HarmonicAggregationAblationKind {
    CrossObservation,
    Linear,
    NoPreorder
};

using DiagramPermutation =
    std::vector<std::size_t>;

const char* harmonic_aggregation_ablation_name(
    HarmonicAggregationAblationKind kind
);

DiagramPermutation
make_cross_observation_derangement(
    std::size_t diagram_count,
    std::uint64_t seed
);

HarmonicValues
evaluate_cross_observation_ablation(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters,
    const DiagramPermutation& permutation
);

HarmonicValues
evaluate_linear_ablation(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters
);

HarmonicValues
evaluate_no_preorder_ablation(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters
);

}  // namespace vpd

#endif