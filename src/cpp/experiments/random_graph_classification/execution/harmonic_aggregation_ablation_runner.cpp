#include "harmonic_aggregation_ablation_runner.hpp"

#include "../../../core/utils/seed_derivation.hpp"
#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/random_fourier_features.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vpd {

namespace {

constexpr std::uint64_t
    kBaselineCharacterBankTag =
        0U;

std::vector<VPD1> extract_diagrams(
    const BenchmarkBatch& batch
) {
    std::vector<VPD1> diagrams;

    diagrams.reserve(
        batch.graphs.size()
    );

    for (const BenchmarkGraphRecord& record :
         batch.graphs) {
        diagrams.push_back(
            record.persistence_diagram
        );
    }

    return diagrams;
}

HarmonicValues evaluate_ablation(
    std::uint64_t root_seed,
    const ExecutionTask& task,
    const BenchmarkBatch& benchmark_batch,
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters,
    std::optional<std::uint64_t>& ablation_seed
) {
    switch (*task.ablation_kind) {
        case HarmonicAggregationAblationKind::
                CrossObservation: {
            const std::uint64_t seed =
                derive_seed(
                    root_seed,
                    SeedDomain::
                        CrossObservationAblation,
                    {
                        static_cast<std::uint64_t>(
                            task.model_index
                        ),
                        static_cast<std::uint64_t>(
                            benchmark_batch.
                                parameter_condition_index
                        ),
                        static_cast<std::uint64_t>(
                            task.batch_index
                        ),
                        static_cast<std::uint64_t>(
                            task.realization_index
                        )
                    }
                );

            ablation_seed =
                seed;

            return
                evaluate_cross_observation_ablation(
                    diagrams,
                    characters,
                    make_cross_observation_derangement(
                        diagrams.size(),
                        seed
                    )
                );
        }

        case HarmonicAggregationAblationKind::
                Linear:
            return evaluate_linear_ablation(
                diagrams,
                characters
            );

        case HarmonicAggregationAblationKind::
                NoPreorder:
            return evaluate_no_preorder_ablation(
                diagrams,
                characters
            );
    }

    throw std::logic_error(
        "Unknown harmonic-aggregation ablation kind."
    );
}

}  // namespace

std::size_t
HarmonicAggregationAblationBatch::graph_count() const {
    return benchmark_batch.graph_count();
}

HarmonicAggregationAblationBatch
generate_harmonic_aggregation_ablation_batch(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task
) {
    BenchmarkBatch benchmark_batch =
        generate_benchmark_batch(
            root_seed,
            specification,
            task.model_index,
            task.batch_index,
            specification.graphs_per_batch
        );

    const std::vector<VPD1> diagrams =
        extract_diagrams(
            benchmark_batch
        );

    const std::uint64_t character_seed =
        derive_seed(
            root_seed,
            SeedDomain::CharacterBank,
            {
                kBaselineCharacterBankTag
            }
        );

    const CharacterBank characters =
        sample_character_bank(
            canonical.
                random_features.
                baseline_character_count,
            character_seed
        );

    std::optional<std::uint64_t>
        ablation_seed;

    HarmonicValues harmonic_values =
        evaluate_ablation(
            root_seed,
            task,
            benchmark_batch,
            diagrams,
            characters,
            ablation_seed
        );

    return HarmonicAggregationAblationBatch{
        std::move(
            benchmark_batch
        ),
        *task.ablation_kind,
        task.realization_index,
        character_seed,
        ablation_seed,
        std::move(
            harmonic_values
        )
    };
}

void run_harmonic_aggregation_ablation_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const HarmonicAggregationAblationConsumer&
        consume_batch
) {
    consume_batch(
        generate_harmonic_aggregation_ablation_batch(
            root_seed,
            canonical,
            specification,
            task
        )
    );
}

}  // namespace vpd