#include "rff_sensitivity_runner.hpp"

#include "../experiment_design/rff_sensitivity.hpp"

#include "../../../core/utils/seed_derivation.hpp"
#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/random_fourier_features.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vpd {

namespace {

constexpr std::uint64_t
    kRffSensitivityCharacterBankTag =
        1U;

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

CharacterBank character_bank_prefix(
    const CharacterBank& bank,
    int character_count
) {
    CharacterBank prefix;

    prefix.values.assign(
        bank.values.begin(),
        bank.values.begin() +
            static_cast<std::ptrdiff_t>(
                character_count
            )
    );

    return prefix;
}

}  // namespace

void run_rff_sensitivity_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const RffSensitivityConsumer& consume_result
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

    const RffSensitivityPlan plan =
        make_rff_sensitivity_plan(
            canonical.random_features
        );

    const std::uint64_t character_seed =
        derive_seed(
            root_seed,
            SeedDomain::CharacterBank,
            {
                kRffSensitivityCharacterBankTag,
                static_cast<std::uint64_t>(
                    task.replicate_index
                )
            }
        );

    const CharacterBank maximal_bank =
        sample_character_bank(
            plan.maximum_character_count,
            character_seed
        );

    std::vector<
        RffSensitivityConditionResult
    > conditions;

    conditions.reserve(
        canonical.
            random_features.
            character_counts.size()
    );

    for (
        int character_count :
        canonical.
            random_features.
            character_counts
    ) {
        const CharacterBank characters =
            character_bank_prefix(
                maximal_bank,
                character_count
            );

        conditions.push_back(
            RffSensitivityConditionResult{
                character_count,
                mean_pair_fourier(
                    diagrams,
                    characters
                )
            }
        );
    }

    consume_result(
        RffSensitivityResult{
            std::move(
                benchmark_batch
            ),
            task.replicate_index,
            character_seed,
            std::move(
                conditions
            )
        }
    );
}

}  // namespace vpd