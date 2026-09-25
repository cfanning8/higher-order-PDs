#include "batch_size_sensitivity_runner.hpp"

#include "../experiment_design/batch_size_sensitivity.hpp"

#include "../../../core/utils/seed_derivation.hpp"
#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/random_fourier_features.hpp"

#include <cstddef>
#include <cstdint>
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

}  // namespace

void run_batch_size_sensitivity_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const BatchSizeSensitivityConsumer& consume_result
) {
    const BatchSizeSensitivityPlan plan =
        make_batch_size_sensitivity_plan(
            canonical.batch_size_sensitivity
        );

    BenchmarkBatch benchmark_batch =
        generate_benchmark_batch(
            root_seed,
            specification,
            task.model_index,
            task.batch_index,
            plan.maximum_graphs_per_batch
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

    std::vector<
        BatchSizeSensitivityConditionResult
    > conditions;

    conditions.reserve(
        canonical.
            batch_size_sensitivity.
            graphs_per_batch.size()
    );

    for (
        int graphs_per_batch :
        canonical.
            batch_size_sensitivity.
            graphs_per_batch
    ) {
        const std::vector<VPD1> prefix(
            diagrams.begin(),
            diagrams.begin() +
                static_cast<std::ptrdiff_t>(
                    graphs_per_batch
                )
        );

        conditions.push_back(
            BatchSizeSensitivityConditionResult{
                graphs_per_batch,
                mean_pair_fourier(
                    prefix,
                    characters
                )
            }
        );
    }

    consume_result(
        BatchSizeSensitivityResult{
            std::move(
                benchmark_batch
            ),
            character_seed,
            std::move(
                conditions
            )
        }
    );
}

}  // namespace vpd