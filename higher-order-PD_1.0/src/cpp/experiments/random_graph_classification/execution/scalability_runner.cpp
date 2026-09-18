#include "scalability_runner.hpp"

#include "../../../core/utils/portable_random.hpp"
#include "../../../core/utils/seed_derivation.hpp"
#include "../../../core/utils/timer.hpp"
#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/random_fourier_features.hpp"
#include "../../../methods/TDA/persistent_homology/diagrams.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace vpd {

namespace {

constexpr std::uint64_t
    kScalabilityCharacterBankTag =
        2U;

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

std::size_t maximum_input_support_size(
    const CanonicalExperimentSpecification& canonical
) {
    return
        *std::max_element(
            canonical.
                scalability.
                input_support_sizes.begin(),
            canonical.
                scalability.
                input_support_sizes.end()
        );
}

int maximum_character_count(
    const CanonicalExperimentSpecification& canonical
) {
    return
        *std::max_element(
            canonical.
                scalability.
                character_counts.begin(),
            canonical.
                scalability.
                character_counts.end()
        );
}

VPD1 make_scalability_diagram(
    std::uint64_t sample_seed,
    int graph_index,
    std::size_t support_size,
    std::size_t maximum_support_size
) {
    const std::uint64_t graph_seed =
        derive_seed(
            sample_seed,
            SeedDomain::Scalability,
            {
                static_cast<std::uint64_t>(
                    graph_index
                )
            }
        );

    std::mt19937_64 engine(
        graph_seed
    );

    std::vector<std::size_t> permutation(
        maximum_support_size
    );

    std::iota(
        permutation.begin(),
        permutation.end(),
        std::size_t{0}
    );

    portable_shuffle(
        permutation,
        engine
    );

    const double denominator =
        static_cast<double>(
            maximum_support_size +
            1U
        );

    VPD1 diagram;

    for (std::size_t atom_index = 0U;
         atom_index <
             support_size;
         ++atom_index) {
        const double birth =
            0.25 *
            static_cast<double>(
                atom_index +
                1U
            ) /
            denominator;

        const double death =
            0.75 +
            0.25 *
            static_cast<double>(
                permutation[
                    atom_index
                ] +
                1U
            ) /
            denominator;

        add_to_diagram(
            diagram,
            Atom1(
                birth,
                death
            ),
            1.0
        );
    }

    return diagram;
}

std::vector<VPD1> make_scalability_diagrams(
    std::uint64_t sample_seed,
    const ScalabilityCondition& condition,
    std::size_t maximum_support_size
) {
    std::vector<VPD1> diagrams;

    diagrams.reserve(
        static_cast<std::size_t>(
            condition.graphs_per_batch
        )
    );

    for (int graph_index = 0;
         graph_index <
             condition.graphs_per_batch;
         ++graph_index) {
        diagrams.push_back(
            make_scalability_diagram(
                sample_seed,
                graph_index,
                condition.input_support_size,
                maximum_support_size
            )
        );
    }

    return diagrams;
}

}  // namespace

ScalabilityResult execute_scalability_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionTask& task
) {
    const ScalabilityPlan plan =
        make_scalability_plan(
            canonical.scalability
        );

    const ScalabilityCondition condition =
        plan.conditions[
            static_cast<std::size_t>(
                task.scalability_condition_index
            )
        ];

    const std::uint64_t sample_seed =
        derive_seed(
            root_seed,
            SeedDomain::Scalability,
            {
                static_cast<std::uint64_t>(
                    condition.repeat_index
                )
            }
        );

    const std::vector<VPD1> diagrams =
        make_scalability_diagrams(
            sample_seed,
            condition,
            maximum_input_support_size(
                canonical
            )
        );

    const std::uint64_t character_seed =
        derive_seed(
            root_seed,
            SeedDomain::CharacterBank,
            {
                kScalabilityCharacterBankTag,
                static_cast<std::uint64_t>(
                    condition.repeat_index
                )
            }
        );

    const CharacterBank maximal_bank =
        sample_character_bank(
            maximum_character_count(
                canonical
            ),
            character_seed
        );

    const CharacterBank characters =
        character_bank_prefix(
            maximal_bank,
            condition.character_count
        );

    Timer timer;

    HarmonicValues harmonic_values =
        mean_pair_fourier(
            diagrams,
            characters
        );

    const double representation_seconds =
        timer.elapsed_seconds();

    return ScalabilityResult{
        condition,
        sample_seed,
        character_seed,
        std::move(
            harmonic_values
        ),
        ScalabilityComputationMetrics{
            representation_seconds
        }
    };
}

void run_scalability_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionTask& task,
    const ScalabilityResultConsumer& consume_result
) {
    consume_result(
        execute_scalability_task(
            root_seed,
            canonical,
            task
        )
    );
}

}  // namespace vpd