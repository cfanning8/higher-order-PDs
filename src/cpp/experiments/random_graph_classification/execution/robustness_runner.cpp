#include "robustness_runner.hpp"

#include "../experiment_design/robustness.hpp"

#include "../../../core/utils/seed_derivation.hpp"
#include "../../../methods/TDA/persistent_homology/persistence.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vpd {

namespace {

Graph apply_robustness_condition(
    const Graph& graph,
    const RobustnessCondition& condition,
    std::uint64_t seed
) {
    switch (condition.perturbation) {
        case RobustnessPerturbation::EdgeDeletion:
            return apply_edge_deletion(
                graph,
                condition.severity,
                seed
            );

        case RobustnessPerturbation::EdgeInsertion:
            return apply_edge_insertion(
                graph,
                condition.severity,
                seed
            );

        case RobustnessPerturbation::
                DegreePreservingRewiring:
            return apply_degree_preserving_rewiring(
                graph,
                condition.severity,
                seed
            );

        case RobustnessPerturbation::FiltrationNoise:
            return apply_filtration_noise(
                graph,
                condition.severity,
                seed
            );
    }

    throw std::logic_error(
        "Unknown robustness perturbation."
    );
}

VPD1 compute_persistence_diagram(
    const Graph& graph
) {
    VPD1 diagram =
        compute_H1_persistence(
            graph
        );

    clean_diagram(
        diagram
    );

    return diagram;
}

RobustnessGraphRecord generate_robustness_graph(
    std::uint64_t root_seed,
    const BenchmarkGraphRecord& clean_record,
    int condition_index,
    const RobustnessCondition& condition,
    int realization_index
) {
    const GraphKey& key =
        clean_record.key;

    const std::uint64_t perturbation_seed =
        derive_seed(
            root_seed,
            SeedDomain::Robustness,
            {
                static_cast<std::uint64_t>(
                    key.model_index
                ),
                static_cast<std::uint64_t>(
                    key.parameter_condition_index
                ),
                static_cast<std::uint64_t>(
                    key.batch_index
                ),
                static_cast<std::uint64_t>(
                    key.graph_index
                ),
                static_cast<std::uint64_t>(
                    condition_index
                ),
                static_cast<std::uint64_t>(
                    realization_index
                )
            }
        );

    Graph perturbed_graph =
        apply_robustness_condition(
            clean_record.network.graph,
            condition,
            perturbation_seed
        );

    VPD1 persistence_diagram =
        compute_persistence_diagram(
            perturbed_graph
        );

    return RobustnessGraphRecord{
        key,
        perturbation_seed,
        std::move(
            perturbed_graph
        ),
        std::move(
            persistence_diagram
        )
    };
}

}  // namespace

std::size_t RobustnessBatch::graph_count() const {
    return perturbed_graphs.size();
}

RobustnessBatch generate_robustness_batch(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task
) {
    const int condition_index =
        task.robustness_condition_index;

    const RobustnessCondition condition =
        canonical.robustness.conditions[
            static_cast<std::size_t>(
                condition_index
            )
        ];

    BenchmarkBatch clean_batch =
        generate_benchmark_batch(
            root_seed,
            specification,
            task.model_index,
            task.batch_index,
            specification.graphs_per_batch
        );

    std::vector<RobustnessGraphRecord>
        perturbed_graphs;

    perturbed_graphs.reserve(
        clean_batch.graphs.size()
    );

    for (const BenchmarkGraphRecord& clean_record :
         clean_batch.graphs) {
        perturbed_graphs.push_back(
            generate_robustness_graph(
                root_seed,
                clean_record,
                condition_index,
                condition,
                task.realization_index
            )
        );
    }

    return RobustnessBatch{
        std::move(
            clean_batch
        ),
        condition_index,
        condition,
        task.realization_index,
        std::move(
            perturbed_graphs
        )
    };
}

void run_robustness_task(
    std::uint64_t root_seed,
    const CanonicalExperimentSpecification& canonical,
    const BenchmarkSpecification& specification,
    const ExecutionTask& task,
    const RobustnessBatchConsumer& consume_batch
) {
    consume_batch(
        generate_robustness_batch(
            root_seed,
            canonical,
            specification,
            task
        )
    );
}

}  // namespace vpd