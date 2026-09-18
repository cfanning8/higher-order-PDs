#include "output_layout.hpp"

#include <filesystem>
#include <utility>

namespace vpd {

OutputLayout::OutputLayout(
    std::filesystem::path root
)
    : root_(
          std::filesystem::absolute(
              std::move(
                  root
              )
          ).lexically_normal()
      ) {
}

OutputLayout OutputLayout::from_config(
    const ExperimentConfig& config
) {
    return OutputLayout(
        config.output_root
    );
}

const std::filesystem::path&
OutputLayout::root() const {
    return root_;
}

std::filesystem::path
OutputLayout::manifest_json() const {
    return
        root_ /
        "manifest.json";
}

std::filesystem::path
OutputLayout::configuration_json() const {
    return
        root_ /
        "configuration.json";
}

std::filesystem::path
OutputLayout::models_csv() const {
    return
        root_ /
        "models.csv";
}

std::filesystem::path
OutputLayout::model_parameters_csv() const {
    return
        root_ /
        "model_parameters.csv";
}

std::filesystem::path
OutputLayout::benchmark_directory() const {
    return
        root_ /
        "benchmark";
}

std::filesystem::path
OutputLayout::benchmark_batches_csv() const {
    return
        benchmark_directory() /
        "batches.csv";
}

std::filesystem::path
OutputLayout::benchmark_graphs_csv() const {
    return
        benchmark_directory() /
        "graphs.csv";
}

std::filesystem::path
OutputLayout::benchmark_graph_fields_csv() const {
    return
        benchmark_directory() /
        "graph_fields.csv";
}

std::filesystem::path
OutputLayout::benchmark_vertex_fields_csv() const {
    return
        benchmark_directory() /
        "vertex_fields.csv";
}

std::filesystem::path
OutputLayout::benchmark_edge_fields_csv() const {
    return
        benchmark_directory() /
        "edge_fields.csv";
}

std::filesystem::path
OutputLayout::benchmark_persistence_diagrams_csv() const {
    return
        benchmark_directory() /
        "persistence_diagrams.csv";
}

std::filesystem::path
OutputLayout::benchmark_graph_statistics_csv() const {
    return
        benchmark_directory() /
        "graph_statistics.csv";
}

std::filesystem::path
OutputLayout::benchmark_persistence_landscapes_csv() const {
    return
        benchmark_directory() /
        "persistence_landscapes.csv";
}

std::filesystem::path
OutputLayout::benchmark_persistence_images_csv() const {
    return
        benchmark_directory() /
        "persistence_images.csv";
}

std::filesystem::path
OutputLayout::benchmark_harmonic_aggregation_csv() const {
    return
        benchmark_directory() /
        "harmonic_aggregation.csv";
}

std::filesystem::path
OutputLayout::
harmonic_aggregation_ablation_directory() const {
    return
        root_ /
        "harmonic_aggregation_ablation";
}

std::filesystem::path
OutputLayout::
harmonic_aggregation_ablation_conditions_csv() const {
    return
        harmonic_aggregation_ablation_directory() /
        "conditions.csv";
}

std::filesystem::path
OutputLayout::
harmonic_aggregation_ablation_features_csv() const {
    return
        harmonic_aggregation_ablation_directory() /
        "harmonic_aggregation.csv";
}

std::filesystem::path
OutputLayout::robustness_directory() const {
    return
        root_ /
        "robustness";
}

std::filesystem::path
OutputLayout::robustness_conditions_csv() const {
    return
        robustness_directory() /
        "conditions.csv";
}

std::filesystem::path
OutputLayout::robustness_graphs_csv() const {
    return
        robustness_directory() /
        "graphs.csv";
}

std::filesystem::path
OutputLayout::robustness_persistence_diagrams_csv() const {
    return
        robustness_directory() /
        "persistence_diagrams.csv";
}

std::filesystem::path
OutputLayout::robustness_graph_statistics_csv() const {
    return
        robustness_directory() /
        "graph_statistics.csv";
}

std::filesystem::path
OutputLayout::
robustness_persistence_landscapes_csv() const {
    return
        robustness_directory() /
        "persistence_landscapes.csv";
}

std::filesystem::path
OutputLayout::robustness_persistence_images_csv() const {
    return
        robustness_directory() /
        "persistence_images.csv";
}

std::filesystem::path
OutputLayout::robustness_harmonic_aggregation_csv() const {
    return
        robustness_directory() /
        "harmonic_aggregation.csv";
}

std::filesystem::path
OutputLayout::batch_size_sensitivity_directory() const {
    return
        root_ /
        "batch_size_sensitivity";
}

std::filesystem::path
OutputLayout::
batch_size_sensitivity_conditions_csv() const {
    return
        batch_size_sensitivity_directory() /
        "conditions.csv";
}

std::filesystem::path
OutputLayout::
batch_size_sensitivity_graph_membership_csv() const {
    return
        batch_size_sensitivity_directory() /
        "graph_membership.csv";
}

std::filesystem::path
OutputLayout::
batch_size_sensitivity_graphs_csv() const {
    return
        batch_size_sensitivity_directory() /
        "graphs.csv";
}

std::filesystem::path
OutputLayout::
batch_size_sensitivity_persistence_diagrams_csv() const {
    return
        batch_size_sensitivity_directory() /
        "persistence_diagrams.csv";
}

std::filesystem::path
OutputLayout::
batch_size_sensitivity_harmonic_aggregation_csv() const {
    return
        batch_size_sensitivity_directory() /
        "harmonic_aggregation.csv";
}

std::filesystem::path
OutputLayout::rff_sensitivity_directory() const {
    return
        root_ /
        "rff_sensitivity";
}

std::filesystem::path
OutputLayout::rff_sensitivity_conditions_csv() const {
    return
        rff_sensitivity_directory() /
        "conditions.csv";
}

std::filesystem::path
OutputLayout::rff_sensitivity_harmonic_aggregation_csv() const {
    return
        rff_sensitivity_directory() /
        "harmonic_aggregation.csv";
}

std::filesystem::path
OutputLayout::scalability_directory() const {
    return
        root_ /
        "scalability";
}

std::filesystem::path
OutputLayout::scalability_conditions_csv() const {
    return
        scalability_directory() /
        "conditions.csv";
}

std::filesystem::path
OutputLayout::scalability_timings_csv() const {
    return
        scalability_directory() /
        "timings.csv";
}

std::filesystem::path
OutputLayout::scalability_harmonic_aggregation_csv() const {
    return
        scalability_directory() /
        "harmonic_aggregation.csv";
}

void OutputLayout::create_directories() const {
    std::filesystem::create_directories(
        root_
    );

    std::filesystem::create_directories(
        benchmark_directory()
    );

    std::filesystem::create_directories(
        harmonic_aggregation_ablation_directory()
    );

    std::filesystem::create_directories(
        robustness_directory()
    );

    std::filesystem::create_directories(
        batch_size_sensitivity_directory()
    );

    std::filesystem::create_directories(
        rff_sensitivity_directory()
    );

    std::filesystem::create_directories(
        scalability_directory()
    );
}

}  // namespace vpd