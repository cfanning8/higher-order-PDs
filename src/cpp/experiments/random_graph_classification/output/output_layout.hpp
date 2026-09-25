#ifndef GRAPH_BENCHMARK_OUTPUT_LAYOUT_HPP
#define GRAPH_BENCHMARK_OUTPUT_LAYOUT_HPP

#include "../configuration/experiment_config.hpp"

#include <filesystem>

namespace vpd {

class OutputLayout {
public:
    explicit OutputLayout(
        std::filesystem::path root
    );

    static OutputLayout from_config(
        const ExperimentConfig& config
    );

    const std::filesystem::path& root() const;

    std::filesystem::path manifest_json() const;

    std::filesystem::path configuration_json() const;

    std::filesystem::path models_csv() const;

    std::filesystem::path model_parameters_csv() const;

    std::filesystem::path
    benchmark_directory() const;

    std::filesystem::path
    benchmark_batches_csv() const;

    std::filesystem::path
    benchmark_graphs_csv() const;

    std::filesystem::path
    benchmark_graph_fields_csv() const;

    std::filesystem::path
    benchmark_vertex_fields_csv() const;

    std::filesystem::path
    benchmark_edge_fields_csv() const;

    std::filesystem::path
    benchmark_persistence_diagrams_csv() const;

    std::filesystem::path
    benchmark_graph_statistics_csv() const;

    std::filesystem::path
    benchmark_persistence_landscapes_csv() const;

    std::filesystem::path
    benchmark_persistence_images_csv() const;

    std::filesystem::path
    benchmark_harmonic_aggregation_csv() const;

    std::filesystem::path
    harmonic_aggregation_ablation_directory() const;

    std::filesystem::path
    harmonic_aggregation_ablation_conditions_csv() const;

    std::filesystem::path
    harmonic_aggregation_ablation_features_csv() const;

    std::filesystem::path
    robustness_directory() const;

    std::filesystem::path
    robustness_conditions_csv() const;

    std::filesystem::path
    robustness_graphs_csv() const;

    std::filesystem::path
    robustness_persistence_diagrams_csv() const;

    std::filesystem::path
    robustness_graph_statistics_csv() const;

    std::filesystem::path
    robustness_persistence_landscapes_csv() const;

    std::filesystem::path
    robustness_persistence_images_csv() const;

    std::filesystem::path
    robustness_harmonic_aggregation_csv() const;

    std::filesystem::path
    batch_size_sensitivity_directory() const;

    std::filesystem::path
    batch_size_sensitivity_conditions_csv() const;

    std::filesystem::path
    batch_size_sensitivity_graph_membership_csv() const;

    std::filesystem::path
    batch_size_sensitivity_graphs_csv() const;

    std::filesystem::path
    batch_size_sensitivity_persistence_diagrams_csv() const;

    std::filesystem::path
    batch_size_sensitivity_harmonic_aggregation_csv() const;

    std::filesystem::path
    rff_sensitivity_directory() const;

    std::filesystem::path
    rff_sensitivity_conditions_csv() const;

    std::filesystem::path
    rff_sensitivity_harmonic_aggregation_csv() const;

    std::filesystem::path
    scalability_directory() const;

    std::filesystem::path
    scalability_conditions_csv() const;

    std::filesystem::path
    scalability_timings_csv() const;

    std::filesystem::path
    scalability_harmonic_aggregation_csv() const;

    void create_directories() const;

private:
    std::filesystem::path root_;
};

}  // namespace vpd

#endif