#ifndef GRAPH_BENCHMARK_EXPERIMENT_CONFIG_HPP
#define GRAPH_BENCHMARK_EXPERIMENT_CONFIG_HPP

#include "../../../core/random_network_models/random_network_models.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace vpd {

inline constexpr int
kBaselineAggregationOrder = 2;

inline constexpr int
kBaselineHighestExplicitOrder = 1;

enum class RobustnessPerturbation {
    EdgeDeletion,
    EdgeInsertion,
    DegreePreservingRewiring,
    FiltrationNoise
};

struct GraphSizeSpecification {
    int vertex_count{};

    int edge_count{};
};

struct ModelParameterSpecification {
    std::vector<WattsStrogatzParameters>
        watts_strogatz;

    std::vector<BarabasiAlbertParameters>
        barabasi_albert;

    std::vector<ConfigurationModelParameters>
        configuration_model;

    std::vector<StochasticBlockModelParameters>
        stochastic_block_model;

    std::vector<ChungLuParameters>
        chung_lu;

    std::vector<KleinbergParameters>
        kleinberg;

    std::vector<GirgParameters>
        girg;

    std::vector<HyperbolicRandomGraphParameters>
        hyperbolic_random_graph;

    std::vector<ExponentialRandomGraphModelParameters>
        exponential_random_graph_model;
};

struct BenchmarkDesignSpecification {
    GraphSizeSpecification graph_size;

    int batches_per_model{};

    int graphs_per_batch{};

    ModelParameterSpecification
        model_parameters;
};

struct RandomFeatureSpecification {
    int baseline_character_count{};

    int replicate_count{};

    std::vector<int>
        character_counts;
};

struct RobustnessCondition {
    RobustnessPerturbation perturbation{};

    double severity{};
};

struct RobustnessSpecification {
    int realization_count{};

    std::vector<RobustnessCondition>
        conditions;
};

struct BatchSizeSensitivitySpecification {
    std::vector<int>
        graphs_per_batch;
};

struct ScalabilitySpecification {
    std::vector<int>
        graphs_per_batch;

    std::vector<std::size_t>
        input_support_sizes;

    std::vector<int>
        character_counts;

    int repeat_count{};
};

struct CanonicalExperimentSpecification {
    BenchmarkDesignSpecification benchmark;

    RandomFeatureSpecification
        random_features;

    RobustnessSpecification
        robustness;

    BatchSizeSensitivitySpecification
        batch_size_sensitivity;

    ScalabilitySpecification
        scalability;

    int cross_observation_ablation_realizations{};

    std::uint64_t root_seed{};
};

struct ExperimentConfig {
    std::filesystem::path root;

    std::filesystem::path output_root;

    std::optional<std::uint64_t>
        root_seed_override;

    double maximum_estimated_output_gib{};

    double estimated_diagram_atoms_per_graph{};

    bool resume{};

    bool force{};

    bool allow_compiler_mismatch{};

    bool allow_build_type_mismatch{};

    bool dry_run{};
};

const CanonicalExperimentSpecification&
canonical_experiment_specification();

std::uint64_t experiment_root_seed(
    const ExperimentConfig& config
);

ExperimentConfig default_experiment_config();

void validate_experiment_config(
    ExperimentConfig& config
);

}  // namespace vpd

#endif