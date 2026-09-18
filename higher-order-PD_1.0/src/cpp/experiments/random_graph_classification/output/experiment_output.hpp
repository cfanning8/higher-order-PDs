#ifndef GRAPH_BENCHMARK_EXPERIMENT_OUTPUT_HPP
#define GRAPH_BENCHMARK_EXPERIMENT_OUTPUT_HPP

#include "../configuration/experiment_config.hpp"
#include "../execution/batch_size_sensitivity_runner.hpp"
#include "../execution/benchmark.hpp"
#include "../execution/execution_plan_runner.hpp"
#include "../execution/harmonic_aggregation_ablation_runner.hpp"
#include "../execution/rff_sensitivity_runner.hpp"
#include "../execution/robustness_runner.hpp"
#include "../execution/scalability_runner.hpp"
#include "csv_writer.hpp"
#include "output_layout.hpp"

#include "../../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/aggregation.hpp"

#include <cstdint>
#include <memory>
#include <set>
#include <string>

namespace vpd {

enum class ExecutionTaskState {
    Absent,
    Complete
};

struct BenchmarkOutputRecord {
    BenchmarkBatch batch;

    std::uint64_t character_seed{};

    HarmonicValues harmonic_values;
};

struct RobustnessOutputRecord {
    RobustnessBatch batch;

    std::uint64_t character_seed{};

    HarmonicValues harmonic_values;
};

class ExperimentOutput {
public:
    ExperimentOutput(
        OutputLayout layout,
        const CanonicalExperimentSpecification& canonical,
        const ExperimentConfig& config
    );

    ExperimentOutput(
        const ExperimentOutput&
    ) = delete;

    ExperimentOutput& operator=(
        const ExperimentOutput&
    ) = delete;

    ExperimentOutput(
        ExperimentOutput&&
    ) = delete;

    ExperimentOutput& operator=(
        ExperimentOutput&&
    ) = delete;

    ~ExperimentOutput();

    const OutputLayout& layout() const;

    ExecutionTaskState existing_task_state(
        const ExecutionTask& task
    ) const;

    void write(
        const BenchmarkOutputRecord& record
    );

    void write(
        const HarmonicAggregationAblationBatch& result
    );

    void write(
        const RobustnessOutputRecord& record
    );

    void write(
        const BatchSizeSensitivityResult& result
    );

    void write(
        const RffSensitivityResult& result
    );

    void write(
        int scalability_condition_index,
        const ScalabilityResult& result
    );

    void flush();

    void finalize();

    bool is_finalized() const;

private:
    OutputLayout layout_;

    BenchmarkSpecification benchmark_;

    bool resume_{};

    bool finalized_{};

    std::set<std::string>
        completed_task_keys_;

    std::set<std::string>
        recovered_payload_keys_;

    std::unique_ptr<CsvWriter>
        benchmark_batches_;

    std::unique_ptr<CsvWriter>
        benchmark_graphs_;

    std::unique_ptr<CsvWriter>
        benchmark_graph_fields_;

    std::unique_ptr<CsvWriter>
        benchmark_vertex_fields_;

    std::unique_ptr<CsvWriter>
        benchmark_edge_fields_;

    std::unique_ptr<CsvWriter>
        benchmark_diagrams_;

    std::unique_ptr<CsvWriter>
        benchmark_harmonic_aggregation_;

    std::unique_ptr<CsvWriter>
        ablation_conditions_;

    std::unique_ptr<CsvWriter>
        ablation_harmonic_aggregation_;

    std::unique_ptr<CsvWriter>
        robustness_conditions_;

    std::unique_ptr<CsvWriter>
        robustness_graphs_;

    std::unique_ptr<CsvWriter>
        robustness_diagrams_;

    std::unique_ptr<CsvWriter>
        robustness_harmonic_aggregation_;

    std::unique_ptr<CsvWriter>
        batch_size_conditions_;

    std::unique_ptr<CsvWriter>
        batch_size_harmonic_aggregation_;

    std::unique_ptr<CsvWriter>
        rff_conditions_;

    std::unique_ptr<CsvWriter>
        rff_harmonic_aggregation_;

    std::unique_ptr<CsvWriter>
        scalability_conditions_;

    std::unique_ptr<CsvWriter>
        scalability_timings_;

    std::unique_ptr<CsvWriter>
        scalability_harmonic_aggregation_;

    void build_resume_index();

    void index_working_files();

    void index_baseline_output();

    void index_ablation_output();

    void index_robustness_output();

    void index_batch_size_output();

    void index_rff_output();

    void index_scalability_output();

    void purge_partial_tasks(
        const std::set<std::string>& partial_task_keys
    );

    void open_writers();

    std::string task_key(
        const ExecutionTask& task
    ) const;

    void record_payload_key(
        const std::string& key
    );

    void record_completed_key(
        const std::string& key
    );
};

}  // namespace vpd

#endif