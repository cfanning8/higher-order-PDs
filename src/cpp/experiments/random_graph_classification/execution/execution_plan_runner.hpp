#ifndef GRAPH_BENCHMARK_EXECUTION_PLAN_RUNNER_HPP
#define GRAPH_BENCHMARK_EXECUTION_PLAN_RUNNER_HPP

#include "../configuration/experiment_config.hpp"
#include "../experiment_design/benchmark.hpp"
#include "../experiment_design/harmonic_aggregation_ablation.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace vpd {

enum class ExecutionStudy {
    Baseline,
    HarmonicAggregationAblation,
    Robustness,
    BatchSizeSensitivity,
    RffSensitivity,
    Scalability
};

struct ExecutionTask {
    ExecutionStudy study{
        ExecutionStudy::Baseline
    };

    int model_index{-1};

    int batch_index{-1};

    int robustness_condition_index{-1};

    int realization_index{-1};

    int replicate_index{-1};

    int scalability_condition_index{-1};

    std::optional<
        HarmonicAggregationAblationKind
    > ablation_kind;
};

struct ExecutionPlan {
    BenchmarkSpecification benchmark;

    std::vector<ExecutionTask> tasks;

    std::size_t task_count() const;

    bool empty() const;
};

struct ExecutionCallbacks {
    std::function<void(const ExecutionTask&)>
        run_baseline;

    std::function<void(const ExecutionTask&)>
        run_harmonic_aggregation_ablation;

    std::function<void(const ExecutionTask&)>
        run_robustness;

    std::function<void(const ExecutionTask&)>
        run_batch_size_sensitivity;

    std::function<void(const ExecutionTask&)>
        run_rff_sensitivity;

    std::function<void(const ExecutionTask&)>
        run_scalability;
};

using ExecutionTaskCompletionPredicate =
    std::function<bool(
        const ExecutionTask&
    )>;

ExecutionPlan build_execution_plan(
    const CanonicalExperimentSpecification& canonical
);

void run_execution_plan(
    const ExecutionPlan& plan,
    const ExecutionCallbacks& callbacks,
    const ExecutionTaskCompletionPredicate&
        is_task_complete = {}
);

const char* execution_study_name(
    ExecutionStudy study
);

}  // namespace vpd

#endif