#include "execution_plan_runner.hpp"

#include "../experiment_design/scalability.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace vpd {

namespace {

ExecutionTask model_batch_task(
    ExecutionStudy study,
    int model_index,
    int batch_index
) {
    ExecutionTask task;

    task.study =
        study;

    task.model_index =
        model_index;

    task.batch_index =
        batch_index;

    return task;
}

void append_baseline_tasks(
    std::vector<ExecutionTask>& tasks,
    const BenchmarkSpecification& benchmark
) {
    for (const BenchmarkModel& model :
         benchmark.models) {
        for (int batch_index = 0;
             batch_index <
                 benchmark.batches_per_model;
             ++batch_index) {
            tasks.push_back(
                model_batch_task(
                    ExecutionStudy::Baseline,
                    model.model_index,
                    batch_index
                )
            );
        }
    }
}

void append_ablation_tasks(
    std::vector<ExecutionTask>& tasks,
    const BenchmarkSpecification& benchmark,
    const CanonicalExperimentSpecification& canonical
) {
    for (const BenchmarkModel& model :
         benchmark.models) {
        for (int batch_index = 0;
             batch_index <
                 benchmark.batches_per_model;
             ++batch_index) {
            for (
                int realization_index = 0;
                realization_index <
                    canonical.
                        cross_observation_ablation_realizations;
                ++realization_index
            ) {
                ExecutionTask task =
                    model_batch_task(
                        ExecutionStudy::
                            HarmonicAggregationAblation,
                        model.model_index,
                        batch_index
                    );

                task.ablation_kind =
                    HarmonicAggregationAblationKind::
                        CrossObservation;

                task.realization_index =
                    realization_index;

                tasks.push_back(
                    task
                );
            }

            ExecutionTask linear =
                model_batch_task(
                    ExecutionStudy::
                        HarmonicAggregationAblation,
                    model.model_index,
                    batch_index
                );

            linear.ablation_kind =
                HarmonicAggregationAblationKind::Linear;

            linear.realization_index =
                0;

            tasks.push_back(
                linear
            );

            ExecutionTask no_preorder =
                model_batch_task(
                    ExecutionStudy::
                        HarmonicAggregationAblation,
                    model.model_index,
                    batch_index
                );

            no_preorder.ablation_kind =
                HarmonicAggregationAblationKind::
                    NoPreorder;

            no_preorder.realization_index =
                0;

            tasks.push_back(
                no_preorder
            );
        }
    }
}

void append_robustness_tasks(
    std::vector<ExecutionTask>& tasks,
    const BenchmarkSpecification& benchmark,
    const CanonicalExperimentSpecification& canonical
) {
    for (const BenchmarkModel& model :
         benchmark.models) {
        for (int batch_index = 0;
             batch_index <
                 benchmark.batches_per_model;
             ++batch_index) {
            for (
                std::size_t condition_index = 0U;
                condition_index <
                    canonical.
                        robustness.
                        conditions.size();
                ++condition_index
            ) {
                for (
                    int realization_index = 0;
                    realization_index <
                        canonical.
                            robustness.
                            realization_count;
                    ++realization_index
                ) {
                    ExecutionTask task =
                        model_batch_task(
                            ExecutionStudy::Robustness,
                            model.model_index,
                            batch_index
                        );

                    task.robustness_condition_index =
                        static_cast<int>(
                            condition_index
                        );

                    task.realization_index =
                        realization_index;

                    tasks.push_back(
                        task
                    );
                }
            }
        }
    }
}

void append_batch_size_tasks(
    std::vector<ExecutionTask>& tasks,
    const BenchmarkSpecification& benchmark
) {
    for (const BenchmarkModel& model :
         benchmark.models) {
        for (int batch_index = 0;
             batch_index <
                 benchmark.batches_per_model;
             ++batch_index) {
            tasks.push_back(
                model_batch_task(
                    ExecutionStudy::
                        BatchSizeSensitivity,
                    model.model_index,
                    batch_index
                )
            );
        }
    }
}

void append_rff_tasks(
    std::vector<ExecutionTask>& tasks,
    const BenchmarkSpecification& benchmark,
    const CanonicalExperimentSpecification& canonical
) {
    for (const BenchmarkModel& model :
         benchmark.models) {
        for (int batch_index = 0;
             batch_index <
                 benchmark.batches_per_model;
             ++batch_index) {
            for (
                int replicate_index = 0;
                replicate_index <
                    canonical.
                        random_features.
                        replicate_count;
                ++replicate_index
            ) {
                ExecutionTask task =
                    model_batch_task(
                        ExecutionStudy::RffSensitivity,
                        model.model_index,
                        batch_index
                    );

                task.replicate_index =
                    replicate_index;

                tasks.push_back(
                    task
                );
            }
        }
    }
}

void append_scalability_tasks(
    std::vector<ExecutionTask>& tasks,
    const CanonicalExperimentSpecification& canonical
) {
    const ScalabilityPlan plan =
        make_scalability_plan(
            canonical.scalability
        );

    for (std::size_t condition_index = 0U;
         condition_index <
             plan.conditions.size();
         ++condition_index) {
        ExecutionTask task;

        task.study =
            ExecutionStudy::Scalability;

        task.scalability_condition_index =
            static_cast<int>(
                condition_index
            );

        tasks.push_back(
            task
        );
    }
}

}  // namespace

std::size_t ExecutionPlan::task_count() const {
    return tasks.size();
}

bool ExecutionPlan::empty() const {
    return tasks.empty();
}

const char* execution_study_name(
    ExecutionStudy study
) {
    switch (study) {
        case ExecutionStudy::Baseline:
            return "baseline";

        case ExecutionStudy::
                HarmonicAggregationAblation:
            return
                "harmonic_aggregation_ablation";

        case ExecutionStudy::Robustness:
            return "robustness";

        case ExecutionStudy::
                BatchSizeSensitivity:
            return "batch_size_sensitivity";

        case ExecutionStudy::RffSensitivity:
            return "rff_sensitivity";

        case ExecutionStudy::Scalability:
            return "scalability";
    }

    throw std::logic_error(
        "Unknown execution study."
    );
}

ExecutionPlan build_execution_plan(
    const CanonicalExperimentSpecification& canonical
) {
    ExecutionPlan plan;

    plan.benchmark =
        make_benchmark_specification(
            canonical
        );

    append_baseline_tasks(
        plan.tasks,
        plan.benchmark
    );

    append_ablation_tasks(
        plan.tasks,
        plan.benchmark,
        canonical
    );

    append_robustness_tasks(
        plan.tasks,
        plan.benchmark,
        canonical
    );

    append_batch_size_tasks(
        plan.tasks,
        plan.benchmark
    );

    append_rff_tasks(
        plan.tasks,
        plan.benchmark,
        canonical
    );

    append_scalability_tasks(
        plan.tasks,
        canonical
    );

    return plan;
}

void run_execution_plan(
    const ExecutionPlan& plan,
    const ExecutionCallbacks& callbacks,
    const ExecutionTaskCompletionPredicate&
        is_task_complete
) {
    for (const ExecutionTask& task :
         plan.tasks) {
        if (
            is_task_complete &&
            is_task_complete(
                task
            )
        ) {
            continue;
        }

        switch (task.study) {
            case ExecutionStudy::Baseline:
                callbacks.run_baseline(
                    task
                );

                break;

            case ExecutionStudy::
                    HarmonicAggregationAblation:
                callbacks.
                    run_harmonic_aggregation_ablation(
                        task
                    );

                break;

            case ExecutionStudy::Robustness:
                callbacks.run_robustness(
                    task
                );

                break;

            case ExecutionStudy::
                    BatchSizeSensitivity:
                callbacks.
                    run_batch_size_sensitivity(
                        task
                    );

                break;

            case ExecutionStudy::RffSensitivity:
                callbacks.run_rff_sensitivity(
                    task
                );

                break;

            case ExecutionStudy::Scalability:
                callbacks.run_scalability(
                    task
                );

                break;
        }
    }
}

}  // namespace vpd