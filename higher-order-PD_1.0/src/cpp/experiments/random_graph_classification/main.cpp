#include "configuration/command_line.hpp"
#include "configuration/experiment_config.hpp"

#include "execution/batch_size_sensitivity_runner.hpp"
#include "execution/benchmark.hpp"
#include "execution/execution_plan_runner.hpp"
#include "execution/harmonic_aggregation_ablation_runner.hpp"
#include "execution/rff_sensitivity_runner.hpp"
#include "execution/robustness_runner.hpp"
#include "execution/scalability_runner.hpp"

#include "output/experiment_output.hpp"
#include "output/metadata_writer.hpp"
#include "output/output_layout.hpp"

#include "../../core/utils/seed_derivation.hpp"
#include "../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/aggregation.hpp"
#include "../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/random_fourier_features.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef GRAPH_BENCHMARK_BUILD_TYPE
#define GRAPH_BENCHMARK_BUILD_TYPE "unknown"
#endif

namespace vpd {

namespace {

constexpr int kExitSuccess =
    0;

constexpr int kExitFailure =
    1;

constexpr int kExitInvalidArguments =
    2;

constexpr std::size_t kFlushEveryTasks =
    10U;

constexpr std::uint64_t
    kBaselineCharacterBankTag =
        0U;

struct TaskProgress {
    std::size_t completed{};

    std::size_t completed_this_run{};

    std::size_t total{};

    void record_completion(
        ExperimentOutput& output,
        const ExecutionTask& task
    ) {
        ++completed;
        ++completed_this_run;

        std::cout
            << "["
            << completed
            << "/"
            << total
            << "] "
            << execution_study_name(
                   task.study
               )
            << "\n";

        if (
            completed_this_run %
                kFlushEveryTasks ==
            0U
        ) {
            output.flush();
        }
    }
};

std::string failure_message(
    const std::exception& error
) {
    const std::string message =
        error.what();

    return
        message.empty()
            ? "The experiment failed."
            : message;
}

std::tm utc_time(
    std::time_t value
) {
    std::tm result{};

#if defined(_WIN32)
    if (
        gmtime_s(
            &result,
            &value
        ) !=
        0
    ) {
        throw std::runtime_error(
            "Could not convert the current time to UTC."
        );
    }
#else
    if (
        gmtime_r(
            &value,
            &result
        ) ==
        nullptr
    ) {
        throw std::runtime_error(
            "Could not convert the current time to UTC."
        );
    }
#endif

    return result;
}

std::string utc_timestamp_now() {
    const auto now =
        std::chrono::system_clock::now();

    const std::time_t time =
        std::chrono::system_clock::to_time_t(
            now
        );

    const std::tm utc =
        utc_time(
            time
        );

    std::ostringstream stream;

    stream
        << std::put_time(
               &utc,
               "%Y-%m-%dT%H:%M:%SZ"
           );

    return stream.str();
}

std::string compiler_name() {
#if defined(__clang__) && defined(_MSC_VER)
    return "Clang-cl";
#elif defined(__clang__)
    return "Clang";
#elif defined(_MSC_VER)
    return "MSVC";
#elif defined(__GNUC__)
    return "GCC";
#else
    return "unknown";
#endif
}

std::string compiler_version() {
    std::ostringstream stream;

#if defined(__clang__)
    stream
        << __clang_major__
        << "."
        << __clang_minor__
        << "."
        << __clang_patchlevel__;
#elif defined(_MSC_FULL_VER)
    stream
        << _MSC_FULL_VER;

#if defined(_MSC_BUILD)
    stream
        << "."
        << _MSC_BUILD;
#endif

#elif defined(_MSC_VER)
    stream
        << _MSC_VER;
#elif defined(__GNUC__)
    stream
        << __GNUC__
        << "."
        << __GNUC_MINOR__
        << "."
        << __GNUC_PATCHLEVEL__;
#else
    return "unknown";
#endif

    return stream.str();
}

std::string operating_system_name() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#elif defined(__FreeBSD__)
    return "FreeBSD";
#else
    return "unknown";
#endif
}

std::string executable_argument(
    int argc,
    char** argv
) {
    if (
        argc > 0 &&
        argv[0] != nullptr
    ) {
        return argv[0];
    }

    return "unknown";
}

BuildMetadata make_build_metadata(
    int argc,
    char** argv
) {
    return BuildMetadata{
        utc_timestamp_now(),
        compiler_name(),
        compiler_version(),
        GRAPH_BENCHMARK_BUILD_TYPE,
        operating_system_name(),
        executable_argument(
            argc,
            argv
        )
    };
}

void prepare_output_root(
    const OutputLayout& layout,
    const ExperimentConfig& config
) {
    const std::filesystem::path& root =
        layout.root();

    if (config.resume) {
        if (
            !std::filesystem::exists(
                root
            )
        ) {
            throw std::runtime_error(
                "Resume requires an existing output "
                "directory: '" +
                root.string() +
                "'."
            );
        }

        if (
            !std::filesystem::is_directory(
                root
            )
        ) {
            throw std::runtime_error(
                "The resume output path is not a "
                "directory: '" +
                root.string() +
                "'."
            );
        }

        return;
    }

    if (
        !std::filesystem::exists(
            root
        )
    ) {
        return;
    }

    if (
        !std::filesystem::is_directory(
            root
        )
    ) {
        throw std::runtime_error(
            "The output path is not a directory: '" +
            root.string() +
            "'."
        );
    }

    if (
        std::filesystem::is_empty(
            root
        )
    ) {
        return;
    }

    if (!config.force) {
        throw std::runtime_error(
            "The output directory is nonempty. "
            "Use --force to replace it: '" +
            root.string() +
            "'."
        );
    }

    std::filesystem::remove_all(
        root
    );
}

template <class Value>
void print_vector(
    const char* label,
    const std::vector<Value>& values
) {
    std::cout
        << label
        << ": ";

    for (std::size_t index = 0U;
         index < values.size();
         ++index) {
        if (index != 0U) {
            std::cout
                << ",";
        }

        std::cout
            << values[index];
    }

    std::cout
        << "\n";
}

void print_selected_models(
    const ExecutionPlan& plan
) {
    std::cout
        << "Models:\n";

    for (const BenchmarkModel& model :
         plan.benchmark.models) {
        std::cout
            << "  "
            << model.model_index
            << ": "
            << model.descriptor.stable_id
            << " ("
            << model.descriptor.display_name
            << ", "
            << model.descriptor.variant
            << ")\n";
    }
}

void print_dry_run_summary(
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionPlan& plan,
    const OutputLayout& layout
) {
    std::cout
        << "Dry run\n"
        << "Output root: "
        << layout.root().string()
        << "\n"
        << "Effective root seed: "
        << experiment_root_seed(
               config
           )
        << "\n"
        << "Model count: "
        << plan.benchmark.model_count()
        << "\n"
        << "Task count: "
        << plan.task_count()
        << "\n"
        << "Vertex count: "
        << canonical.
               benchmark.
               graph_size.
               vertex_count
        << "\n"
        << "Edge count: "
        << canonical.
               benchmark.
               graph_size.
               edge_count
        << "\n"
        << "Batches per model: "
        << canonical.
               benchmark.
               batches_per_model
        << "\n"
        << "Graphs per benchmark batch: "
        << canonical.
               benchmark.
               graphs_per_batch
        << "\n"
        << "Baseline character count: "
        << canonical.
               random_features.
               baseline_character_count
        << "\n"
        << "RFF replicate count: "
        << canonical.
               random_features.
               replicate_count
        << "\n"
        << "Robustness realization count: "
        << canonical.
               robustness.
               realization_count
        << "\n"
        << "Robustness condition count: "
        << canonical.
               robustness.
               conditions.size()
        << "\n"
        << "Cross-observation ablation realizations: "
        << canonical.
               cross_observation_ablation_realizations
        << "\n"
        << "Scalability repeat count: "
        << canonical.
               scalability.
               repeat_count
        << "\n"
        << "Maximum estimated output GiB: "
        << config.maximum_estimated_output_gib
        << "\n"
        << "Estimated diagram atoms per graph: "
        << config.estimated_diagram_atoms_per_graph
        << "\n";

    print_selected_models(
        plan
    );

    print_vector(
        "RFF character counts",
        canonical.
            random_features.
            character_counts
    );

    print_vector(
        "Batch-size sensitivity graph counts",
        canonical.
            batch_size_sensitivity.
            graphs_per_batch
    );

    print_vector(
        "Scalability graph counts",
        canonical.
            scalability.
            graphs_per_batch
    );

    print_vector(
        "Scalability input support sizes",
        canonical.
            scalability.
            input_support_sizes
    );

    print_vector(
        "Scalability character counts",
        canonical.
            scalability.
            character_counts
    );

    std::cout
        << "No output was written.\n";
}

bool task_is_complete(
    const ExperimentOutput& output,
    const ExecutionTask& task
) {
    return
        output.existing_task_state(
            task
        ) ==
        ExecutionTaskState::Complete;
}

std::size_t count_completed_tasks(
    const ExecutionPlan& plan,
    const ExperimentOutput& output
) {
    std::size_t completed =
        0U;

    for (const ExecutionTask& task :
         plan.tasks) {
        if (
            task_is_complete(
                output,
                task
            )
        ) {
            ++completed;
        }
    }

    return completed;
}

void print_resume_summary(
    std::size_t completed,
    std::size_t total
) {
    std::cout
        << "Resume state: "
        << completed
        << " of "
        << total
        << " tasks complete; "
        << (
               total -
               completed
           )
        << " remaining.\n";
}

std::vector<VPD1> benchmark_diagrams(
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

std::vector<VPD1> robustness_diagrams(
    const RobustnessBatch& batch
) {
    std::vector<VPD1> diagrams;

    diagrams.reserve(
        batch.perturbed_graphs.size()
    );

    for (const RobustnessGraphRecord& record :
         batch.perturbed_graphs) {
        diagrams.push_back(
            record.persistence_diagram
        );
    }

    return diagrams;
}

class ExperimentExecutor {
public:
    ExperimentExecutor(
        std::uint64_t root_seed,
        const CanonicalExperimentSpecification& canonical,
        const ExecutionPlan& plan,
        ExperimentOutput& output,
        TaskProgress& progress
    )
        : root_seed_(
              root_seed
          ),
          canonical_(
              canonical
          ),
          plan_(
              plan
          ),
          output_(
              output
          ),
          progress_(
              progress
          ),
          baseline_character_seed_(
              derive_seed(
                  root_seed_,
                  SeedDomain::CharacterBank,
                  {
                      kBaselineCharacterBankTag
                  }
              )
          ),
          baseline_characters_(
              sample_character_bank(
                  canonical_.
                      random_features.
                      baseline_character_count,
                  baseline_character_seed_
              )
          ) {
    }

    ExecutionCallbacks callbacks() {
        return ExecutionCallbacks{
            [this](
                const ExecutionTask& task
            ) {
                run_baseline(
                    task
                );
            },
            [this](
                const ExecutionTask& task
            ) {
                run_ablation(
                    task
                );
            },
            [this](
                const ExecutionTask& task
            ) {
                run_robustness(
                    task
                );
            },
            [this](
                const ExecutionTask& task
            ) {
                run_batch_size_sensitivity(
                    task
                );
            },
            [this](
                const ExecutionTask& task
            ) {
                run_rff_sensitivity(
                    task
                );
            },
            [this](
                const ExecutionTask& task
            ) {
                run_scalability(
                    task
                );
            }
        };
    }

private:
    std::uint64_t root_seed_{};

    const CanonicalExperimentSpecification&
        canonical_;

    const ExecutionPlan& plan_;

    ExperimentOutput& output_;

    TaskProgress& progress_;

    std::uint64_t baseline_character_seed_{};

    CharacterBank baseline_characters_;

    void finish_task(
        const ExecutionTask& task
    ) {
        progress_.record_completion(
            output_,
            task
        );
    }

    void run_baseline(
        const ExecutionTask& task
    ) {
        run_benchmark_task(
            root_seed_,
            plan_.benchmark,
            task,
            [this](
                BenchmarkBatch&& batch
            ) {
                HarmonicValues harmonic_values =
                    mean_pair_fourier(
                        benchmark_diagrams(
                            batch
                        ),
                        baseline_characters_
                    );

                output_.write(
                    BenchmarkOutputRecord{
                        std::move(
                            batch
                        ),
                        baseline_character_seed_,
                        std::move(
                            harmonic_values
                        )
                    }
                );
            }
        );

        finish_task(
            task
        );
    }

    void run_ablation(
        const ExecutionTask& task
    ) {
        run_harmonic_aggregation_ablation_task(
            root_seed_,
            canonical_,
            plan_.benchmark,
            task,
            [this](
                HarmonicAggregationAblationBatch&& result
            ) {
                output_.write(
                    result
                );
            }
        );

        finish_task(
            task
        );
    }

    void run_robustness(
        const ExecutionTask& task
    ) {
        run_robustness_task(
            root_seed_,
            canonical_,
            plan_.benchmark,
            task,
            [this](
                RobustnessBatch&& batch
            ) {
                HarmonicValues harmonic_values =
                    mean_pair_fourier(
                        robustness_diagrams(
                            batch
                        ),
                        baseline_characters_
                    );

                output_.write(
                    RobustnessOutputRecord{
                        std::move(
                            batch
                        ),
                        baseline_character_seed_,
                        std::move(
                            harmonic_values
                        )
                    }
                );
            }
        );

        finish_task(
            task
        );
    }

    void run_batch_size_sensitivity(
        const ExecutionTask& task
    ) {
        run_batch_size_sensitivity_task(
            root_seed_,
            canonical_,
            plan_.benchmark,
            task,
            [this](
                BatchSizeSensitivityResult&& result
            ) {
                output_.write(
                    result
                );
            }
        );

        finish_task(
            task
        );
    }

    void run_rff_sensitivity(
        const ExecutionTask& task
    ) {
        run_rff_sensitivity_task(
            root_seed_,
            canonical_,
            plan_.benchmark,
            task,
            [this](
                RffSensitivityResult&& result
            ) {
                output_.write(
                    result
                );
            }
        );

        finish_task(
            task
        );
    }

    void run_scalability(
        const ExecutionTask& task
    ) {
        run_scalability_task(
            root_seed_,
            canonical_,
            task,
            [this, &task](
                ScalabilityResult&& result
            ) {
                output_.write(
                    task.scalability_condition_index,
                    result
                );
            }
        );

        finish_task(
            task
        );
    }
};

void try_flush_output(
    ExperimentOutput* output
) noexcept {
    if (
        output == nullptr ||
        output->is_finalized()
    ) {
        return;
    }

    try {
        output->flush();
    } catch (const std::exception& error) {
        std::cerr
            << "Could not flush partial output: "
            << error.what()
            << "\n";
    } catch (...) {
        std::cerr
            << "Could not flush partial output.\n";
    }
}

void try_mark_failed(
    const std::optional<OutputLayout>& layout,
    const std::optional<ExperimentConfig>& config,
    const CanonicalExperimentSpecification& canonical,
    const std::optional<ExecutionPlan>& plan,
    const std::optional<BuildMetadata>& build,
    const std::string& message
) noexcept {
    if (
        !layout.has_value() ||
        !config.has_value() ||
        !plan.has_value() ||
        !build.has_value()
    ) {
        return;
    }

    try {
        mark_metadata_failed(
            *layout,
            *config,
            canonical,
            *plan,
            *build,
            message
        );
    } catch (const std::exception& error) {
        std::cerr
            << "Could not mark experiment metadata "
            << "as failed: "
            << error.what()
            << "\n";
    } catch (...) {
        std::cerr
            << "Could not mark experiment metadata "
            << "as failed.\n";
    }
}

int execute_configured_experiment(
    int argc,
    char** argv,
    ExperimentConfig config
) {
    const CanonicalExperimentSpecification&
        canonical =
            canonical_experiment_specification();

    const std::uint64_t root_seed =
        experiment_root_seed(
            config
        );

    std::optional<ExperimentConfig>
        stored_config(
            std::move(
                config
            )
        );

    std::optional<ExecutionPlan>
        plan;

    std::optional<OutputLayout>
        layout;

    std::optional<BuildMetadata>
        build;

    std::unique_ptr<ExperimentOutput>
        output;

    bool metadata_active =
        false;

    try {
        plan =
            build_execution_plan(
                canonical
            );

        layout =
            OutputLayout::from_config(
                *stored_config
            );

        if (stored_config->dry_run) {
            print_dry_run_summary(
                *stored_config,
                canonical,
                *plan,
                *layout
            );

            return kExitSuccess;
        }

        prepare_output_root(
            *layout,
            *stored_config
        );

        build =
            make_build_metadata(
                argc,
                argv
            );

        write_initial_metadata(
            *layout,
            *stored_config,
            canonical,
            *plan,
            *build
        );

        metadata_active =
            true;

        output =
            std::make_unique<ExperimentOutput>(
                *layout,
                canonical,
                *stored_config
            );

        const std::size_t
            completed_before_execution =
                count_completed_tasks(
                    *plan,
                    *output
                );

        if (stored_config->resume) {
            print_resume_summary(
                completed_before_execution,
                plan->task_count()
            );
        }

        TaskProgress progress{
            completed_before_execution,
            0U,
            plan->task_count()
        };

        ExperimentExecutor executor(
            root_seed,
            canonical,
            *plan,
            *output,
            progress
        );

        run_execution_plan(
            *plan,
            executor.callbacks(),
            [output_ptr = output.get()](
                const ExecutionTask& task
            ) {
                return task_is_complete(
                    *output_ptr,
                    task
                );
            }
        );

        output->finalize();

        mark_metadata_complete(
            *layout,
            *stored_config,
            canonical,
            *plan,
            *build
        );

        std::cout
            << "Completed "
            << progress.completed
            << " tasks.\n";

        if (stored_config->resume) {
            std::cout
                << "Executed "
                << progress.completed_this_run
                << " tasks during this resume.\n";
        }

        std::cout
            << "Output root: "
            << layout->root().string()
            << "\n";

        return kExitSuccess;
    } catch (const std::exception& error) {
        const std::string message =
            failure_message(
                error
            );

        try_flush_output(
            output.get()
        );

        if (metadata_active) {
            try_mark_failed(
                layout,
                stored_config,
                canonical,
                plan,
                build,
                message
            );
        }

        std::cerr
            << "Experiment failed: "
            << message
            << "\n";

        return kExitFailure;
    } catch (...) {
        const std::string message =
            "The experiment failed with an unknown "
            "exception.";

        try_flush_output(
            output.get()
        );

        if (metadata_active) {
            try_mark_failed(
                layout,
                stored_config,
                canonical,
                plan,
                build,
                message
            );
        }

        std::cerr
            << "Experiment failed: "
            << message
            << "\n";

        return kExitFailure;
    }
}

}  // namespace

int run_random_graph_classification(
    int argc,
    char** argv
) {
    try {
        return execute_configured_experiment(
            argc,
            argv,
            parse_arguments(
                argc,
                argv
            )
        );
    } catch (const std::invalid_argument& error) {
        std::cerr
            << "Argument error: "
            << failure_message(
                   error
               )
            << "\n";

        return kExitInvalidArguments;
    } catch (const std::exception& error) {
        std::cerr
            << "Command-line processing failed: "
            << failure_message(
                   error
               )
            << "\n";

        return kExitFailure;
    } catch (...) {
        std::cerr
            << "Command-line processing failed with "
            << "an unknown exception.\n";

        return kExitFailure;
    }
}

}  // namespace vpd

int main(
    int argc,
    char** argv
) {
    return vpd::run_random_graph_classification(
        argc,
        argv
    );
}