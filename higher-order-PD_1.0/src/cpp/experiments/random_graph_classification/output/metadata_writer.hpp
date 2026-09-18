#ifndef GRAPH_BENCHMARK_METADATA_WRITER_HPP
#define GRAPH_BENCHMARK_METADATA_WRITER_HPP

#include "../configuration/experiment_config.hpp"
#include "../execution/execution_plan_runner.hpp"
#include "output_layout.hpp"

#include <string>

namespace vpd {

struct BuildMetadata {
    std::string created_utc;

    std::string compiler;

    std::string compiler_version;

    std::string build_type;

    std::string operating_system;

    std::string executable;
};

void write_initial_metadata(
    const OutputLayout& layout,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionPlan& plan,
    const BuildMetadata& build
);

void mark_metadata_complete(
    const OutputLayout& layout,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionPlan& plan,
    const BuildMetadata& build
);

void mark_metadata_failed(
    const OutputLayout& layout,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionPlan& plan,
    const BuildMetadata& build,
    const std::string& failure_message
);

}  // namespace vpd

#endif