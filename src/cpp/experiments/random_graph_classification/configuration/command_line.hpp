#ifndef GRAPH_BENCHMARK_COMMAND_LINE_HPP
#define GRAPH_BENCHMARK_COMMAND_LINE_HPP

#include "experiment_config.hpp"

namespace vpd {

ExperimentConfig parse_arguments(
    int argc,
    char** argv
);

void print_usage(
    const char* executable
);

}  // namespace vpd

#endif