#ifndef GRAPH_BENCHMARK_BATCH_SIZE_SENSITIVITY_HPP
#define GRAPH_BENCHMARK_BATCH_SIZE_SENSITIVITY_HPP

#include "../configuration/experiment_config.hpp"

namespace vpd {

struct BatchSizeSensitivityPlan {
    int maximum_graphs_per_batch{};
};

BatchSizeSensitivityPlan
make_batch_size_sensitivity_plan(
    const BatchSizeSensitivitySpecification&
        specification
);

}  // namespace vpd

#endif