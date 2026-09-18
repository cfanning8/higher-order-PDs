#ifndef GRAPH_BENCHMARK_SCALABILITY_HPP
#define GRAPH_BENCHMARK_SCALABILITY_HPP

#include "../configuration/experiment_config.hpp"

#include <cstddef>
#include <vector>

namespace vpd {

struct ScalabilityCondition {
    int graphs_per_batch{};

    std::size_t input_support_size{};

    int character_count{};

    int repeat_index{};
};

struct ScalabilityPlan {
    std::vector<ScalabilityCondition>
        conditions;

    std::size_t condition_count() const;
};

ScalabilityPlan make_scalability_plan(
    const ScalabilitySpecification& specification
);

}  // namespace vpd

#endif