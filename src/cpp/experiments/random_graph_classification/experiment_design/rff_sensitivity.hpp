#ifndef GRAPH_BENCHMARK_RFF_SENSITIVITY_HPP
#define GRAPH_BENCHMARK_RFF_SENSITIVITY_HPP

#include "../configuration/experiment_config.hpp"

namespace vpd {

struct RffSensitivityPlan {
    int maximum_character_count{};
};

RffSensitivityPlan make_rff_sensitivity_plan(
    const RandomFeatureSpecification& specification
);

}  // namespace vpd

#endif