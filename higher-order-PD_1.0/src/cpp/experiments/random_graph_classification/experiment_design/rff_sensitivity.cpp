#include "rff_sensitivity.hpp"

#include <algorithm>

namespace vpd {

RffSensitivityPlan make_rff_sensitivity_plan(
    const RandomFeatureSpecification& specification
) {
    return RffSensitivityPlan{
        *std::max_element(
            specification.character_counts.begin(),
            specification.character_counts.end()
        )
    };
}

}  // namespace vpd