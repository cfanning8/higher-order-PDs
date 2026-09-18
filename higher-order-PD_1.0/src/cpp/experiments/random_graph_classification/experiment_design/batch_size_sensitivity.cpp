#include "batch_size_sensitivity.hpp"

#include <algorithm>

namespace vpd {

BatchSizeSensitivityPlan
make_batch_size_sensitivity_plan(
    const BatchSizeSensitivitySpecification&
        specification
) {
    return BatchSizeSensitivityPlan{
        *std::max_element(
            specification.graphs_per_batch.begin(),
            specification.graphs_per_batch.end()
        )
    };
}

}  // namespace vpd