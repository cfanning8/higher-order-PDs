#include "scalability.hpp"

#include <cstddef>

namespace vpd {

std::size_t
ScalabilityPlan::condition_count() const {
    return conditions.size();
}

ScalabilityPlan make_scalability_plan(
    const ScalabilitySpecification& specification
) {
    ScalabilityPlan plan;

    plan.conditions.reserve(
        static_cast<std::size_t>(
            specification.repeat_count
        ) *
        specification.graphs_per_batch.size() *
        specification.input_support_sizes.size() *
        specification.character_counts.size()
    );

    for (int repeat_index = 0;
         repeat_index <
             specification.repeat_count;
         ++repeat_index) {
        for (int graphs_per_batch :
             specification.graphs_per_batch) {
            for (std::size_t input_support_size :
                 specification.input_support_sizes) {
                for (int character_count :
                     specification.character_counts) {
                    plan.conditions.push_back(
                        ScalabilityCondition{
                            graphs_per_batch,
                            input_support_size,
                            character_count,
                            repeat_index
                        }
                    );
                }
            }
        }
    }

    return plan;
}

}  // namespace vpd