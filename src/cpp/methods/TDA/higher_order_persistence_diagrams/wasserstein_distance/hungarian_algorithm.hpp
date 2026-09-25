#ifndef GRAPH_BENCHMARK_HUNGARIAN_ALGORITHM_HPP
#define GRAPH_BENCHMARK_HUNGARIAN_ALGORITHM_HPP

#include <vector>

namespace vpd {

double minimum_assignment_cost(
    const std::vector<
        std::vector<double>
    >& cost
);

std::vector<std::vector<double>>
make_square_cost_matrix(
    const std::vector<std::vector<double>>&
        off_diagonal_costs,
    const std::vector<double>&
        left_diagonal_costs,
    const std::vector<double>&
        right_diagonal_costs
);

}  // namespace vpd

#endif