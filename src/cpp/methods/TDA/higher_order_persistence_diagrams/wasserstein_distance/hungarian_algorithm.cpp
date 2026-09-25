#include "hungarian_algorithm.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace vpd {

double minimum_assignment_cost(
    const std::vector<
        std::vector<double>
    >& cost
) {
    if (cost.empty()) {
        return 0.0;
    }

    const int size =
        static_cast<int>(
            cost.size()
        );

    const std::size_t workspace_size =
        static_cast<std::size_t>(
            size
        ) +
        1U;

    std::vector<double> row_potential(
        workspace_size,
        0.0
    );

    std::vector<double> column_potential(
        workspace_size,
        0.0
    );

    std::vector<int> matched_row(
        workspace_size,
        0
    );

    std::vector<int> predecessor(
        workspace_size,
        0
    );

    for (int row = 1;
         row <= size;
         ++row) {
        matched_row[0] =
            row;

        int current_column =
            0;

        std::vector<double> minimum_reduced_cost(
            workspace_size,
            std::numeric_limits<double>::infinity()
        );

        std::vector<bool> used(
            workspace_size,
            false
        );

        do {
            used[
                static_cast<std::size_t>(
                    current_column
                )
            ] =
                true;

            const int current_row =
                matched_row[
                    static_cast<std::size_t>(
                        current_column
                    )
                ];

            double delta =
                std::numeric_limits<double>::infinity();

            int next_column =
                0;

            for (int column = 1;
                 column <= size;
                 ++column) {
                if (
                    used[
                        static_cast<std::size_t>(
                            column
                        )
                    ]
                ) {
                    continue;
                }

                const double reduced_cost =
                    cost[
                        static_cast<std::size_t>(
                            current_row - 1
                        )
                    ][
                        static_cast<std::size_t>(
                            column - 1
                        )
                    ] -
                    row_potential[
                        static_cast<std::size_t>(
                            current_row
                        )
                    ] -
                    column_potential[
                        static_cast<std::size_t>(
                            column
                        )
                    ];

                if (
                    reduced_cost <
                    minimum_reduced_cost[
                        static_cast<std::size_t>(
                            column
                        )
                    ]
                ) {
                    minimum_reduced_cost[
                        static_cast<std::size_t>(
                            column
                        )
                    ] =
                        reduced_cost;

                    predecessor[
                        static_cast<std::size_t>(
                            column
                        )
                    ] =
                        current_column;
                }

                if (
                    minimum_reduced_cost[
                        static_cast<std::size_t>(
                            column
                        )
                    ] <
                    delta
                ) {
                    delta =
                        minimum_reduced_cost[
                            static_cast<std::size_t>(
                                column
                            )
                        ];

                    next_column =
                        column;
                }
            }

            for (int column = 0;
                 column <= size;
                 ++column) {
                if (
                    used[
                        static_cast<std::size_t>(
                            column
                        )
                    ]
                ) {
                    row_potential[
                        static_cast<std::size_t>(
                            matched_row[
                                static_cast<std::size_t>(
                                    column
                                )
                            ]
                        )
                    ] +=
                        delta;

                    column_potential[
                        static_cast<std::size_t>(
                            column
                        )
                    ] -=
                        delta;
                } else {
                    minimum_reduced_cost[
                        static_cast<std::size_t>(
                            column
                        )
                    ] -=
                        delta;
                }
            }

            current_column =
                next_column;
        } while (
            matched_row[
                static_cast<std::size_t>(
                    current_column
                )
            ] !=
            0
        );

        do {
            const int previous_column =
                predecessor[
                    static_cast<std::size_t>(
                        current_column
                    )
                ];

            matched_row[
                static_cast<std::size_t>(
                    current_column
                )
            ] =
                matched_row[
                    static_cast<std::size_t>(
                        previous_column
                    )
                ];

            current_column =
                previous_column;
        } while (
            current_column !=
            0
        );
    }

    std::vector<int> row_to_column(
        static_cast<std::size_t>(
            size
        ),
        -1
    );

    for (int column = 1;
         column <= size;
         ++column) {
        row_to_column[
            static_cast<std::size_t>(
                matched_row[
                    static_cast<std::size_t>(
                        column
                    )
                ] -
                1
            )
        ] =
            column -
            1;
    }

    double total =
        0.0;

    for (int row = 0;
         row < size;
         ++row) {
        total +=
            cost[
                static_cast<std::size_t>(
                    row
                )
            ][
                static_cast<std::size_t>(
                    row_to_column[
                        static_cast<std::size_t>(
                            row
                        )
                    ]
                )
            ];
    }

    return total;
}

std::vector<std::vector<double>>
make_square_cost_matrix(
    const std::vector<std::vector<double>>&
        off_diagonal_costs,
    const std::vector<double>&
        left_diagonal_costs,
    const std::vector<double>&
        right_diagonal_costs
) {
    const std::size_t left_size =
        left_diagonal_costs.size();

    const std::size_t right_size =
        right_diagonal_costs.size();

    if (
        left_size == 0U &&
        right_size == 0U
    ) {
        return {};
    }

    const std::size_t size =
        left_size +
        right_size;

    double maximum_cost =
        0.0;

    for (const auto& row :
         off_diagonal_costs) {
        for (double value :
             row) {
            maximum_cost =
                std::max(
                    maximum_cost,
                    value
                );
        }
    }

    for (double value :
         left_diagonal_costs) {
        maximum_cost =
            std::max(
                maximum_cost,
                value
            );
    }

    for (double value :
         right_diagonal_costs) {
        maximum_cost =
            std::max(
                maximum_cost,
                value
            );
    }

    const double forbidden_cost =
        (
            static_cast<double>(
                size
            ) +
            1.0
        ) *
        (
            maximum_cost +
            1.0
        );

    std::vector<std::vector<double>> cost(
        size,
        std::vector<double>(
            size,
            0.0
        )
    );

    for (std::size_t left = 0U;
         left <
             left_size;
         ++left) {
        for (std::size_t right = 0U;
             right <
                 right_size;
             ++right) {
            cost[left][right] =
                off_diagonal_costs[
                    left
                ][
                    right
                ];
        }
    }

    for (std::size_t left = 0U;
         left <
             left_size;
         ++left) {
        for (std::size_t diagonal = 0U;
             diagonal <
                 left_size;
             ++diagonal) {
            cost[
                left
            ][
                right_size +
                diagonal
            ] =
                left ==
                        diagonal
                    ? left_diagonal_costs[
                          left
                      ]
                    : forbidden_cost;
        }
    }

    for (std::size_t right = 0U;
         right <
             right_size;
         ++right) {
        for (std::size_t diagonal = 0U;
             diagonal <
                 right_size;
             ++diagonal) {
            cost[
                left_size +
                right
            ][
                diagonal
            ] =
                right ==
                        diagonal
                    ? right_diagonal_costs[
                          right
                      ]
                    : forbidden_cost;
        }
    }

    return cost;
}

}  // namespace vpd