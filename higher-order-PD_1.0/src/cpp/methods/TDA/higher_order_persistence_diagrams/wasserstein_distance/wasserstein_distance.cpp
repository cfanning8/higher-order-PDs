#include "wasserstein_distance.hpp"

#include "hungarian_algorithm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace vpd {

namespace {

constexpr double
    kFlowTolerance = 1.0e-12;

Atom2PairKey symmetric_atom2_key(
    const Atom2& first,
    const Atom2& second
) {
    if (second < first) {
        return Atom2PairKey(
            second,
            first
        );
    }

    return Atom2PairKey(
        first,
        second
    );
}

template <class Atom>
std::vector<Atom> expanded_support(
    const Diagram<Atom>& diagram
) {
    std::size_t total =
        0U;

    for (const auto& [
             atom,
             coefficient
         ] : diagram) {
        (void)atom;

        total +=
            static_cast<std::size_t>(
                coefficient
            );
    }

    std::vector<Atom> atoms;

    atoms.reserve(
        total
    );

    for (const auto& [
             atom,
             coefficient
         ] : diagram) {
        const std::size_t multiplicity =
            static_cast<std::size_t>(
                coefficient
            );

        for (std::size_t copy = 0U;
             copy <
                 multiplicity;
             ++copy) {
            atoms.push_back(
                atom
            );
        }
    }

    return atoms;
}

double assignment_cost_atom1(
    const std::vector<Atom1>& first,
    const std::vector<Atom1>& second
) {
    const std::size_t first_size =
        first.size();

    const std::size_t second_size =
        second.size();

    std::vector<std::vector<double>>
        off_diagonal(
            first_size,
            std::vector<double>(
                second_size,
                0.0
            )
        );

    std::vector<double> first_diagonal(
        first_size,
        0.0
    );

    std::vector<double> second_diagonal(
        second_size,
        0.0
    );

    for (std::size_t index = 0U;
         index <
             first_size;
         ++index) {
        first_diagonal[index] =
            atom1_diagonal_distance(
                first[index]
            );
    }

    for (std::size_t index = 0U;
         index <
             second_size;
         ++index) {
        second_diagonal[index] =
            atom1_diagonal_distance(
                second[index]
            );
    }

    for (std::size_t first_index = 0U;
         first_index <
             first_size;
         ++first_index) {
        for (std::size_t second_index = 0U;
             second_index <
                 second_size;
             ++second_index) {
            off_diagonal[
                first_index
            ][
                second_index
            ] =
                atom1_linf_distance(
                    first[
                        first_index
                    ],
                    second[
                        second_index
                    ]
                );
        }
    }

    return minimum_assignment_cost(
        make_square_cost_matrix(
            off_diagonal,
            first_diagonal,
            second_diagonal
        )
    );
}

double assignment_cost_atom2_direct(
    const std::vector<Atom2>& first,
    const std::vector<Atom2>& second
) {
    const std::size_t first_size =
        first.size();

    const std::size_t second_size =
        second.size();

    std::vector<std::vector<double>>
        off_diagonal(
            first_size,
            std::vector<double>(
                second_size,
                0.0
            )
        );

    std::vector<double> first_diagonal(
        first_size,
        0.0
    );

    std::vector<double> second_diagonal(
        second_size,
        0.0
    );

    for (std::size_t index = 0U;
         index <
             first_size;
         ++index) {
        first_diagonal[index] =
            atom2_diagonal_distance(
                first[index]
            );
    }

    for (std::size_t index = 0U;
         index <
             second_size;
         ++index) {
        second_diagonal[index] =
            atom2_diagonal_distance(
                second[index]
            );
    }

    for (std::size_t first_index = 0U;
         first_index <
             first_size;
         ++first_index) {
        for (std::size_t second_index = 0U;
             second_index <
                 second_size;
             ++second_index) {
            off_diagonal[
                first_index
            ][
                second_index
            ] =
                atom2_basepoint_distance(
                    first[
                        first_index
                    ],
                    second[
                        second_index
                    ]
                );
        }
    }

    return minimum_assignment_cost(
        make_square_cost_matrix(
            off_diagonal,
            first_diagonal,
            second_diagonal
        )
    );
}

double assignment_cost_atom2_memoized(
    const std::vector<Atom2>& first,
    const std::vector<Atom2>& second,
    WassersteinMemo& memo
) {
    const std::size_t first_size =
        first.size();

    const std::size_t second_size =
        second.size();

    std::vector<std::vector<double>>
        off_diagonal(
            first_size,
            std::vector<double>(
                second_size,
                0.0
            )
        );

    std::vector<double> first_diagonal(
        first_size,
        0.0
    );

    std::vector<double> second_diagonal(
        second_size,
        0.0
    );

    for (std::size_t index = 0U;
         index <
             first_size;
         ++index) {
        first_diagonal[index] =
            atom2_diagonal_distance_memoized(
                first[index],
                memo
            );
    }

    for (std::size_t index = 0U;
         index <
             second_size;
         ++index) {
        second_diagonal[index] =
            atom2_diagonal_distance_memoized(
                second[index],
                memo
            );
    }

    for (std::size_t first_index = 0U;
         first_index <
             first_size;
         ++first_index) {
        for (std::size_t second_index = 0U;
             second_index <
                 second_size;
             ++second_index) {
            off_diagonal[
                first_index
            ][
                second_index
            ] =
                atom2_basepoint_distance_memoized(
                    first[
                        first_index
                    ],
                    second[
                        second_index
                    ],
                    memo
                );
        }
    }

    return minimum_assignment_cost(
        make_square_cost_matrix(
            off_diagonal,
            first_diagonal,
            second_diagonal
        )
    );
}

struct TransportNode {
    bool basepoint{};
    Atom2 atom{};

    static TransportNode make_basepoint() {
        TransportNode result;

        result.basepoint =
            true;

        return result;
    }

    static TransportNode make_atom(
        const Atom2& atom
    ) {
        TransportNode result;

        result.atom =
            atom;

        return result;
    }
};

struct FlowEdge {
    int to{};
    int reverse{};
    double capacity{};
    double cost{};
};

class MinCostFlow {
public:
    explicit MinCostFlow(
        int vertex_count
    )
        : graph_(
              static_cast<std::size_t>(
                  vertex_count
              )
          ) {
    }

    void add_edge(
        int from,
        int to,
        double capacity,
        double cost
    ) {
        if (
            capacity <=
            kFlowTolerance
        ) {
            return;
        }

        FlowEdge forward{
            to,
            static_cast<int>(
                graph_[
                    static_cast<std::size_t>(
                        to
                    )
                ].size()
            ),
            capacity,
            cost
        };

        FlowEdge reverse{
            from,
            static_cast<int>(
                graph_[
                    static_cast<std::size_t>(
                        from
                    )
                ].size()
            ),
            0.0,
            -cost
        };

        graph_[
            static_cast<std::size_t>(
                from
            )
        ].push_back(
            forward
        );

        graph_[
            static_cast<std::size_t>(
                to
            )
        ].push_back(
            reverse
        );
    }

    double minimum_cost(
        int source,
        int sink,
        double required_flow
    ) {
        if (
            required_flow <=
            kFlowTolerance
        ) {
            return 0.0;
        }

        const int vertex_count =
            static_cast<int>(
                graph_.size()
            );

        std::vector<double> potential(
            static_cast<std::size_t>(
                vertex_count
            ),
            0.0
        );

        double sent =
            0.0;

        double total_cost =
            0.0;

        while (
            sent +
                kFlowTolerance <
            required_flow
        ) {
            std::vector<double> distance(
                static_cast<std::size_t>(
                    vertex_count
                ),
                std::numeric_limits<double>::infinity()
            );

            std::vector<int> parent_vertex(
                static_cast<std::size_t>(
                    vertex_count
                ),
                -1
            );

            std::vector<int> parent_edge(
                static_cast<std::size_t>(
                    vertex_count
                ),
                -1
            );

            using QueueEntry =
                std::pair<double, int>;

            std::priority_queue<
                QueueEntry,
                std::vector<QueueEntry>,
                std::greater<QueueEntry>
            > queue;

            distance[
                static_cast<std::size_t>(
                    source
                )
            ] =
                0.0;

            queue.emplace(
                0.0,
                source
            );

            while (!queue.empty()) {
                const auto [
                    current_distance,
                    vertex
                ] =
                    queue.top();

                queue.pop();

                if (
                    current_distance !=
                    distance[
                        static_cast<std::size_t>(
                            vertex
                        )
                    ]
                ) {
                    continue;
                }

                const auto& edges =
                    graph_[
                        static_cast<std::size_t>(
                            vertex
                        )
                    ];

                for (std::size_t edge_index = 0U;
                     edge_index <
                         edges.size();
                     ++edge_index) {
                    const FlowEdge& edge =
                        edges[
                            edge_index
                        ];

                    if (
                        edge.capacity <=
                        kFlowTolerance
                    ) {
                        continue;
                    }

                    const double reduced_cost =
                        edge.cost +
                        potential[
                            static_cast<std::size_t>(
                                vertex
                            )
                        ] -
                        potential[
                            static_cast<std::size_t>(
                                edge.to
                            )
                        ];

                    const double candidate =
                        current_distance +
                        reduced_cost;

                    if (
                        candidate <
                        distance[
                            static_cast<std::size_t>(
                                edge.to
                            )
                        ]
                    ) {
                        distance[
                            static_cast<std::size_t>(
                                edge.to
                            )
                        ] =
                            candidate;

                        parent_vertex[
                            static_cast<std::size_t>(
                                edge.to
                            )
                        ] =
                            vertex;

                        parent_edge[
                            static_cast<std::size_t>(
                                edge.to
                            )
                        ] =
                            static_cast<int>(
                                edge_index
                            );

                        queue.emplace(
                            candidate,
                            edge.to
                        );
                    }
                }
            }

            for (int vertex = 0;
                 vertex <
                     vertex_count;
                 ++vertex) {
                if (
                    std::isfinite(
                        distance[
                            static_cast<std::size_t>(
                                vertex
                            )
                        ]
                    )
                ) {
                    potential[
                        static_cast<std::size_t>(
                            vertex
                        )
                    ] +=
                        distance[
                            static_cast<std::size_t>(
                                vertex
                            )
                        ];
                }
            }

            double augmentation =
                required_flow -
                sent;

            for (int vertex = sink;
                 vertex != source;) {
                const int previous =
                    parent_vertex[
                        static_cast<std::size_t>(
                            vertex
                        )
                    ];

                const int edge_index =
                    parent_edge[
                        static_cast<std::size_t>(
                            vertex
                        )
                    ];

                const FlowEdge& edge =
                    graph_[
                        static_cast<std::size_t>(
                            previous
                        )
                    ][
                        static_cast<std::size_t>(
                            edge_index
                        )
                    ];

                augmentation =
                    std::min(
                        augmentation,
                        edge.capacity
                    );

                vertex =
                    previous;
            }

            for (int vertex = sink;
                 vertex != source;) {
                const int previous =
                    parent_vertex[
                        static_cast<std::size_t>(
                            vertex
                        )
                    ];

                const int edge_index =
                    parent_edge[
                        static_cast<std::size_t>(
                            vertex
                        )
                    ];

                FlowEdge& edge =
                    graph_[
                        static_cast<std::size_t>(
                            previous
                        )
                    ][
                        static_cast<std::size_t>(
                            edge_index
                        )
                    ];

                FlowEdge& reverse =
                    graph_[
                        static_cast<std::size_t>(
                            vertex
                        )
                    ][
                        static_cast<std::size_t>(
                            edge.reverse
                        )
                    ];

                edge.capacity -=
                    augmentation;

                reverse.capacity +=
                    augmentation;

                total_cost +=
                    augmentation *
                    edge.cost;

                vertex =
                    previous;
            }

            sent +=
                augmentation;
        }

        return total_cost;
    }

private:
    std::vector<std::vector<FlowEdge>>
        graph_;
};

double direct_transport_cost(
    const TransportNode& source,
    const TransportNode& target
) {
    if (
        source.basepoint &&
        target.basepoint
    ) {
        return 0.0;
    }

    if (source.basepoint) {
        return atom2_diagonal_distance(
            target.atom
        );
    }

    if (target.basepoint) {
        return atom2_diagonal_distance(
            source.atom
        );
    }

    return atom2_basepoint_distance(
        source.atom,
        target.atom
    );
}

double memoized_transport_cost(
    const TransportNode& source,
    const TransportNode& target,
    WassersteinMemo& memo
) {
    if (
        source.basepoint &&
        target.basepoint
    ) {
        return 0.0;
    }

    if (source.basepoint) {
        return atom2_diagonal_distance_memoized(
            target.atom,
            memo
        );
    }

    if (target.basepoint) {
        return atom2_diagonal_distance_memoized(
            source.atom,
            memo
        );
    }

    return atom2_basepoint_distance_memoized(
        source.atom,
        target.atom,
        memo
    );
}

template <class CostFunction>
double signed_transport_to_zero(
    const VPD2& diagram,
    CostFunction&& transport_cost
) {
    std::vector<TransportNode> supplies;
    std::vector<double> supply_mass;

    std::vector<TransportNode> demands;
    std::vector<double> demand_mass;

    double total_coefficient =
        0.0;

    for (const auto& [
             atom,
             coefficient
         ] : diagram) {
        total_coefficient +=
            coefficient;

        if (
            coefficient >
            coefficient_tolerance()
        ) {
            supplies.push_back(
                TransportNode::make_atom(
                    atom
                )
            );

            supply_mass.push_back(
                coefficient
            );
        } else if (
            coefficient <
            -coefficient_tolerance()
        ) {
            demands.push_back(
                TransportNode::make_atom(
                    atom
                )
            );

            demand_mass.push_back(
                -coefficient
            );
        }
    }

    if (
        total_coefficient <
        -coefficient_tolerance()
    ) {
        supplies.push_back(
            TransportNode::make_basepoint()
        );

        supply_mass.push_back(
            -total_coefficient
        );
    } else if (
        total_coefficient >
        coefficient_tolerance()
    ) {
        demands.push_back(
            TransportNode::make_basepoint()
        );

        demand_mass.push_back(
            total_coefficient
        );
    }

    double total_supply =
        0.0;

    for (double mass :
         supply_mass) {
        total_supply +=
            mass;
    }

    if (
        total_supply <=
        coefficient_tolerance()
    ) {
        return 0.0;
    }

    const int source =
        0;

    const int supply_offset =
        1;

    const int demand_offset =
        supply_offset +
        static_cast<int>(
            supplies.size()
        );

    const int sink =
        demand_offset +
        static_cast<int>(
            demands.size()
        );

    MinCostFlow flow(
        sink +
        1
    );

    for (std::size_t index = 0U;
         index <
             supplies.size();
         ++index) {
        flow.add_edge(
            source,
            supply_offset +
                static_cast<int>(
                    index
                ),
            supply_mass[index],
            0.0
        );
    }

    for (std::size_t index = 0U;
         index <
             demands.size();
         ++index) {
        flow.add_edge(
            demand_offset +
                static_cast<int>(
                    index
                ),
            sink,
            demand_mass[index],
            0.0
        );
    }

    for (std::size_t supply = 0U;
         supply <
             supplies.size();
         ++supply) {
        for (std::size_t demand = 0U;
             demand <
                 demands.size();
             ++demand) {
            flow.add_edge(
                supply_offset +
                    static_cast<int>(
                        supply
                    ),
                demand_offset +
                    static_cast<int>(
                        demand
                    ),
                total_supply,
                transport_cost(
                    supplies[
                        supply
                    ],
                    demands[
                        demand
                    ]
                )
            );
        }
    }

    return flow.minimum_cost(
        source,
        sink,
        total_supply
    );
}

}  // namespace

Atom2PairKey::Atom2PairKey(
    const Atom2& first_value,
    const Atom2& second_value
)
    : first(
          first_value
      ),
      second(
          second_value
      ) {
}

bool Atom2PairKey::operator<(
    const Atom2PairKey& other
) const {
    return
        std::tie(
            first,
            second
        ) <
        std::tie(
            other.first,
            other.second
        );
}

double atom1_linf_distance(
    const Atom1& first,
    const Atom1& second
) {
    return std::max(
        std::abs(
            first.birth -
            second.birth
        ),
        std::abs(
            first.death -
            second.death
        )
    );
}

double atom1_diagonal_distance(
    const Atom1& atom
) {
    return
        0.5 *
        (
            atom.death -
            atom.birth
        );
}

double atom1_basepoint_distance(
    const Atom1& first,
    const Atom1& second
) {
    return std::min(
        atom1_linf_distance(
            first,
            second
        ),
        atom1_diagonal_distance(
            first
        ) +
        atom1_diagonal_distance(
            second
        )
    );
}

double atom2_product_distance(
    const Atom2& first,
    const Atom2& second
) {
    return
        atom1_basepoint_distance(
            first.left,
            second.left
        ) +
        atom1_basepoint_distance(
            first.right,
            second.right
        );
}

double atom2_diagonal_distance(
    const Atom2& atom
) {
    return atom1_basepoint_distance(
        atom.left,
        atom.right
    );
}

double atom2_basepoint_distance(
    const Atom2& first,
    const Atom2& second
) {
    return std::min(
        atom2_product_distance(
            first,
            second
        ),
        atom2_diagonal_distance(
            first
        ) +
        atom2_diagonal_distance(
            second
        )
    );
}

double atom2_diagonal_distance_memoized(
    const Atom2& atom,
    WassersteinMemo& memo
) {
    const auto existing =
        memo.atom2_diagonal_cost.find(
            atom
        );

    if (
        existing !=
        memo.atom2_diagonal_cost.end()
    ) {
        return existing->second;
    }

    const double value =
        atom2_diagonal_distance(
            atom
        );

    memo.atom2_diagonal_cost.emplace(
        atom,
        value
    );

    return value;
}

double atom2_basepoint_distance_memoized(
    const Atom2& first,
    const Atom2& second,
    WassersteinMemo& memo
) {
    const Atom2PairKey key =
        symmetric_atom2_key(
            first,
            second
        );

    const auto existing =
        memo.atom2_pair_cost.find(
            key
        );

    if (
        existing !=
        memo.atom2_pair_cost.end()
    ) {
        return existing->second;
    }

    const double diagonal_path =
        atom2_diagonal_distance_memoized(
            first,
            memo
        ) +
        atom2_diagonal_distance_memoized(
            second,
            memo
        );

    const double lower_bound =
        std::abs(
            atom1_diagonal_distance(
                first.left
            ) -
            atom1_diagonal_distance(
                second.left
            )
        ) +
        std::abs(
            atom1_diagonal_distance(
                first.right
            ) -
            atom1_diagonal_distance(
                second.right
            )
        );

    if (
        lower_bound >=
        diagonal_path
    ) {
        memo.atom2_pair_cost.emplace(
            key,
            diagonal_path
        );

        return diagonal_path;
    }

    const double value =
        std::min(
            atom2_product_distance(
                first,
                second
            ),
            diagonal_path
        );

    memo.atom2_pair_cost.emplace(
        key,
        value
    );

    return value;
}

double W1_PD1(
    const PD1& first,
    const PD1& second
) {
    return assignment_cost_atom1(
        expanded_support(
            first
        ),
        expanded_support(
            second
        )
    );
}

double W1_PD2_direct(
    const PD2& first,
    const PD2& second
) {
    return assignment_cost_atom2_direct(
        expanded_support(
            first
        ),
        expanded_support(
            second
        )
    );
}

double W1_PD2_memoized(
    const PD2& first,
    const PD2& second,
    WassersteinMemo& memo
) {
    return assignment_cost_atom2_memoized(
        expanded_support(
            first
        ),
        expanded_support(
            second
        ),
        memo
    );
}

double W1_VPD2_to_zero_direct(
    const VPD2& diagram
) {
    return signed_transport_to_zero(
        diagram,
        [](
            const TransportNode& source,
            const TransportNode& target
        ) {
            return direct_transport_cost(
                source,
                target
            );
        }
    );
}

double W1_VPD2_to_zero_memoized(
    const VPD2& diagram
) {
    WassersteinMemo memo;

    return W1_VPD2_to_zero_memoized(
        diagram,
        memo
    );
}

double W1_VPD2_to_zero_memoized(
    const VPD2& diagram,
    WassersteinMemo& memo
) {
    return signed_transport_to_zero(
        diagram,
        [&memo](
            const TransportNode& source,
            const TransportNode& target
        ) {
            return memoized_transport_cost(
                source,
                target,
                memo
            );
        }
    );
}

}  // namespace vpd