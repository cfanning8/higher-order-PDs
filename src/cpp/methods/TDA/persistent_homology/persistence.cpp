#include "persistence.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

namespace vpd {

namespace {

std::vector<int> symmetric_difference_sorted(
    const std::vector<int>& first,
    const std::vector<int>& second
) {
    std::vector<int> result;

    result.reserve(
        first.size() +
        second.size()
    );

    std::set_symmetric_difference(
        first.begin(),
        first.end(),
        second.begin(),
        second.end(),
        std::back_inserter(
            result
        )
    );

    return result;
}

std::vector<Simplex>
build_clique_filtration(
    const Graph& graph,
    const std::vector<
        std::array<std::size_t, 3U>
    >& triangles
) {
    std::vector<Simplex> simplices;

    const std::vector<Edge>& edges =
        graph.get_edges();

    simplices.reserve(
        static_cast<std::size_t>(
            graph.num_vertices()
        ) +
        edges.size() +
        triangles.size()
    );

    for (
        int vertex = 0;
        vertex < graph.num_vertices();
        ++vertex
    ) {
        simplices.push_back(
            Simplex{
                {
                    vertex
                },
                0.0,
                0
            }
        );
    }

    for (const Edge& edge :
         edges) {
        simplices.push_back(
            Simplex{
                {
                    std::min(
                        edge.u,
                        edge.v
                    ),
                    std::max(
                        edge.u,
                        edge.v
                    )
                },
                edge.time,
                1
            }
        );
    }

    for (const auto& triangle :
         triangles) {
        const Edge& first_edge =
            edges[
                triangle[0]
            ];

        const Edge& second_edge =
            edges[
                triangle[1]
            ];

        const Edge& third_edge =
            edges[
                triangle[2]
            ];

        std::array<int, 6U> endpoints{
            first_edge.u,
            first_edge.v,
            second_edge.u,
            second_edge.v,
            third_edge.u,
            third_edge.v
        };

        std::sort(
            endpoints.begin(),
            endpoints.end()
        );

        const auto unique_end =
            std::unique(
                endpoints.begin(),
                endpoints.end()
            );

        std::vector<int> vertices(
            endpoints.begin(),
            unique_end
        );

        const double filtration =
            std::max(
                {
                    first_edge.time,
                    second_edge.time,
                    third_edge.time
                }
            );

        simplices.push_back(
            Simplex{
                std::move(
                    vertices
                ),
                filtration,
                2
            }
        );
    }

    std::sort(
        simplices.begin(),
        simplices.end()
    );

    return simplices;
}

VPD1 compute_H1_persistence_from_filtration(
    const std::vector<Simplex>& filtration
) {
    constexpr double terminal_time =
        1.0;

    const std::size_t simplex_count =
        filtration.size();

    std::map<
        std::vector<int>,
        int
    > simplex_index;

    for (
        std::size_t index = 0U;
        index < simplex_count;
        ++index
    ) {
        simplex_index.emplace(
            filtration[
                index
            ].vertices,
            static_cast<int>(
                index
            )
        );
    }

    std::vector<
        std::vector<int>
    > reduced_columns(
        simplex_count
    );

    std::map<int, int> pivot_column;

    std::vector<bool> unpaired_H1_birth(
        simplex_count,
        false
    );

    VPD1 diagram;

    for (
        std::size_t column_index = 0U;
        column_index < simplex_count;
        ++column_index
    ) {
        const Simplex& simplex =
            filtration[
                column_index
            ];

        std::vector<int> column;

        if (
            simplex.dimension >
            0
        ) {
            column.reserve(
                simplex.vertices.size()
            );

            for (
                std::size_t removed = 0U;
                removed <
                    simplex.vertices.size();
                ++removed
            ) {
                std::vector<int> face =
                    simplex.vertices;

                face.erase(
                    face.begin() +
                    static_cast<
                        std::ptrdiff_t
                    >(
                        removed
                    )
                );

                column.push_back(
                    simplex_index.at(
                        face
                    )
                );
            }

            std::sort(
                column.begin(),
                column.end()
            );
        }

        while (!column.empty()) {
            const int low =
                column.back();

            const auto pivot =
                pivot_column.find(
                    low
                );

            if (
                pivot ==
                pivot_column.end()
            ) {
                break;
            }

            column =
                symmetric_difference_sorted(
                    column,
                    reduced_columns[
                        static_cast<
                            std::size_t
                        >(
                            pivot->second
                        )
                    ]
                );
        }

        reduced_columns[
            column_index
        ] =
            column;

        if (column.empty()) {
            if (
                simplex.dimension ==
                1
            ) {
                unpaired_H1_birth[
                    column_index
                ] =
                    true;
            }

            continue;
        }

        const int low =
            column.back();

        pivot_column[
            low
        ] =
            static_cast<int>(
                column_index
            );

        if (
            simplex.dimension ==
                2 &&
            filtration[
                static_cast<
                    std::size_t
                >(
                    low
                )
            ].dimension ==
                1 &&
            unpaired_H1_birth[
                static_cast<
                    std::size_t
                >(
                    low
                )
            ]
        ) {
            diagram[
                Atom1{
                    filtration[
                        static_cast<
                            std::size_t
                        >(
                            low
                        )
                    ].filtration,
                    simplex.filtration
                }
            ] +=
                1.0;

            unpaired_H1_birth[
                static_cast<
                    std::size_t
                >(
                    low
                )
            ] =
                false;
        }
    }

    for (
        std::size_t index = 0U;
        index < simplex_count;
        ++index
    ) {
        if (
            unpaired_H1_birth[
                index
            ]
        ) {
            diagram[
                Atom1{
                    filtration[
                        index
                    ].filtration,
                    terminal_time
                }
            ] +=
                1.0;
        }
    }

    return diagram;
}

}  // namespace

bool Simplex::operator<(
    const Simplex& other
) const {
    if (
        filtration !=
        other.filtration
    ) {
        return
            filtration <
            other.filtration;
    }

    if (
        dimension !=
        other.dimension
    ) {
        return
            dimension <
            other.dimension;
    }

    return
        vertices <
        other.vertices;
}

bool Simplex::operator==(
    const Simplex& other
) const {
    return
        filtration ==
            other.filtration &&
        dimension ==
            other.dimension &&
        vertices ==
            other.vertices;
}

std::vector<Simplex> build_clique_filtration(
    const Graph& graph
) {
    std::vector<Simplex> simplices;

    const std::vector<Edge>& edges =
        graph.get_edges();

    simplices.reserve(
        static_cast<std::size_t>(
            graph.num_vertices()
        ) +
        edges.size()
    );

    for (
        int vertex = 0;
        vertex < graph.num_vertices();
        ++vertex
    ) {
        simplices.push_back(
            Simplex{
                {
                    vertex
                },
                0.0,
                0
            }
        );
    }

    std::map<
        std::pair<int, int>,
        double
    > edge_times;

    for (const Edge& edge :
         edges) {
        const int u =
            std::min(
                edge.u,
                edge.v
            );

        const int v =
            std::max(
                edge.u,
                edge.v
            );

        edge_times.emplace(
            std::make_pair(
                u,
                v
            ),
            edge.time
        );

        simplices.push_back(
            Simplex{
                {
                    u,
                    v
                },
                edge.time,
                1
            }
        );
    }

    for (
        int first = 0;
        first < graph.num_vertices();
        ++first
    ) {
        for (
            int second = first + 1;
            second < graph.num_vertices();
            ++second
        ) {
            const auto first_second =
                edge_times.find(
                    {
                        first,
                        second
                    }
                );

            if (
                first_second ==
                edge_times.end()
            ) {
                continue;
            }

            for (
                int third = second + 1;
                third < graph.num_vertices();
                ++third
            ) {
                const auto first_third =
                    edge_times.find(
                        {
                            first,
                            third
                        }
                    );

                if (
                    first_third ==
                    edge_times.end()
                ) {
                    continue;
                }

                const auto second_third =
                    edge_times.find(
                        {
                            second,
                            third
                        }
                    );

                if (
                    second_third ==
                    edge_times.end()
                ) {
                    continue;
                }

                const double filtration =
                    std::max(
                        {
                            first_second->second,
                            first_third->second,
                            second_third->second
                        }
                    );

                simplices.push_back(
                    Simplex{
                        {
                            first,
                            second,
                            third
                        },
                        filtration,
                        2
                    }
                );
            }
        }
    }

    std::sort(
        simplices.begin(),
        simplices.end()
    );

    return simplices;
}

VPD1 compute_H1_persistence(
    const Graph& graph
) {
    return
        compute_H1_persistence_from_filtration(
            build_clique_filtration(
                graph
            )
        );
}

VPD1 compute_H1_persistence(
    const Graph& graph,
    const std::vector<
        std::array<std::size_t, 3U>
    >& triangles
) {
    return
        compute_H1_persistence_from_filtration(
            build_clique_filtration(
                graph,
                triangles
            )
        );
}

}  // namespace vpd