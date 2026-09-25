#include "tu_dataset.hpp"

#include "../random_graph_classification/output/csv_writer.hpp"

#include "../../core/utils/seed_derivation.hpp"
#include "../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/aggregation.hpp"
#include "../../methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/random_fourier_features.hpp"
#include "../../methods/TDA/persistent_homology/persistence.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vpd {

namespace {

constexpr int kExitSuccess =
    0;

constexpr int kExitFailure =
    1;

constexpr int kExitInvalidArguments =
    2;

constexpr std::uint64_t kRootSeed =
    20260601ULL;

constexpr int kCharacterCount =
    256;

constexpr std::uint64_t kCharacterBankTag =
    0U;

constexpr std::array<
    const char*,
    8U
> kDatasetNames{
    "AIDS",
    "BZR",
    "COX2",
    "MUTAG",
    "Mutagenicity",
    "NCI1",
    "NCI109",
    "PROTEINS"
};

using Triangle =
    std::array<std::size_t, 3U>;

struct EdgeFiltrationKey {
    int trussness{};

    int common_neighbors{};

    int available_neighbors{1};
};

struct FilteredGraphResult {
    Graph graph;

    std::vector<Triangle> triangles;

    std::size_t level_count{};
};

std::pair<int, int> canonical_edge(
    int u,
    int v
) {
    if (u > v) {
        std::swap(
            u,
            v
        );
    }

    return {
        u,
        v
    };
}

std::filesystem::path parse_root(
    int argc,
    char** argv
) {
    if (argc == 1) {
        return
            std::filesystem::absolute(
                "."
            ).lexically_normal();
    }

    if (
        argc == 3 &&
        argv[1] != nullptr &&
        argv[2] != nullptr &&
        std::string(
            argv[1]
        ) ==
        "--root"
    ) {
        return
            std::filesystem::absolute(
                argv[2]
            ).lexically_normal();
    }

    throw std::invalid_argument(
        "Usage: real_world_data.exe [--root <path>]"
    );
}

void prepare_output_root(
    const std::filesystem::path& output_root
) {
    if (
        std::filesystem::exists(
            output_root
        )
    ) {
        if (
            !std::filesystem::is_directory(
                output_root
            )
        ) {
            throw std::runtime_error(
                "The real-world output path is not a "
                "directory: '" +
                output_root.string() +
                "'."
            );
        }

        if (
            !std::filesystem::is_empty(
                output_root
            )
        ) {
            throw std::runtime_error(
                "The real-world output directory is "
                "nonempty: '" +
                output_root.string() +
                "'."
            );
        }
    }

    std::filesystem::create_directories(
        output_root
    );
}

std::vector<std::string>
make_harmonic_header() {
    std::vector<std::string> header{
        "dataset",
        "graph_index",
        "label"
    };

    header.reserve(
        3U +
        2U *
            static_cast<std::size_t>(
                kCharacterCount
            )
    );

    for (
        int character_index = 0;
        character_index <
            kCharacterCount;
        ++character_index
    ) {
        const std::string prefix =
            "character_" +
            std::to_string(
                character_index
            );

        header.push_back(
            prefix +
            "_real"
        );

        header.push_back(
            prefix +
            "_imag"
        );
    }

    return header;
}

std::vector<std::vector<int>>
build_adjacency(
    const Graph& graph
) {
    std::vector<std::vector<int>>
        adjacency(
            static_cast<std::size_t>(
                graph.num_vertices()
            )
        );

    for (const Edge& edge :
         graph.get_edges()) {
        adjacency[
            static_cast<std::size_t>(
                edge.u
            )
        ].push_back(
            edge.v
        );

        adjacency[
            static_cast<std::size_t>(
                edge.v
            )
        ].push_back(
            edge.u
        );
    }

    for (std::vector<int>& neighbors :
         adjacency) {
        std::sort(
            neighbors.begin(),
            neighbors.end()
        );
    }

    return adjacency;
}

std::map<
    std::pair<int, int>,
    std::size_t
> build_edge_index(
    const Graph& graph
) {
    std::map<
        std::pair<int, int>,
        std::size_t
    > edge_index;

    const std::vector<Edge>& edges =
        graph.get_edges();

    for (
        std::size_t index = 0U;
        index < edges.size();
        ++index
    ) {
        edge_index.emplace(
            canonical_edge(
                edges[index].u,
                edges[index].v
            ),
            index
        );
    }

    return edge_index;
}

std::vector<Triangle> enumerate_triangles(
    const Graph& graph,
    const std::vector<
        std::vector<int>
    >& adjacency,
    const std::map<
        std::pair<int, int>,
        std::size_t
    >& edge_index
) {
    std::vector<Triangle> triangles;

    const std::vector<Edge>& edges =
        graph.get_edges();

    for (
        std::size_t edge_index_uv = 0U;
        edge_index_uv < edges.size();
        ++edge_index_uv
    ) {
        const Edge& edge =
            edges[
                edge_index_uv
            ];

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

        const std::vector<int>& neighbors_u =
            adjacency[
                static_cast<std::size_t>(
                    u
                )
            ];

        const std::vector<int>& neighbors_v =
            adjacency[
                static_cast<std::size_t>(
                    v
                )
            ];

        auto first =
            neighbors_u.begin();

        auto second =
            neighbors_v.begin();

        while (
            first !=
                neighbors_u.end() &&
            second !=
                neighbors_v.end()
        ) {
            if (*first < *second) {
                ++first;

                continue;
            }

            if (*second < *first) {
                ++second;

                continue;
            }

            const int w =
                *first;

            if (w > v) {
                triangles.push_back(
                    Triangle{
                        edge_index_uv,
                        edge_index.at(
                            canonical_edge(
                                u,
                                w
                            )
                        ),
                        edge_index.at(
                            canonical_edge(
                                v,
                                w
                            )
                        )
                    }
                );
            }

            ++first;
            ++second;
        }
    }

    return triangles;
}

std::vector<std::vector<std::size_t>>
build_triangle_incidence(
    std::size_t edge_count,
    const std::vector<Triangle>& triangles
) {
    std::vector<
        std::vector<std::size_t>
    > incidence(
        edge_count
    );

    for (
        std::size_t triangle_index = 0U;
        triangle_index <
            triangles.size();
        ++triangle_index
    ) {
        for (
            std::size_t edge_index :
            triangles[
                triangle_index
            ]
        ) {
            incidence[
                edge_index
            ].push_back(
                triangle_index
            );
        }
    }

    return incidence;
}

std::vector<int> triangle_support(
    std::size_t edge_count,
    const std::vector<Triangle>& triangles
) {
    std::vector<int> support(
        edge_count,
        0
    );

    for (const Triangle& triangle :
         triangles) {
        for (
            std::size_t edge_index :
            triangle
        ) {
            ++support[
                edge_index
            ];
        }
    }

    return support;
}

std::vector<int> compute_trussness(
    const std::vector<int>&
        original_support,
    const std::vector<Triangle>&
        triangles,
    const std::vector<
        std::vector<std::size_t>
    >& incidence
) {
    using HeapEntry =
        std::pair<
            int,
            std::size_t
        >;

    std::priority_queue<
        HeapEntry,
        std::vector<HeapEntry>,
        std::greater<HeapEntry>
    > queue;

    std::vector<int> support =
        original_support;

    std::vector<bool> active(
        support.size(),
        true
    );

    std::vector<int> trussness(
        support.size(),
        2
    );

    for (
        std::size_t edge_index = 0U;
        edge_index < support.size();
        ++edge_index
    ) {
        queue.emplace(
            support[
                edge_index
            ],
            edge_index
        );
    }

    while (!queue.empty()) {
        const auto [
            queued_support,
            edge_index
        ] =
            queue.top();

        queue.pop();

        if (
            !active[
                edge_index
            ] ||
            queued_support !=
                support[
                    edge_index
                ]
        ) {
            continue;
        }

        const int level =
            queued_support;

        trussness[
            edge_index
        ] =
            level +
            2;

        active[
            edge_index
        ] =
            false;

        for (
            std::size_t triangle_index :
            incidence[
                edge_index
            ]
        ) {
            const Triangle& triangle =
                triangles[
                    triangle_index
                ];

            std::array<
                std::size_t,
                2U
            > remaining_edges{};

            std::size_t remaining_count =
                0U;

            for (
                std::size_t other_edge :
                triangle
            ) {
                if (
                    other_edge ==
                    edge_index
                ) {
                    continue;
                }

                if (
                    !active[
                        other_edge
                    ]
                ) {
                    remaining_count =
                        0U;

                    break;
                }

                remaining_edges[
                    remaining_count
                ] =
                    other_edge;

                ++remaining_count;
            }

            if (
                remaining_count !=
                2U
            ) {
                continue;
            }

            for (
                std::size_t other_edge :
                remaining_edges
            ) {
                if (
                    support[
                        other_edge
                    ] >
                    level
                ) {
                    --support[
                        other_edge
                    ];

                    queue.emplace(
                        support[
                            other_edge
                        ],
                        other_edge
                    );
                }
            }
        }
    }

    return trussness;
}

EdgeFiltrationKey make_filtration_key(
    const Graph& graph,
    const std::vector<
        std::vector<int>
    >& adjacency,
    const std::vector<int>&
        original_support,
    const std::vector<int>& trussness,
    std::size_t edge_index
) {
    const Edge& edge =
        graph.get_edges()[
            edge_index
        ];

    const std::size_t available_u =
        adjacency[
            static_cast<std::size_t>(
                edge.u
            )
        ].size() -
        1U;

    const std::size_t available_v =
        adjacency[
            static_cast<std::size_t>(
                edge.v
            )
        ].size() -
        1U;

    const int available_neighbors =
        static_cast<int>(
            std::min(
                available_u,
                available_v
            )
        );

    return EdgeFiltrationKey{
        trussness[
            edge_index
        ],
        original_support[
            edge_index
        ],
        available_neighbors == 0
            ? 1
            : available_neighbors
    };
}

int compare_ratio(
    const EdgeFiltrationKey& first,
    const EdgeFiltrationKey& second
) {
    const std::int64_t left =
        static_cast<std::int64_t>(
            first.common_neighbors
        ) *
        static_cast<std::int64_t>(
            second.available_neighbors
        );

    const std::int64_t right =
        static_cast<std::int64_t>(
            second.common_neighbors
        ) *
        static_cast<std::int64_t>(
            first.available_neighbors
        );

    if (left > right) {
        return 1;
    }

    if (left < right) {
        return -1;
    }

    return 0;
}

bool stronger_key(
    const EdgeFiltrationKey& first,
    const EdgeFiltrationKey& second
) {
    if (
        first.trussness !=
        second.trussness
    ) {
        return
            first.trussness >
            second.trussness;
    }

    return
        compare_ratio(
            first,
            second
        ) >
        0;
}

bool equal_key(
    const EdgeFiltrationKey& first,
    const EdgeFiltrationKey& second
) {
    return
        first.trussness ==
            second.trussness &&
        compare_ratio(
            first,
            second
        ) ==
            0;
}

FilteredGraphResult
make_clique_cohesion_filtration(
    const Graph& graph
) {
    const std::vector<Edge>& edges =
        graph.get_edges();

    Graph filtered_graph(
        graph.num_vertices()
    );

    filtered_graph.reserve_edges(
        edges.size()
    );

    if (edges.empty()) {
        return FilteredGraphResult{
            std::move(
                filtered_graph
            ),
            {},
            0U
        };
    }

    const std::vector<
        std::vector<int>
    > adjacency =
        build_adjacency(
            graph
        );

    const std::map<
        std::pair<int, int>,
        std::size_t
    > edge_index =
        build_edge_index(
            graph
        );

    std::vector<Triangle> triangles =
        enumerate_triangles(
            graph,
            adjacency,
            edge_index
        );

    const std::vector<
        std::vector<std::size_t>
    > incidence =
        build_triangle_incidence(
            edges.size(),
            triangles
        );

    const std::vector<int>
        original_support =
            triangle_support(
                edges.size(),
                triangles
            );

    const std::vector<int>
        trussness =
            compute_trussness(
                original_support,
                triangles,
                incidence
            );

    std::vector<EdgeFiltrationKey>
        keys;

    keys.reserve(
        edges.size()
    );

    for (
        std::size_t edge_position = 0U;
        edge_position < edges.size();
        ++edge_position
    ) {
        keys.push_back(
            make_filtration_key(
                graph,
                adjacency,
                original_support,
                trussness,
                edge_position
            )
        );
    }

    std::vector<std::size_t> order(
        edges.size()
    );

    for (
        std::size_t index = 0U;
        index < order.size();
        ++index
    ) {
        order[
            index
        ] =
            index;
    }

    std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t first,
            std::size_t second) {
            const EdgeFiltrationKey&
                first_key =
                    keys[
                        first
                    ];

            const EdgeFiltrationKey&
                second_key =
                    keys[
                        second
                    ];

            if (
                stronger_key(
                    first_key,
                    second_key
                )
            ) {
                return true;
            }

            if (
                stronger_key(
                    second_key,
                    first_key
                )
            ) {
                return false;
            }

            return
                first <
                second;
        }
    );

    std::vector<std::size_t>
        level_by_edge(
            edges.size(),
            0U
        );

    std::size_t level_count =
        0U;

    for (
        std::size_t position = 0U;
        position < order.size();
        ++position
    ) {
        if (
            position == 0U ||
            !equal_key(
                keys[
                    order[
                        position -
                        1U
                    ]
                ],
                keys[
                    order[
                        position
                    ]
                ]
            )
        ) {
            ++level_count;
        }

        level_by_edge[
            order[
                position
            ]
        ] =
            level_count -
            1U;
    }

    for (
        std::size_t edge_position = 0U;
        edge_position < edges.size();
        ++edge_position
    ) {
        const Edge& edge =
            edges[
                edge_position
            ];

        const double time =
            static_cast<double>(
                level_by_edge[
                    edge_position
                ] +
                1U
            ) /
            static_cast<double>(
                level_count +
                1U
            );

        filtered_graph.add_edge(
            edge.u,
            edge.v,
            edge.step,
            time
        );
    }

    return FilteredGraphResult{
        std::move(
            filtered_graph
        ),
        std::move(
            triangles
        ),
        level_count
    };
}

VPD1 compute_persistence_diagram(
    const Graph& graph,
    const std::vector<Triangle>& triangles
) {
    VPD1 diagram =
        compute_H1_persistence(
            graph,
            triangles
        );

    clean_diagram(
        diagram
    );

    return diagram;
}

void write_character_bank(
    CsvWriter& writer,
    const CharacterBank& characters
) {
    for (const CharacterSpec& character :
         characters.values) {
        writer.write_row(
            {
                csv_integer(
                    character.character_index
                ),
                csv_real(
                    character.left_frequencies[0]
                ),
                csv_real(
                    character.left_frequencies[1]
                ),
                csv_real(
                    character.right_frequencies[0]
                ),
                csv_real(
                    character.right_frequencies[1]
                )
            }
        );
    }
}

void write_persistence_diagram(
    CsvWriter& writer,
    const std::string& dataset_name,
    std::size_t graph_index,
    const VPD1& diagram
) {
    std::size_t atom_index =
        0U;

    for (const auto& [
             atom,
             coefficient
         ] : diagram) {
        writer.write_row(
            {
                dataset_name,
                csv_size(
                    graph_index
                ),
                csv_size(
                    atom_index
                ),
                csv_real(
                    atom.birth
                ),
                csv_real(
                    atom.death
                ),
                csv_real(
                    coefficient
                )
            }
        );

        ++atom_index;
    }
}

void write_harmonic_values(
    CsvWriter& writer,
    const std::string& dataset_name,
    const TuGraph& tu_graph,
    const HarmonicValues& values
) {
    CsvRow row;

    row.reserve(
        3U +
        2U *
            values.size()
    );

    row.push_back(
        dataset_name
    );

    row.push_back(
        csv_size(
            tu_graph.graph_index
        )
    );

    row.push_back(
        csv_integer(
            tu_graph.label
        )
    );

    for (const auto& value :
         values) {
        row.push_back(
            csv_real(
                value.real()
            )
        );

        row.push_back(
            csv_real(
                value.imag()
            )
        );
    }

    writer.write_row(
        row
    );
}

void run(
    const std::filesystem::path& root
) {
    const std::filesystem::path data_root =
        root /
        "cpp" /
        "experiments" /
        "real_world_data" /
        "data";

    const std::filesystem::path output_root =
        root /
        "cpp" /
        "results" /
        "raw" /
        "real_world_data";

    if (
        !std::filesystem::exists(
            data_root
        ) ||
        !std::filesystem::is_directory(
            data_root
        )
    ) {
        throw std::runtime_error(
            "The TU dataset root does not exist: '" +
            data_root.string() +
            "'."
        );
    }

    prepare_output_root(
        output_root
    );

    CsvWriter datasets_writer(
        output_root /
            "datasets.csv",
        {
            "dataset",
            "graph_count",
            "vertex_count",
            "edge_count",
            "class_count"
        },
        false
    );

    CsvWriter graphs_writer(
        output_root /
            "graphs.csv",
        {
            "dataset",
            "graph_index",
            "label",
            "vertex_count",
            "edge_count",
            "filtration_level_count",
            "persistence_atom_count"
        },
        false
    );

    CsvWriter persistence_writer(
        output_root /
            "persistence_diagrams.csv",
        {
            "dataset",
            "graph_index",
            "atom_index",
            "birth",
            "death",
            "coefficient"
        },
        false
    );

    CsvWriter harmonic_writer(
        output_root /
            "harmonic_features.csv",
        make_harmonic_header(),
        false
    );

    CsvWriter characters_writer(
        output_root /
            "characters.csv",
        {
            "character_index",
            "left_frequency_birth",
            "left_frequency_death",
            "right_frequency_birth",
            "right_frequency_death"
        },
        false
    );

    const std::uint64_t character_seed =
        derive_seed(
            kRootSeed,
            SeedDomain::CharacterBank,
            {
                kCharacterBankTag
            }
        );

    const CharacterBank characters =
        sample_character_bank(
            kCharacterCount,
            character_seed
        );

    write_character_bank(
        characters_writer,
        characters
    );

    characters_writer.flush();

    for (const char* dataset_name :
         kDatasetNames) {
        TuDataset dataset =
            load_tu_dataset(
                data_root /
                    dataset_name,
                dataset_name
            );

        std::cout
            << "Processing "
            << dataset.name
            << ": "
            << dataset.graphs.size()
            << " graphs\n";

        std::size_t total_vertices =
            0U;

        std::size_t total_edges =
            0U;

        std::set<int> labels;

        for (const TuGraph& tu_graph :
             dataset.graphs) {
            const Graph& graph =
                tu_graph.graph;

            total_vertices +=
                static_cast<std::size_t>(
                    graph.num_vertices()
                );

            total_edges +=
                static_cast<std::size_t>(
                    graph.num_edges()
                );

            labels.insert(
                tu_graph.label
            );

            FilteredGraphResult filtered =
                make_clique_cohesion_filtration(
                    graph
                );

            VPD1 diagram =
                compute_persistence_diagram(
                    filtered.graph,
                    filtered.triangles
                );

            HarmonicValues harmonic_values =
                evaluate_pair_fourier(
                    diagram,
                    characters
                );

            graphs_writer.write_row(
                {
                    dataset.name,
                    csv_size(
                        tu_graph.graph_index
                    ),
                    csv_integer(
                        tu_graph.label
                    ),
                    csv_integer(
                        graph.num_vertices()
                    ),
                    csv_integer(
                        graph.num_edges()
                    ),
                    csv_size(
                        filtered.level_count
                    ),
                    csv_size(
                        diagram.size()
                    )
                }
            );

            write_persistence_diagram(
                persistence_writer,
                dataset.name,
                tu_graph.graph_index,
                diagram
            );

            write_harmonic_values(
                harmonic_writer,
                dataset.name,
                tu_graph,
                harmonic_values
            );
        }

        datasets_writer.write_row(
            {
                dataset.name,
                csv_size(
                    dataset.graphs.size()
                ),
                csv_size(
                    total_vertices
                ),
                csv_size(
                    total_edges
                ),
                csv_size(
                    labels.size()
                )
            }
        );

        datasets_writer.flush();
        graphs_writer.flush();
        persistence_writer.flush();
        harmonic_writer.flush();

        std::cout
            << "Completed "
            << dataset.name
            << "\n";
    }

    characters_writer.finalize();
    datasets_writer.finalize();
    graphs_writer.finalize();
    persistence_writer.finalize();
    harmonic_writer.finalize();

    std::cout
        << "Completed real-world TU cache.\n"
        << "Output root: "
        << output_root.string()
        << "\n";
}

}  // namespace

int run_real_world_data(
    int argc,
    char** argv
) {
    std::filesystem::path root;

    try {
        root =
            parse_root(
                argc,
                argv
            );
    } catch (const std::invalid_argument& error) {
        std::cerr
            << "Argument error: "
            << error.what()
            << "\n";

        return kExitInvalidArguments;
    }

    try {
        run(
            root
        );

        return kExitSuccess;
    } catch (const std::exception& error) {
        std::cerr
            << "Real-world experiment failed: "
            << error.what()
            << "\n";

        return kExitFailure;
    } catch (...) {
        std::cerr
            << "Real-world experiment failed with an "
            << "unknown exception.\n";

        return kExitFailure;
    }
}

}  // namespace vpd

int main(
    int argc,
    char** argv
) {
    return
        vpd::run_real_world_data(
            argc,
            argv
        );
}