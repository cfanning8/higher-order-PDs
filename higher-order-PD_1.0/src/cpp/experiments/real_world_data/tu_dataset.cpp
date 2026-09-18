#include "tu_dataset.hpp"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace vpd {

namespace {

bool is_ascii_whitespace(
    char character
) {
    return
        character == ' ' ||
        character == '\t' ||
        character == '\n' ||
        character == '\r' ||
        character == '\f' ||
        character == '\v';
}

std::string_view trim(
    std::string_view value
) {
    while (
        !value.empty() &&
        is_ascii_whitespace(
            value.front()
        )
    ) {
        value.remove_prefix(
            1U
        );
    }

    while (
        !value.empty() &&
        is_ascii_whitespace(
            value.back()
        )
    ) {
        value.remove_suffix(
            1U
        );
    }

    return value;
}

std::ifstream open_input_file(
    const std::filesystem::path& path
) {
    std::ifstream stream(
        path
    );

    if (!stream.is_open()) {
        throw std::runtime_error(
            "Could not open TU dataset file '" +
            path.string() +
            "'."
        );
    }

    return stream;
}

int parse_integer(
    std::string_view text,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    text =
        trim(
            text
        );

    if (text.empty()) {
        throw std::runtime_error(
            "TU dataset file '" +
            path.string() +
            "' contains an empty integer on line " +
            std::to_string(
                line_number
            ) +
            "."
        );
    }

    int value{};

    const char* const begin =
        text.data();

    const char* const end =
        begin +
        text.size();

    const auto result =
        std::from_chars(
            begin,
            end,
            value
        );

    if (
        result.ec != std::errc{} ||
        result.ptr != end
    ) {
        throw std::runtime_error(
            "TU dataset file '" +
            path.string() +
            "' contains an invalid integer on line " +
            std::to_string(
                line_number
            ) +
            "."
        );
    }

    return value;
}

std::vector<int> read_integer_lines(
    const std::filesystem::path& path
) {
    std::ifstream stream =
        open_input_file(
            path
        );

    std::vector<int> values;

    std::string line;
    std::size_t line_number =
        0U;

    while (
        std::getline(
            stream,
            line
        )
    ) {
        ++line_number;

        if (
            trim(
                line
            ).empty()
        ) {
            throw std::runtime_error(
                "TU dataset file '" +
                path.string() +
                "' contains a blank line on line " +
                std::to_string(
                    line_number
                ) +
                "."
            );
        }

        values.push_back(
            parse_integer(
                line,
                path,
                line_number
            )
        );
    }

    if (stream.bad()) {
        throw std::runtime_error(
            "Could not read TU dataset file '" +
            path.string() +
            "'."
        );
    }

    return values;
}

std::pair<int, int> parse_adjacency_line(
    std::string_view line,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    const std::size_t comma =
        line.find(
            ','
        );

    if (
        comma == std::string_view::npos ||
        line.find(
            ',',
            comma +
                1U
        ) != std::string_view::npos
    ) {
        throw std::runtime_error(
            "TU adjacency file '" +
            path.string() +
            "' contains an invalid adjacency row on line " +
            std::to_string(
                line_number
            ) +
            "."
        );
    }

    const std::string_view left =
        trim(
            line.substr(
                0U,
                comma
            )
        );

    const std::string_view right =
        trim(
            line.substr(
                comma +
                    1U
            )
        );

    if (
        left.empty() ||
        right.empty()
    ) {
        throw std::runtime_error(
            "TU adjacency file '" +
            path.string() +
            "' contains an empty endpoint on line " +
            std::to_string(
                line_number
            ) +
            "."
        );
    }

    return {
        parse_integer(
            left,
            path,
            line_number
        ),
        parse_integer(
            right,
            path,
            line_number
        )
    };
}

void validate_global_endpoint(
    int endpoint,
    std::size_t vertex_count,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    if (
        endpoint <= 0 ||
        static_cast<std::size_t>(
            endpoint
        ) > vertex_count
    ) {
        throw std::runtime_error(
            "TU adjacency file '" +
            path.string() +
            "' references an invalid node ID on line " +
            std::to_string(
                line_number
            ) +
            "."
        );
    }
}

}  // namespace

TuDataset load_tu_dataset(
    const std::filesystem::path& directory,
    const std::string& name
) {
    if (name.empty()) {
        throw std::invalid_argument(
            "A TU dataset name cannot be empty."
        );
    }

    const std::filesystem::path adjacency_path =
        directory /
        (
            name +
            "_A.txt"
        );

    const std::filesystem::path indicator_path =
        directory /
        (
            name +
            "_graph_indicator.txt"
        );

    const std::filesystem::path labels_path =
        directory /
        (
            name +
            "_graph_labels.txt"
        );

    const std::vector<int> indicators =
        read_integer_lines(
            indicator_path
        );

    if (indicators.empty()) {
        throw std::runtime_error(
            "TU dataset '" +
            name +
            "' contains no vertices."
        );
    }

    const std::vector<int> labels =
        read_integer_lines(
            labels_path
        );

    if (labels.empty()) {
        throw std::runtime_error(
            "TU dataset '" +
            name +
            "' contains no graph labels."
        );
    }

    const std::size_t vertex_count =
        indicators.size();

    const std::size_t graph_count =
        labels.size();

    for (
        std::size_t global_index = 0U;
        global_index < vertex_count;
        ++global_index
    ) {
        const int graph_id =
            indicators[
                global_index
            ];

        if (
            graph_id <= 0 ||
            static_cast<std::size_t>(
                graph_id
            ) > graph_count
        ) {
            throw std::runtime_error(
                "TU graph-indicator file '" +
                indicator_path.string() +
                "' references an invalid graph ID on line " +
                std::to_string(
                    global_index +
                        1U
                ) +
                "."
            );
        }
    }

    std::vector<std::size_t> graph_vertex_counts(
        graph_count,
        0U
    );

    for (
        std::size_t global_index = 0U;
        global_index < vertex_count;
        ++global_index
    ) {
        const std::size_t graph_index =
            static_cast<std::size_t>(
                indicators[
                    global_index
                ] -
                1
            );

        ++graph_vertex_counts[
            graph_index
        ];
    }

    for (
        std::size_t graph_index = 0U;
        graph_index < graph_count;
        ++graph_index
    ) {
        if (
            graph_vertex_counts[
                graph_index
            ] >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max()
            )
        ) {
            throw std::runtime_error(
                "TU dataset '" +
                name +
                "' contains too many vertices in graph " +
                std::to_string(
                    graph_index
                ) +
                "."
            );
        }
    }

    std::vector<int> local_vertex_indices(
        vertex_count
    );

    std::vector<int> next_local_index(
        graph_count,
        0
    );

    for (
        std::size_t global_index = 0U;
        global_index < vertex_count;
        ++global_index
    ) {
        const std::size_t graph_index =
            static_cast<std::size_t>(
                indicators[
                    global_index
                ] -
                1
            );

        local_vertex_indices[
            global_index
        ] =
            next_local_index[
                graph_index
            ];

        ++next_local_index[
            graph_index
        ];
    }

    std::vector<TuGraph> graphs;

    graphs.reserve(
        graph_count
    );

    for (
        std::size_t graph_index = 0U;
        graph_index < graph_count;
        ++graph_index
    ) {
        graphs.push_back(
            TuGraph{
                Graph(
                    static_cast<int>(
                        graph_vertex_counts[
                            graph_index
                        ]
                    )
                ),
                labels[
                    graph_index
                ],
                graph_index
            }
        );
    }

    std::ifstream adjacency_stream =
        open_input_file(
            adjacency_path
        );

    std::string line;
    std::size_t line_number =
        0U;

    while (
        std::getline(
            adjacency_stream,
            line
        )
    ) {
        ++line_number;

        if (
            trim(
                line
            ).empty()
        ) {
            throw std::runtime_error(
                "TU adjacency file '" +
                adjacency_path.string() +
                "' contains a blank line on line " +
                std::to_string(
                    line_number
                ) +
                "."
            );
        }

        const auto [u, v] =
            parse_adjacency_line(
                line,
                adjacency_path,
                line_number
            );

        validate_global_endpoint(
            u,
            vertex_count,
            adjacency_path,
            line_number
        );

        validate_global_endpoint(
            v,
            vertex_count,
            adjacency_path,
            line_number
        );

        const std::size_t global_u =
            static_cast<std::size_t>(
                u -
                1
            );

        const std::size_t global_v =
            static_cast<std::size_t>(
                v -
                1
            );

        const std::size_t graph_u =
            static_cast<std::size_t>(
                indicators[
                    global_u
                ] -
                1
            );

        const std::size_t graph_v =
            static_cast<std::size_t>(
                indicators[
                    global_v
                ] -
                1
            );

        if (
            graph_u != graph_v
        ) {
            throw std::runtime_error(
                "TU adjacency file '" +
                adjacency_path.string() +
                "' contains an edge between different graphs on line " +
                std::to_string(
                    line_number
                ) +
                "."
            );
        }

        const int local_u =
            local_vertex_indices[
                global_u
            ];

        const int local_v =
            local_vertex_indices[
                global_v
            ];

        if (
            local_u == local_v
        ) {
            continue;
        }

        graphs[
            graph_u
        ].graph.add_edge(
            local_u,
            local_v
        );
    }

    if (adjacency_stream.bad()) {
        throw std::runtime_error(
            "Could not read TU adjacency file '" +
            adjacency_path.string() +
            "'."
        );
    }

    return TuDataset{
        name,
        std::move(
            graphs
        )
    };
}

}  // namespace vpd