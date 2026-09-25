#include "experiment_output.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace vpd {

namespace {

constexpr int kOutputSchemaVersion =
    3;

using Row =
    CsvRow;

std::filesystem::path working_csv_path(
    const std::filesystem::path& final_path
) {
    std::filesystem::path path =
        final_path;

    path +=
        ".tmp";

    return path;
}

bool path_exists_checked(
    const std::filesystem::path& path
) {
    std::error_code error;

    const bool exists =
        std::filesystem::exists(
            path,
            error
        );

    if (error) {
        throw std::runtime_error(
            "Could not inspect output path '" +
            path.string() +
            "': " +
            error.message()
        );
    }

    return exists;
}

void rename_checked(
    const std::filesystem::path& source,
    const std::filesystem::path& target
) {
    std::error_code error;

    std::filesystem::rename(
        source,
        target,
        error
    );

    if (error) {
        throw std::runtime_error(
            "Could not restore finalized output '" +
            source.string() +
            "' to resumable working path '" +
            target.string() +
            "': " +
            error.message()
        );
    }
}

void normalize_finalized_csv_for_resume(
    const std::filesystem::path& final_path
) {
    if (
        !path_exists_checked(
            final_path
        )
    ) {
        return;
    }

    rename_checked(
        final_path,
        working_csv_path(
            final_path
        )
    );
}

void require_header(
    const std::vector<std::string>& actual,
    const std::vector<std::string>& expected,
    const std::filesystem::path& path
) {
    if (actual != expected) {
        throw std::runtime_error(
            "CSV schema mismatch in '" +
            path.string() +
            "'."
        );
    }
}

CsvScanResult scan_working_csv(
    const std::filesystem::path& final_path,
    const std::vector<std::string>& expected_header,
    const CsvRowConsumer& consume_row = {}
) {
    const std::filesystem::path working =
        working_csv_path(
            final_path
        );

    if (
        !path_exists_checked(
            working
        )
    ) {
        CsvScanResult result;

        result.header =
            expected_header;

        return result;
    }

    const CsvScanResult initial =
        scan_csv(
            working
        );

    if (
        initial.header.empty()
    ) {
        return CsvScanResult{
            expected_header,
            0U,
            false
        };
    }

    require_header(
        initial.header,
        expected_header,
        working
    );

    if (
        initial.trailing_record_incomplete
    ) {
        rewrite_csv_temporary_filtered(
            final_path,
            expected_header,
            [](const Row&) {
                return true;
            }
        );
    }

    const CsvScanResult result =
        scan_csv(
            working,
            consume_row
        );

    require_header(
        result.header,
        expected_header,
        working
    );

    return result;
}

int parse_int(
    const std::string& value,
    const char* description
) {
    int parsed{};

    const char* const begin =
        value.data();

    const char* const end =
        begin +
        value.size();

    const auto result =
        std::from_chars(
            begin,
            end,
            parsed
        );

    if (
        result.ec !=
            std::errc{} ||
        result.ptr !=
            end
    ) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " is not a valid integer."
        );
    }

    return parsed;
}

std::uint64_t parse_uint64(
    const std::string& value,
    const char* description
) {
    std::uint64_t parsed{};

    const char* const begin =
        value.data();

    const char* const end =
        begin +
        value.size();

    const auto result =
        std::from_chars(
            begin,
            end,
            parsed
        );

    if (
        result.ec !=
            std::errc{} ||
        result.ptr !=
            end
    ) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " is not a valid unsigned integer."
        );
    }

    return parsed;
}

void require_schema_version(
    const Row& row
) {
    if (
        parse_int(
            row[0],
            "CSV schema version"
        ) !=
        kOutputSchemaVersion
    ) {
        throw std::runtime_error(
            "Recovered CSV output uses an incompatible "
            "schema version."
        );
    }
}

HarmonicAggregationAblationKind
parse_harmonic_aggregation_ablation_kind(
    const std::string& value
) {
    if (
        value ==
        "cross_observation"
    ) {
        return
            HarmonicAggregationAblationKind::
                CrossObservation;
    }

    if (
        value ==
        "linear"
    ) {
        return
            HarmonicAggregationAblationKind::
                Linear;
    }

    if (
        value ==
        "no_preorder"
    ) {
        return
            HarmonicAggregationAblationKind::
                NoPreorder;
    }

    throw std::runtime_error(
        "Recovered harmonic-aggregation ablation kind "
        "is invalid."
    );
}

const char* robustness_perturbation_name(
    RobustnessPerturbation perturbation
) {
    switch (perturbation) {
        case RobustnessPerturbation::EdgeDeletion:
            return
                "edge_deletion";

        case RobustnessPerturbation::EdgeInsertion:
            return
                "edge_insertion";

        case RobustnessPerturbation::
                DegreePreservingRewiring:
            return
                "degree_preserving_rewiring";

        case RobustnessPerturbation::FiltrationNoise:
            return
                "filtration_noise";
    }

    throw std::logic_error(
        "Unknown robustness perturbation."
    );
}

std::string baseline_key(
    int model_index,
    int parameter_condition_index,
    int batch_index
) {
    return
        "baseline|model=" +
        std::to_string(
            model_index
        ) +
        "|parameter=" +
        std::to_string(
            parameter_condition_index
        ) +
        "|batch=" +
        std::to_string(
            batch_index
        );
}

std::string ablation_key(
    int model_index,
    int parameter_condition_index,
    int batch_index,
    HarmonicAggregationAblationKind kind,
    int realization_index
) {
    return
        "ablation|model=" +
        std::to_string(
            model_index
        ) +
        "|parameter=" +
        std::to_string(
            parameter_condition_index
        ) +
        "|batch=" +
        std::to_string(
            batch_index
        ) +
        "|kind=" +
        harmonic_aggregation_ablation_name(
            kind
        ) +
        "|realization=" +
        std::to_string(
            realization_index
        );
}

std::string robustness_key(
    int model_index,
    int parameter_condition_index,
    int batch_index,
    int condition_index,
    int realization_index
) {
    return
        "robustness|model=" +
        std::to_string(
            model_index
        ) +
        "|parameter=" +
        std::to_string(
            parameter_condition_index
        ) +
        "|batch=" +
        std::to_string(
            batch_index
        ) +
        "|condition=" +
        std::to_string(
            condition_index
        ) +
        "|realization=" +
        std::to_string(
            realization_index
        );
}

std::string batch_size_key(
    int model_index,
    int parameter_condition_index,
    int batch_index
) {
    return
        "batch_size|model=" +
        std::to_string(
            model_index
        ) +
        "|parameter=" +
        std::to_string(
            parameter_condition_index
        ) +
        "|batch=" +
        std::to_string(
            batch_index
        );
}

std::string rff_key(
    int model_index,
    int parameter_condition_index,
    int batch_index,
    int replicate_index
) {
    return
        "rff|model=" +
        std::to_string(
            model_index
        ) +
        "|parameter=" +
        std::to_string(
            parameter_condition_index
        ) +
        "|batch=" +
        std::to_string(
            batch_index
        ) +
        "|replicate=" +
        std::to_string(
            replicate_index
        );
}

std::string scalability_key(
    int condition_index
) {
    return
        "scalability|condition=" +
        std::to_string(
            condition_index
        );
}

std::vector<std::string>
benchmark_batch_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "model_name",
        "model_variant",
        "parameter_condition_index",
        "batch_index",
        "graphs_per_batch",
        "latent_seed",
        "character_seed"
    };
}

std::vector<std::string>
benchmark_graph_header() {
    return {
        "schema_version",
        "record_type",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "latent_seed",
        "sampling_seed",
        "vertex_count",
        "edge_count",
        "u",
        "v",
        "filtration_step",
        "filtration_time"
    };
}

std::vector<std::string>
benchmark_graph_field_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "field_name",
        "value"
    };
}

std::vector<std::string>
benchmark_vertex_field_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "vertex_index",
        "field_name",
        "value"
    };
}

std::vector<std::string>
benchmark_edge_field_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "edge_index",
        "u",
        "v",
        "field_name",
        "value"
    };
}

std::vector<std::string>
benchmark_diagram_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "birth",
        "death",
        "coefficient"
    };
}

std::vector<std::string>
benchmark_harmonic_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "character_seed",
        "character_index",
        "real",
        "imag"
    };
}

std::vector<std::string>
ablation_condition_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "ablation_kind",
        "realization_index",
        "graphs_per_batch",
        "aggregation_order",
        "character_count",
        "character_seed",
        "ablation_seed"
    };
}

std::vector<std::string>
ablation_harmonic_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "ablation_kind",
        "realization_index",
        "character_index",
        "real",
        "imag"
    };
}

std::vector<std::string>
robustness_condition_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "condition_index",
        "perturbation",
        "severity",
        "realization_index",
        "graphs_per_batch",
        "character_seed"
    };
}

std::vector<std::string>
robustness_graph_header() {
    return {
        "schema_version",
        "record_type",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "condition_index",
        "perturbation",
        "severity",
        "realization_index",
        "perturbation_seed",
        "vertex_count",
        "edge_count",
        "u",
        "v",
        "filtration_step",
        "filtration_time"
    };
}

std::vector<std::string>
robustness_diagram_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graph_index",
        "condition_index",
        "perturbation",
        "severity",
        "realization_index",
        "birth",
        "death",
        "coefficient"
    };
}

std::vector<std::string>
robustness_harmonic_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "condition_index",
        "realization_index",
        "character_seed",
        "character_index",
        "real",
        "imag"
    };
}

std::vector<std::string>
batch_size_condition_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "maximum_graphs_per_batch",
        "character_seed",
        "condition_count"
    };
}

std::vector<std::string>
batch_size_harmonic_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "graphs_per_batch",
        "character_index",
        "real",
        "imag"
    };
}

std::vector<std::string>
rff_condition_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "replicate_index",
        "character_seed",
        "condition_count"
    };
}

std::vector<std::string>
rff_harmonic_header() {
    return {
        "schema_version",
        "model_index",
        "model_id",
        "parameter_condition_index",
        "batch_index",
        "replicate_index",
        "character_count",
        "character_index",
        "real",
        "imag"
    };
}

std::vector<std::string>
scalability_condition_header() {
    return {
        "schema_version",
        "scalability_condition_index",
        "repeat_index",
        "graphs_per_batch",
        "input_support_size",
        "character_count",
        "sample_seed",
        "character_seed"
    };
}

std::vector<std::string>
scalability_timing_header() {
    return {
        "schema_version",
        "scalability_condition_index",
        "representation_seconds"
    };
}

std::vector<std::string>
scalability_harmonic_header() {
    return {
        "schema_version",
        "scalability_condition_index",
        "character_index",
        "real",
        "imag"
    };
}

void write_harmonic_rows(
    CsvWriter& writer,
    const std::vector<std::string>& identity_fields,
    const HarmonicValues& values
) {
    for (std::size_t character_index = 0U;
         character_index <
             values.size();
         ++character_index) {
        Row row =
            identity_fields;

        row.push_back(
            csv_size(
                character_index
            )
        );

        row.push_back(
            csv_real(
                values[
                    character_index
                ].real()
            )
        );

        row.push_back(
            csv_real(
                values[
                    character_index
                ].imag()
            )
        );

        writer.write_row(
            row
        );
    }
}

void write_benchmark_graph_rows(
    CsvWriter& writer,
    const BenchmarkBatch& batch
) {
    for (const BenchmarkGraphRecord& record :
         batch.graphs) {
        const Graph& graph =
            record.network.graph;

        const std::vector<Edge>& edges =
            graph.get_edges();

        writer.write_row(
            {
                csv_integer(
                    kOutputSchemaVersion
                ),
                "graph",
                csv_integer(
                    batch.model_index
                ),
                batch.descriptor.stable_id,
                csv_integer(
                    batch.parameter_condition_index
                ),
                csv_integer(
                    batch.batch_index
                ),
                csv_integer(
                    record.key.graph_index
                ),
                csv_unsigned_integer(
                    record.latent_seed
                ),
                csv_unsigned_integer(
                    record.sampling_seed
                ),
                csv_integer(
                    graph.num_vertices()
                ),
                csv_size(
                    edges.size()
                ),
                "",
                "",
                "",
                ""
            }
        );

        for (const Edge& edge :
             edges) {
            writer.write_row(
                {
                    csv_integer(
                        kOutputSchemaVersion
                    ),
                    "edge",
                    csv_integer(
                        batch.model_index
                    ),
                    batch.descriptor.stable_id,
                    csv_integer(
                        batch.parameter_condition_index
                    ),
                    csv_integer(
                        batch.batch_index
                    ),
                    csv_integer(
                        record.key.graph_index
                    ),
                    csv_unsigned_integer(
                        record.latent_seed
                    ),
                    csv_unsigned_integer(
                        record.sampling_seed
                    ),
                    csv_integer(
                        graph.num_vertices()
                    ),
                    csv_size(
                        edges.size()
                    ),
                    csv_integer(
                        std::min(
                            edge.u,
                            edge.v
                        )
                    ),
                    csv_integer(
                        std::max(
                            edge.u,
                            edge.v
                        )
                    ),
                    csv_integer(
                        edge.step
                    ),
                    csv_real(
                        edge.time
                    )
                }
            );
        }
    }
}

void write_benchmark_graph_field_rows(
    CsvWriter& writer,
    const BenchmarkBatch& batch
) {
    for (const BenchmarkGraphRecord& record :
         batch.graphs) {
        for (const ScalarField& field :
             record.network.graph_fields) {
            writer.write_row(
                {
                    csv_integer(
                        kOutputSchemaVersion
                    ),
                    csv_integer(
                        batch.model_index
                    ),
                    batch.descriptor.stable_id,
                    csv_integer(
                        batch.parameter_condition_index
                    ),
                    csv_integer(
                        batch.batch_index
                    ),
                    csv_integer(
                        record.key.graph_index
                    ),
                    field.name,
                    csv_real(
                        field.value
                    )
                }
            );
        }
    }
}

void write_benchmark_vertex_field_rows(
    CsvWriter& writer,
    const BenchmarkBatch& batch
) {
    for (const BenchmarkGraphRecord& record :
         batch.graphs) {
        for (const VertexField& field :
             record.network.vertex_fields) {
            for (std::size_t vertex_index = 0U;
                 vertex_index <
                     field.values.size();
                 ++vertex_index) {
                writer.write_row(
                    {
                        csv_integer(
                            kOutputSchemaVersion
                        ),
                        csv_integer(
                            batch.model_index
                        ),
                        batch.descriptor.stable_id,
                        csv_integer(
                            batch.parameter_condition_index
                        ),
                        csv_integer(
                            batch.batch_index
                        ),
                        csv_integer(
                            record.key.graph_index
                        ),
                        csv_size(
                            vertex_index
                        ),
                        field.name,
                        csv_real(
                            field.values[
                                vertex_index
                            ]
                        )
                    }
                );
            }
        }
    }
}

void write_benchmark_edge_field_rows(
    CsvWriter& writer,
    const BenchmarkBatch& batch
) {
    for (const BenchmarkGraphRecord& record :
         batch.graphs) {
        const std::vector<Edge>& edges =
            record.network.graph.get_edges();

        for (const EdgeField& field :
             record.network.edge_fields) {
            for (std::size_t edge_index = 0U;
                 edge_index <
                     field.values.size();
                 ++edge_index) {
                const Edge& edge =
                    edges[
                        edge_index
                    ];

                writer.write_row(
                    {
                        csv_integer(
                            kOutputSchemaVersion
                        ),
                        csv_integer(
                            batch.model_index
                        ),
                        batch.descriptor.stable_id,
                        csv_integer(
                            batch.parameter_condition_index
                        ),
                        csv_integer(
                            batch.batch_index
                        ),
                        csv_integer(
                            record.key.graph_index
                        ),
                        csv_size(
                            edge_index
                        ),
                        csv_integer(
                            std::min(
                                edge.u,
                                edge.v
                            )
                        ),
                        csv_integer(
                            std::max(
                                edge.u,
                                edge.v
                            )
                        ),
                        field.name,
                        csv_real(
                            field.values[
                                edge_index
                            ]
                        )
                    }
                );
            }
        }
    }
}

void write_benchmark_diagram_rows(
    CsvWriter& writer,
    const BenchmarkBatch& batch
) {
    for (const BenchmarkGraphRecord& record :
         batch.graphs) {
        for (const auto& [
                 atom,
                 coefficient
             ] :
             record.persistence_diagram) {
            writer.write_row(
                {
                    csv_integer(
                        kOutputSchemaVersion
                    ),
                    csv_integer(
                        batch.model_index
                    ),
                    batch.descriptor.stable_id,
                    csv_integer(
                        batch.parameter_condition_index
                    ),
                    csv_integer(
                        batch.batch_index
                    ),
                    csv_integer(
                        record.key.graph_index
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
        }
    }
}

void write_robustness_graph_rows(
    CsvWriter& writer,
    const RobustnessBatch& batch
) {
    const BenchmarkBatch& clean =
        batch.clean_batch;

    for (const RobustnessGraphRecord& record :
         batch.perturbed_graphs) {
        const Graph& graph =
            record.perturbed_graph;

        const std::vector<Edge>& edges =
            graph.get_edges();

        writer.write_row(
            {
                csv_integer(
                    kOutputSchemaVersion
                ),
                "graph",
                csv_integer(
                    clean.model_index
                ),
                clean.descriptor.stable_id,
                csv_integer(
                    clean.parameter_condition_index
                ),
                csv_integer(
                    clean.batch_index
                ),
                csv_integer(
                    record.key.graph_index
                ),
                csv_integer(
                    batch.condition_index
                ),
                robustness_perturbation_name(
                    batch.condition.perturbation
                ),
                csv_real(
                    batch.condition.severity
                ),
                csv_integer(
                    batch.realization_index
                ),
                csv_unsigned_integer(
                    record.perturbation_seed
                ),
                csv_integer(
                    graph.num_vertices()
                ),
                csv_size(
                    edges.size()
                ),
                "",
                "",
                "",
                ""
            }
        );

        for (const Edge& edge :
             edges) {
            writer.write_row(
                {
                    csv_integer(
                        kOutputSchemaVersion
                    ),
                    "edge",
                    csv_integer(
                        clean.model_index
                    ),
                    clean.descriptor.stable_id,
                    csv_integer(
                        clean.parameter_condition_index
                    ),
                    csv_integer(
                        clean.batch_index
                    ),
                    csv_integer(
                        record.key.graph_index
                    ),
                    csv_integer(
                        batch.condition_index
                    ),
                    robustness_perturbation_name(
                        batch.condition.perturbation
                    ),
                    csv_real(
                        batch.condition.severity
                    ),
                    csv_integer(
                        batch.realization_index
                    ),
                    csv_unsigned_integer(
                        record.perturbation_seed
                    ),
                    csv_integer(
                        graph.num_vertices()
                    ),
                    csv_size(
                        edges.size()
                    ),
                    csv_integer(
                        std::min(
                            edge.u,
                            edge.v
                        )
                    ),
                    csv_integer(
                        std::max(
                            edge.u,
                            edge.v
                        )
                    ),
                    csv_integer(
                        edge.step
                    ),
                    csv_real(
                        edge.time
                    )
                }
            );
        }
    }
}

void write_robustness_diagram_rows(
    CsvWriter& writer,
    const RobustnessBatch& batch
) {
    const BenchmarkBatch& clean =
        batch.clean_batch;

    for (const RobustnessGraphRecord& record :
         batch.perturbed_graphs) {
        for (const auto& [
                 atom,
                 coefficient
             ] :
             record.persistence_diagram) {
            writer.write_row(
                {
                    csv_integer(
                        kOutputSchemaVersion
                    ),
                    csv_integer(
                        clean.model_index
                    ),
                    clean.descriptor.stable_id,
                    csv_integer(
                        clean.parameter_condition_index
                    ),
                    csv_integer(
                        clean.batch_index
                    ),
                    csv_integer(
                        record.key.graph_index
                    ),
                    csv_integer(
                        batch.condition_index
                    ),
                    robustness_perturbation_name(
                        batch.condition.perturbation
                    ),
                    csv_real(
                        batch.condition.severity
                    ),
                    csv_integer(
                        batch.realization_index
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
        }
    }
}

template <class KeyFunction>
void rewrite_without_keys(
    const std::filesystem::path& final_path,
    const std::vector<std::string>& header,
    const std::set<std::string>& removed_keys,
    const KeyFunction& key_function
) {
    if (
        removed_keys.empty()
    ) {
        return;
    }

    rewrite_csv_temporary_filtered(
        final_path,
        header,
        [&](const Row& row) {
            return
                removed_keys.find(
                    key_function(
                        row
                    )
                ) ==
                removed_keys.end();
        }
    );
}

}  // namespace

ExperimentOutput::ExperimentOutput(
    OutputLayout layout,
    const CanonicalExperimentSpecification& canonical,
    const ExperimentConfig& config
)
    : layout_(
          std::move(
              layout
          )
      ),
      benchmark_(
          make_benchmark_specification(
              canonical
          )
      ),
      resume_(
          config.resume
      ) {
    layout_.create_directories();

    if (resume_) {
        normalize_finalized_csv_for_resume(
            layout_.benchmark_batches_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.benchmark_graphs_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.benchmark_graph_fields_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.benchmark_vertex_fields_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.benchmark_edge_fields_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                benchmark_persistence_diagrams_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                benchmark_harmonic_aggregation_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                harmonic_aggregation_ablation_conditions_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                harmonic_aggregation_ablation_features_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.robustness_conditions_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.robustness_graphs_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                robustness_persistence_diagrams_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                robustness_harmonic_aggregation_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                batch_size_sensitivity_conditions_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                batch_size_sensitivity_harmonic_aggregation_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                rff_sensitivity_conditions_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                rff_sensitivity_harmonic_aggregation_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.scalability_conditions_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.scalability_timings_csv()
        );

        normalize_finalized_csv_for_resume(
            layout_.
                scalability_harmonic_aggregation_csv()
        );
    }

    build_resume_index();

    open_writers();
}

ExperimentOutput::~ExperimentOutput() =
    default;

const OutputLayout&
ExperimentOutput::layout() const {
    return layout_;
}

std::string ExperimentOutput::task_key(
    const ExecutionTask& task
) const {
    if (
        task.study ==
        ExecutionStudy::Scalability
    ) {
        return scalability_key(
            task.scalability_condition_index
        );
    }

    const BenchmarkModel& model =
        benchmark_model_for_index(
            benchmark_,
            task.model_index
        );

    const int parameter_condition =
        static_cast<int>(
            parameter_condition_index(
                benchmark_,
                model,
                task.batch_index
            )
        );

    switch (task.study) {
        case ExecutionStudy::Baseline:
            return baseline_key(
                task.model_index,
                parameter_condition,
                task.batch_index
            );

        case ExecutionStudy::
                HarmonicAggregationAblation:
            return ablation_key(
                task.model_index,
                parameter_condition,
                task.batch_index,
                *task.ablation_kind,
                task.realization_index
            );

        case ExecutionStudy::Robustness:
            return robustness_key(
                task.model_index,
                parameter_condition,
                task.batch_index,
                task.robustness_condition_index,
                task.realization_index
            );

        case ExecutionStudy::
                BatchSizeSensitivity:
            return batch_size_key(
                task.model_index,
                parameter_condition,
                task.batch_index
            );

        case ExecutionStudy::RffSensitivity:
            return rff_key(
                task.model_index,
                parameter_condition,
                task.batch_index,
                task.replicate_index
            );

        case ExecutionStudy::Scalability:
            break;
    }

    throw std::logic_error(
        "Unknown execution study."
    );
}

ExecutionTaskState
ExperimentOutput::existing_task_state(
    const ExecutionTask& task
) const {
    return
        completed_task_keys_.find(
            task_key(
                task
            )
        ) !=
            completed_task_keys_.end()
        ? ExecutionTaskState::Complete
        : ExecutionTaskState::Absent;
}

void ExperimentOutput::record_payload_key(
    const std::string& key
) {
    recovered_payload_keys_.insert(
        key
    );
}

void ExperimentOutput::record_completed_key(
    const std::string& key
) {
    if (
        !completed_task_keys_.insert(
            key
        ).second
    ) {
        throw std::runtime_error(
            "Recovered output contains a duplicate "
            "task commit marker."
        );
    }
}

void ExperimentOutput::build_resume_index() {
    completed_task_keys_.clear();
    recovered_payload_keys_.clear();

    if (!resume_) {
        return;
    }

    index_working_files();

    std::set<std::string>
        partial_task_keys;

    std::set_difference(
        recovered_payload_keys_.begin(),
        recovered_payload_keys_.end(),
        completed_task_keys_.begin(),
        completed_task_keys_.end(),
        std::inserter(
            partial_task_keys,
            partial_task_keys.end()
        )
    );

    purge_partial_tasks(
        partial_task_keys
    );

    recovered_payload_keys_.clear();
}

void ExperimentOutput::index_working_files() {
    index_baseline_output();

    index_ablation_output();

    index_robustness_output();

    index_batch_size_output();

    index_rff_output();

    index_scalability_output();
}

void ExperimentOutput::index_baseline_output() {
    scan_working_csv(
        layout_.benchmark_batches_csv(),
        benchmark_batch_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_completed_key(
                baseline_key(
                    parse_int(
                        row[1],
                        "Baseline model index"
                    ),
                    parse_int(
                        row[5],
                        "Baseline parameter condition"
                    ),
                    parse_int(
                        row[6],
                        "Baseline batch index"
                    )
                )
            );
        }
    );

    const auto payload_key =
        [](const Row& row) {
            return baseline_key(
                parse_int(
                    row[2],
                    "Baseline graph model index"
                ),
                parse_int(
                    row[4],
                    "Baseline graph parameter condition"
                ),
                parse_int(
                    row[5],
                    "Baseline graph batch index"
                )
            );
        };

    scan_working_csv(
        layout_.benchmark_graphs_csv(),
        benchmark_graph_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                payload_key(
                    row
                )
            );
        }
    );

    const auto field_payload_key =
        [](const Row& row) {
            return baseline_key(
                parse_int(
                    row[1],
                    "Baseline field model index"
                ),
                parse_int(
                    row[3],
                    "Baseline field parameter condition"
                ),
                parse_int(
                    row[4],
                    "Baseline field batch index"
                )
            );
        };

    scan_working_csv(
        layout_.benchmark_graph_fields_csv(),
        benchmark_graph_field_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                field_payload_key(
                    row
                )
            );
        }
    );

    scan_working_csv(
        layout_.benchmark_vertex_fields_csv(),
        benchmark_vertex_field_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                field_payload_key(
                    row
                )
            );
        }
    );

    scan_working_csv(
        layout_.benchmark_edge_fields_csv(),
        benchmark_edge_field_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                field_payload_key(
                    row
                )
            );
        }
    );

    scan_working_csv(
        layout_.
            benchmark_persistence_diagrams_csv(),
        benchmark_diagram_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                baseline_key(
                    parse_int(
                        row[1],
                        "Baseline diagram model index"
                    ),
                    parse_int(
                        row[3],
                        "Baseline diagram parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "Baseline diagram batch index"
                    )
                )
            );
        }
    );

    scan_working_csv(
        layout_.
            benchmark_harmonic_aggregation_csv(),
        benchmark_harmonic_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                baseline_key(
                    parse_int(
                        row[1],
                        "Baseline harmonic model index"
                    ),
                    parse_int(
                        row[3],
                        "Baseline harmonic parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "Baseline harmonic batch index"
                    )
                )
            );
        }
    );
}

void ExperimentOutput::index_ablation_output() {
    scan_working_csv(
        layout_.
            harmonic_aggregation_ablation_conditions_csv(),
        ablation_condition_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_completed_key(
                ablation_key(
                    parse_int(
                        row[1],
                        "Ablation model index"
                    ),
                    parse_int(
                        row[3],
                        "Ablation parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "Ablation batch index"
                    ),
                    parse_harmonic_aggregation_ablation_kind(
                        row[5]
                    ),
                    parse_int(
                        row[6],
                        "Ablation realization index"
                    )
                )
            );
        }
    );

    scan_working_csv(
        layout_.
            harmonic_aggregation_ablation_features_csv(),
        ablation_harmonic_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                ablation_key(
                    parse_int(
                        row[1],
                        "Ablation harmonic model index"
                    ),
                    parse_int(
                        row[3],
                        "Ablation harmonic parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "Ablation harmonic batch index"
                    ),
                    parse_harmonic_aggregation_ablation_kind(
                        row[5]
                    ),
                    parse_int(
                        row[6],
                        "Ablation harmonic realization index"
                    )
                )
            );
        }
    );
}

void ExperimentOutput::index_robustness_output() {
    scan_working_csv(
        layout_.robustness_conditions_csv(),
        robustness_condition_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_completed_key(
                robustness_key(
                    parse_int(
                        row[1],
                        "Robustness model index"
                    ),
                    parse_int(
                        row[3],
                        "Robustness parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "Robustness batch index"
                    ),
                    parse_int(
                        row[5],
                        "Robustness condition index"
                    ),
                    parse_int(
                        row[8],
                        "Robustness realization index"
                    )
                )
            );
        }
    );

    scan_working_csv(
        layout_.robustness_graphs_csv(),
        robustness_graph_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                robustness_key(
                    parse_int(
                        row[2],
                        "Robustness graph model index"
                    ),
                    parse_int(
                        row[4],
                        "Robustness graph parameter condition"
                    ),
                    parse_int(
                        row[5],
                        "Robustness graph batch index"
                    ),
                    parse_int(
                        row[7],
                        "Robustness graph condition index"
                    ),
                    parse_int(
                        row[10],
                        "Robustness graph realization index"
                    )
                )
            );
        }
    );

    scan_working_csv(
        layout_.
            robustness_persistence_diagrams_csv(),
        robustness_diagram_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                robustness_key(
                    parse_int(
                        row[1],
                        "Robustness diagram model index"
                    ),
                    parse_int(
                        row[3],
                        "Robustness diagram parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "Robustness diagram batch index"
                    ),
                    parse_int(
                        row[6],
                        "Robustness diagram condition index"
                    ),
                    parse_int(
                        row[9],
                        "Robustness diagram realization index"
                    )
                )
            );
        }
    );

    scan_working_csv(
        layout_.
            robustness_harmonic_aggregation_csv(),
        robustness_harmonic_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                robustness_key(
                    parse_int(
                        row[1],
                        "Robustness harmonic model index"
                    ),
                    parse_int(
                        row[3],
                        "Robustness harmonic parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "Robustness harmonic batch index"
                    ),
                    parse_int(
                        row[5],
                        "Robustness harmonic condition index"
                    ),
                    parse_int(
                        row[6],
                        "Robustness harmonic realization index"
                    )
                )
            );
        }
    );
}

void ExperimentOutput::index_batch_size_output() {
    scan_working_csv(
        layout_.
            batch_size_sensitivity_conditions_csv(),
        batch_size_condition_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_completed_key(
                batch_size_key(
                    parse_int(
                        row[1],
                        "Batch-size model index"
                    ),
                    parse_int(
                        row[3],
                        "Batch-size parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "Batch-size batch index"
                    )
                )
            );
        }
    );

    scan_working_csv(
        layout_.
            batch_size_sensitivity_harmonic_aggregation_csv(),
        batch_size_harmonic_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                batch_size_key(
                    parse_int(
                        row[1],
                        "Batch-size harmonic model index"
                    ),
                    parse_int(
                        row[3],
                        "Batch-size harmonic parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "Batch-size harmonic batch index"
                    )
                )
            );
        }
    );
}

void ExperimentOutput::index_rff_output() {
    scan_working_csv(
        layout_.rff_sensitivity_conditions_csv(),
        rff_condition_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_completed_key(
                rff_key(
                    parse_int(
                        row[1],
                        "RFF model index"
                    ),
                    parse_int(
                        row[3],
                        "RFF parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "RFF batch index"
                    ),
                    parse_int(
                        row[5],
                        "RFF replicate index"
                    )
                )
            );
        }
    );

    scan_working_csv(
        layout_.
            rff_sensitivity_harmonic_aggregation_csv(),
        rff_harmonic_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                rff_key(
                    parse_int(
                        row[1],
                        "RFF harmonic model index"
                    ),
                    parse_int(
                        row[3],
                        "RFF harmonic parameter condition"
                    ),
                    parse_int(
                        row[4],
                        "RFF harmonic batch index"
                    ),
                    parse_int(
                        row[5],
                        "RFF harmonic replicate index"
                    )
                )
            );
        }
    );
}

void ExperimentOutput::index_scalability_output() {
    scan_working_csv(
        layout_.scalability_conditions_csv(),
        scalability_condition_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_completed_key(
                scalability_key(
                    parse_int(
                        row[1],
                        "Scalability condition index"
                    )
                )
            );
        }
    );

    scan_working_csv(
        layout_.scalability_timings_csv(),
        scalability_timing_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                scalability_key(
                    parse_int(
                        row[1],
                        "Scalability timing condition index"
                    )
                )
            );
        }
    );

    scan_working_csv(
        layout_.
            scalability_harmonic_aggregation_csv(),
        scalability_harmonic_header(),
        [&](const Row& row) {
            require_schema_version(
                row
            );

            record_payload_key(
                scalability_key(
                    parse_int(
                        row[1],
                        "Scalability harmonic condition index"
                    )
                )
            );
        }
    );
}

void ExperimentOutput::purge_partial_tasks(
    const std::set<std::string>& partial_task_keys
) {
    if (
        partial_task_keys.empty()
    ) {
        return;
    }

    rewrite_without_keys(
        layout_.benchmark_graphs_csv(),
        benchmark_graph_header(),
        partial_task_keys,
        [](const Row& row) {
            return baseline_key(
                parse_int(
                    row[2],
                    "Baseline graph model index"
                ),
                parse_int(
                    row[4],
                    "Baseline graph parameter condition"
                ),
                parse_int(
                    row[5],
                    "Baseline graph batch index"
                )
            );
        }
    );

    const auto baseline_field_key =
        [](const Row& row) {
            return baseline_key(
                parse_int(
                    row[1],
                    "Baseline field model index"
                ),
                parse_int(
                    row[3],
                    "Baseline field parameter condition"
                ),
                parse_int(
                    row[4],
                    "Baseline field batch index"
                )
            );
        };

    rewrite_without_keys(
        layout_.benchmark_graph_fields_csv(),
        benchmark_graph_field_header(),
        partial_task_keys,
        baseline_field_key
    );

    rewrite_without_keys(
        layout_.benchmark_vertex_fields_csv(),
        benchmark_vertex_field_header(),
        partial_task_keys,
        baseline_field_key
    );

    rewrite_without_keys(
        layout_.benchmark_edge_fields_csv(),
        benchmark_edge_field_header(),
        partial_task_keys,
        baseline_field_key
    );

    rewrite_without_keys(
        layout_.
            benchmark_persistence_diagrams_csv(),
        benchmark_diagram_header(),
        partial_task_keys,
        [](const Row& row) {
            return baseline_key(
                parse_int(
                    row[1],
                    "Baseline diagram model index"
                ),
                parse_int(
                    row[3],
                    "Baseline diagram parameter condition"
                ),
                parse_int(
                    row[4],
                    "Baseline diagram batch index"
                )
            );
        }
    );

    rewrite_without_keys(
        layout_.
            benchmark_harmonic_aggregation_csv(),
        benchmark_harmonic_header(),
        partial_task_keys,
        [](const Row& row) {
            return baseline_key(
                parse_int(
                    row[1],
                    "Baseline harmonic model index"
                ),
                parse_int(
                    row[3],
                    "Baseline harmonic parameter condition"
                ),
                parse_int(
                    row[4],
                    "Baseline harmonic batch index"
                )
            );
        }
    );

    rewrite_without_keys(
        layout_.
            harmonic_aggregation_ablation_features_csv(),
        ablation_harmonic_header(),
        partial_task_keys,
        [](const Row& row) {
            return ablation_key(
                parse_int(
                    row[1],
                    "Ablation harmonic model index"
                ),
                parse_int(
                    row[3],
                    "Ablation harmonic parameter condition"
                ),
                parse_int(
                    row[4],
                    "Ablation harmonic batch index"
                ),
                parse_harmonic_aggregation_ablation_kind(
                    row[5]
                ),
                parse_int(
                    row[6],
                    "Ablation harmonic realization index"
                )
            );
        }
    );

    rewrite_without_keys(
        layout_.robustness_graphs_csv(),
        robustness_graph_header(),
        partial_task_keys,
        [](const Row& row) {
            return robustness_key(
                parse_int(
                    row[2],
                    "Robustness graph model index"
                ),
                parse_int(
                    row[4],
                    "Robustness graph parameter condition"
                ),
                parse_int(
                    row[5],
                    "Robustness graph batch index"
                ),
                parse_int(
                    row[7],
                    "Robustness graph condition index"
                ),
                parse_int(
                    row[10],
                    "Robustness graph realization index"
                )
            );
        }
    );

    rewrite_without_keys(
        layout_.
            robustness_persistence_diagrams_csv(),
        robustness_diagram_header(),
        partial_task_keys,
        [](const Row& row) {
            return robustness_key(
                parse_int(
                    row[1],
                    "Robustness diagram model index"
                ),
                parse_int(
                    row[3],
                    "Robustness diagram parameter condition"
                ),
                parse_int(
                    row[4],
                    "Robustness diagram batch index"
                ),
                parse_int(
                    row[6],
                    "Robustness diagram condition index"
                ),
                parse_int(
                    row[9],
                    "Robustness diagram realization index"
                )
            );
        }
    );

    rewrite_without_keys(
        layout_.
            robustness_harmonic_aggregation_csv(),
        robustness_harmonic_header(),
        partial_task_keys,
        [](const Row& row) {
            return robustness_key(
                parse_int(
                    row[1],
                    "Robustness harmonic model index"
                ),
                parse_int(
                    row[3],
                    "Robustness harmonic parameter condition"
                ),
                parse_int(
                    row[4],
                    "Robustness harmonic batch index"
                ),
                parse_int(
                    row[5],
                    "Robustness harmonic condition index"
                ),
                parse_int(
                    row[6],
                    "Robustness harmonic realization index"
                )
            );
        }
    );

    rewrite_without_keys(
        layout_.
            batch_size_sensitivity_harmonic_aggregation_csv(),
        batch_size_harmonic_header(),
        partial_task_keys,
        [](const Row& row) {
            return batch_size_key(
                parse_int(
                    row[1],
                    "Batch-size harmonic model index"
                ),
                parse_int(
                    row[3],
                    "Batch-size harmonic parameter condition"
                ),
                parse_int(
                    row[4],
                    "Batch-size harmonic batch index"
                )
            );
        }
    );

    rewrite_without_keys(
        layout_.
            rff_sensitivity_harmonic_aggregation_csv(),
        rff_harmonic_header(),
        partial_task_keys,
        [](const Row& row) {
            return rff_key(
                parse_int(
                    row[1],
                    "RFF harmonic model index"
                ),
                parse_int(
                    row[3],
                    "RFF harmonic parameter condition"
                ),
                parse_int(
                    row[4],
                    "RFF harmonic batch index"
                ),
                parse_int(
                    row[5],
                    "RFF harmonic replicate index"
                )
            );
        }
    );

    rewrite_without_keys(
        layout_.scalability_timings_csv(),
        scalability_timing_header(),
        partial_task_keys,
        [](const Row& row) {
            return scalability_key(
                parse_int(
                    row[1],
                    "Scalability timing condition index"
                )
            );
        }
    );

    rewrite_without_keys(
        layout_.
            scalability_harmonic_aggregation_csv(),
        scalability_harmonic_header(),
        partial_task_keys,
        [](const Row& row) {
            return scalability_key(
                parse_int(
                    row[1],
                    "Scalability harmonic condition index"
                )
            );
        }
    );
}

void ExperimentOutput::open_writers() {
    benchmark_batches_ =
        std::make_unique<CsvWriter>(
            layout_.benchmark_batches_csv(),
            benchmark_batch_header(),
            resume_
        );

    benchmark_graphs_ =
        std::make_unique<CsvWriter>(
            layout_.benchmark_graphs_csv(),
            benchmark_graph_header(),
            resume_
        );

    benchmark_graph_fields_ =
        std::make_unique<CsvWriter>(
            layout_.benchmark_graph_fields_csv(),
            benchmark_graph_field_header(),
            resume_
        );

    benchmark_vertex_fields_ =
        std::make_unique<CsvWriter>(
            layout_.benchmark_vertex_fields_csv(),
            benchmark_vertex_field_header(),
            resume_
        );

    benchmark_edge_fields_ =
        std::make_unique<CsvWriter>(
            layout_.benchmark_edge_fields_csv(),
            benchmark_edge_field_header(),
            resume_
        );

    benchmark_diagrams_ =
        std::make_unique<CsvWriter>(
            layout_.
                benchmark_persistence_diagrams_csv(),
            benchmark_diagram_header(),
            resume_
        );

    benchmark_harmonic_aggregation_ =
        std::make_unique<CsvWriter>(
            layout_.
                benchmark_harmonic_aggregation_csv(),
            benchmark_harmonic_header(),
            resume_
        );

    ablation_conditions_ =
        std::make_unique<CsvWriter>(
            layout_.
                harmonic_aggregation_ablation_conditions_csv(),
            ablation_condition_header(),
            resume_
        );

    ablation_harmonic_aggregation_ =
        std::make_unique<CsvWriter>(
            layout_.
                harmonic_aggregation_ablation_features_csv(),
            ablation_harmonic_header(),
            resume_
        );

    robustness_conditions_ =
        std::make_unique<CsvWriter>(
            layout_.robustness_conditions_csv(),
            robustness_condition_header(),
            resume_
        );

    robustness_graphs_ =
        std::make_unique<CsvWriter>(
            layout_.robustness_graphs_csv(),
            robustness_graph_header(),
            resume_
        );

    robustness_diagrams_ =
        std::make_unique<CsvWriter>(
            layout_.
                robustness_persistence_diagrams_csv(),
            robustness_diagram_header(),
            resume_
        );

    robustness_harmonic_aggregation_ =
        std::make_unique<CsvWriter>(
            layout_.
                robustness_harmonic_aggregation_csv(),
            robustness_harmonic_header(),
            resume_
        );

    batch_size_conditions_ =
        std::make_unique<CsvWriter>(
            layout_.
                batch_size_sensitivity_conditions_csv(),
            batch_size_condition_header(),
            resume_
        );

    batch_size_harmonic_aggregation_ =
        std::make_unique<CsvWriter>(
            layout_.
                batch_size_sensitivity_harmonic_aggregation_csv(),
            batch_size_harmonic_header(),
            resume_
        );

    rff_conditions_ =
        std::make_unique<CsvWriter>(
            layout_.
                rff_sensitivity_conditions_csv(),
            rff_condition_header(),
            resume_
        );

    rff_harmonic_aggregation_ =
        std::make_unique<CsvWriter>(
            layout_.
                rff_sensitivity_harmonic_aggregation_csv(),
            rff_harmonic_header(),
            resume_
        );

    scalability_conditions_ =
        std::make_unique<CsvWriter>(
            layout_.scalability_conditions_csv(),
            scalability_condition_header(),
            resume_
        );

    scalability_timings_ =
        std::make_unique<CsvWriter>(
            layout_.scalability_timings_csv(),
            scalability_timing_header(),
            resume_
        );

    scalability_harmonic_aggregation_ =
        std::make_unique<CsvWriter>(
            layout_.
                scalability_harmonic_aggregation_csv(),
            scalability_harmonic_header(),
            resume_
        );
}

void ExperimentOutput::write(
    const BenchmarkOutputRecord& record
) {
    const BenchmarkBatch& batch =
        record.batch;

    const std::string key =
        baseline_key(
            batch.model_index,
            batch.parameter_condition_index,
            batch.batch_index
        );

    write_benchmark_graph_rows(
        *benchmark_graphs_,
        batch
    );

    write_benchmark_graph_field_rows(
        *benchmark_graph_fields_,
        batch
    );

    write_benchmark_vertex_field_rows(
        *benchmark_vertex_fields_,
        batch
    );

    write_benchmark_edge_field_rows(
        *benchmark_edge_fields_,
        batch
    );

    write_benchmark_diagram_rows(
        *benchmark_diagrams_,
        batch
    );

    write_harmonic_rows(
        *benchmark_harmonic_aggregation_,
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                batch.model_index
            ),
            batch.descriptor.stable_id,
            csv_integer(
                batch.parameter_condition_index
            ),
            csv_integer(
                batch.batch_index
            ),
            csv_unsigned_integer(
                record.character_seed
            )
        },
        record.harmonic_values
    );

    benchmark_graphs_->flush();

    benchmark_graph_fields_->flush();

    benchmark_vertex_fields_->flush();

    benchmark_edge_fields_->flush();

    benchmark_diagrams_->flush();

    benchmark_harmonic_aggregation_->flush();

    benchmark_batches_->write_row(
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                batch.model_index
            ),
            batch.descriptor.stable_id,
            batch.descriptor.display_name,
            batch.descriptor.variant,
            csv_integer(
                batch.parameter_condition_index
            ),
            csv_integer(
                batch.batch_index
            ),
            csv_size(
                batch.graph_count()
            ),
            csv_unsigned_integer(
                batch.graphs.front().
                    latent_seed
            ),
            csv_unsigned_integer(
                record.character_seed
            )
        }
    );

    benchmark_batches_->flush();

    completed_task_keys_.insert(
        key
    );
}

void ExperimentOutput::write(
    const HarmonicAggregationAblationBatch& result
) {
    const BenchmarkBatch& batch =
        result.benchmark_batch;

    const std::string key =
        ablation_key(
            batch.model_index,
            batch.parameter_condition_index,
            batch.batch_index,
            result.ablation_kind,
            result.realization_index
        );

    write_harmonic_rows(
        *ablation_harmonic_aggregation_,
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                batch.model_index
            ),
            batch.descriptor.stable_id,
            csv_integer(
                batch.parameter_condition_index
            ),
            csv_integer(
                batch.batch_index
            ),
            harmonic_aggregation_ablation_name(
                result.ablation_kind
            ),
            csv_integer(
                result.realization_index
            )
        },
        result.harmonic_values
    );

    ablation_harmonic_aggregation_->flush();

    ablation_conditions_->write_row(
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                batch.model_index
            ),
            batch.descriptor.stable_id,
            csv_integer(
                batch.parameter_condition_index
            ),
            csv_integer(
                batch.batch_index
            ),
            harmonic_aggregation_ablation_name(
                result.ablation_kind
            ),
            csv_integer(
                result.realization_index
            ),
            csv_size(
                batch.graph_count()
            ),
            csv_integer(
                kBaselineAggregationOrder
            ),
            csv_size(
                result.harmonic_values.size()
            ),
            csv_unsigned_integer(
                result.character_seed
            ),
            result.ablation_seed.has_value()
                ? csv_unsigned_integer(
                      *result.ablation_seed
                  )
                : ""
        }
    );

    ablation_conditions_->flush();

    completed_task_keys_.insert(
        key
    );
}

void ExperimentOutput::write(
    const RobustnessOutputRecord& record
) {
    const RobustnessBatch& result =
        record.batch;

    const BenchmarkBatch& clean =
        result.clean_batch;

    const std::string key =
        robustness_key(
            clean.model_index,
            clean.parameter_condition_index,
            clean.batch_index,
            result.condition_index,
            result.realization_index
        );

    write_robustness_graph_rows(
        *robustness_graphs_,
        result
    );

    write_robustness_diagram_rows(
        *robustness_diagrams_,
        result
    );

    write_harmonic_rows(
        *robustness_harmonic_aggregation_,
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                clean.model_index
            ),
            clean.descriptor.stable_id,
            csv_integer(
                clean.parameter_condition_index
            ),
            csv_integer(
                clean.batch_index
            ),
            csv_integer(
                result.condition_index
            ),
            csv_integer(
                result.realization_index
            ),
            csv_unsigned_integer(
                record.character_seed
            )
        },
        record.harmonic_values
    );

    robustness_graphs_->flush();

    robustness_diagrams_->flush();

    robustness_harmonic_aggregation_->flush();

    robustness_conditions_->write_row(
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                clean.model_index
            ),
            clean.descriptor.stable_id,
            csv_integer(
                clean.parameter_condition_index
            ),
            csv_integer(
                clean.batch_index
            ),
            csv_integer(
                result.condition_index
            ),
            robustness_perturbation_name(
                result.condition.perturbation
            ),
            csv_real(
                result.condition.severity
            ),
            csv_integer(
                result.realization_index
            ),
            csv_size(
                result.graph_count()
            ),
            csv_unsigned_integer(
                record.character_seed
            )
        }
    );

    robustness_conditions_->flush();

    completed_task_keys_.insert(
        key
    );
}

void ExperimentOutput::write(
    const BatchSizeSensitivityResult& result
) {
    const BenchmarkBatch& batch =
        result.benchmark_batch;

    const std::string key =
        batch_size_key(
            batch.model_index,
            batch.parameter_condition_index,
            batch.batch_index
        );

    for (
        const BatchSizeSensitivityConditionResult&
            condition :
        result.conditions
    ) {
        write_harmonic_rows(
            *batch_size_harmonic_aggregation_,
            {
                csv_integer(
                    kOutputSchemaVersion
                ),
                csv_integer(
                    batch.model_index
                ),
                batch.descriptor.stable_id,
                csv_integer(
                    batch.parameter_condition_index
                ),
                csv_integer(
                    batch.batch_index
                ),
                csv_integer(
                    condition.graphs_per_batch
                )
            },
            condition.harmonic_values
        );
    }

    batch_size_harmonic_aggregation_->flush();

    batch_size_conditions_->write_row(
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                batch.model_index
            ),
            batch.descriptor.stable_id,
            csv_integer(
                batch.parameter_condition_index
            ),
            csv_integer(
                batch.batch_index
            ),
            csv_size(
                batch.graph_count()
            ),
            csv_unsigned_integer(
                result.character_seed
            ),
            csv_size(
                result.conditions.size()
            )
        }
    );

    batch_size_conditions_->flush();

    completed_task_keys_.insert(
        key
    );
}

void ExperimentOutput::write(
    const RffSensitivityResult& result
) {
    const BenchmarkBatch& batch =
        result.benchmark_batch;

    const std::string key =
        rff_key(
            batch.model_index,
            batch.parameter_condition_index,
            batch.batch_index,
            result.replicate_index
        );

    for (
        const RffSensitivityConditionResult&
            condition :
        result.conditions
    ) {
        write_harmonic_rows(
            *rff_harmonic_aggregation_,
            {
                csv_integer(
                    kOutputSchemaVersion
                ),
                csv_integer(
                    batch.model_index
                ),
                batch.descriptor.stable_id,
                csv_integer(
                    batch.parameter_condition_index
                ),
                csv_integer(
                    batch.batch_index
                ),
                csv_integer(
                    result.replicate_index
                ),
                csv_integer(
                    condition.character_count
                )
            },
            condition.harmonic_values
        );
    }

    rff_harmonic_aggregation_->flush();

    rff_conditions_->write_row(
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                batch.model_index
            ),
            batch.descriptor.stable_id,
            csv_integer(
                batch.parameter_condition_index
            ),
            csv_integer(
                batch.batch_index
            ),
            csv_integer(
                result.replicate_index
            ),
            csv_unsigned_integer(
                result.character_seed
            ),
            csv_size(
                result.conditions.size()
            )
        }
    );

    rff_conditions_->flush();

    completed_task_keys_.insert(
        key
    );
}

void ExperimentOutput::write(
    int scalability_condition_index,
    const ScalabilityResult& result
) {
    const std::string key =
        scalability_key(
            scalability_condition_index
        );

    write_harmonic_rows(
        *scalability_harmonic_aggregation_,
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                scalability_condition_index
            )
        },
        result.harmonic_values
    );

    scalability_timings_->write_row(
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                scalability_condition_index
            ),
            csv_real(
                result.metrics.
                    representation_seconds
            )
        }
    );

    scalability_harmonic_aggregation_->flush();

    scalability_timings_->flush();

    scalability_conditions_->write_row(
        {
            csv_integer(
                kOutputSchemaVersion
            ),
            csv_integer(
                scalability_condition_index
            ),
            csv_integer(
                result.condition.repeat_index
            ),
            csv_integer(
                result.condition.graphs_per_batch
            ),
            csv_size(
                result.condition.input_support_size
            ),
            csv_integer(
                result.condition.character_count
            ),
            csv_unsigned_integer(
                result.sample_seed
            ),
            csv_unsigned_integer(
                result.character_seed
            )
        }
    );

    scalability_conditions_->flush();

    completed_task_keys_.insert(
        key
    );
}

void ExperimentOutput::flush() {
    benchmark_batches_->flush();

    benchmark_graphs_->flush();

    benchmark_graph_fields_->flush();

    benchmark_vertex_fields_->flush();

    benchmark_edge_fields_->flush();

    benchmark_diagrams_->flush();

    benchmark_harmonic_aggregation_->flush();

    ablation_conditions_->flush();

    ablation_harmonic_aggregation_->flush();

    robustness_conditions_->flush();

    robustness_graphs_->flush();

    robustness_diagrams_->flush();

    robustness_harmonic_aggregation_->flush();

    batch_size_conditions_->flush();

    batch_size_harmonic_aggregation_->flush();

    rff_conditions_->flush();

    rff_harmonic_aggregation_->flush();

    scalability_conditions_->flush();

    scalability_timings_->flush();

    scalability_harmonic_aggregation_->flush();
}

void ExperimentOutput::finalize() {
    if (finalized_) {
        return;
    }

    benchmark_graphs_->finalize();

    benchmark_graph_fields_->finalize();

    benchmark_vertex_fields_->finalize();

    benchmark_edge_fields_->finalize();

    benchmark_diagrams_->finalize();

    benchmark_harmonic_aggregation_->finalize();

    benchmark_batches_->finalize();

    ablation_harmonic_aggregation_->finalize();

    ablation_conditions_->finalize();

    robustness_graphs_->finalize();

    robustness_diagrams_->finalize();

    robustness_harmonic_aggregation_->finalize();

    robustness_conditions_->finalize();

    batch_size_harmonic_aggregation_->finalize();

    batch_size_conditions_->finalize();

    rff_harmonic_aggregation_->finalize();

    rff_conditions_->finalize();

    scalability_harmonic_aggregation_->finalize();

    scalability_timings_->finalize();

    scalability_conditions_->finalize();

    finalized_ =
        true;
}

bool ExperimentOutput::is_finalized() const {
    return finalized_;
}

}  // namespace vpd