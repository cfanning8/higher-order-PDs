#include "command_line.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace vpd {

namespace {

std::string option_value(
    int& index,
    int argc,
    char** argv,
    const std::string& option
) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            "Option '" +
            option +
            "' requires a value."
        );
    }

    ++index;

    return argv[index];
}

std::uint64_t parse_uint64(
    const std::string& text,
    const std::string& option
) {
    std::uint64_t value{};

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
        throw std::invalid_argument(
            "Option '" +
            option +
            "' requires an unsigned integer."
        );
    }

    return value;
}

double parse_real(
    const std::string& text,
    const std::string& option
) {
    double value{};

    const char* const begin =
        text.data();

    const char* const end =
        begin +
        text.size();

    const auto result =
        std::from_chars(
            begin,
            end,
            value,
            std::chars_format::general
        );

    if (
        result.ec != std::errc{} ||
        result.ptr != end ||
        !std::isfinite(
            value
        )
    ) {
        throw std::invalid_argument(
            "Option '" +
            option +
            "' requires a finite real value."
        );
    }

    return value;
}

}  // namespace

void print_usage(
    const char* executable
) {
    std::cout
        << "Usage: "
        << executable
        << " [options]\n"
        << "\n"
        << "The scientific experiment is fixed by the "
        << "canonical experiment specification.\n"
        << "Command-line options control execution only.\n"
        << "\n"
        << "Paths:\n"
        << "  --root PATH\n"
        << "  --output-root PATH\n"
        << "\n"
        << "Reproducibility:\n"
        << "  --seed UINT64    Override the canonical "
        << "experiment root seed\n"
        << "\n"
        << "Output control:\n"
        << "  --maximum-estimated-output-gib FLOAT\n"
        << "  --estimated-diagram-atoms-per-graph FLOAT\n"
        << "  --resume\n"
        << "  --force\n"
        << "  --allow-compiler-mismatch\n"
        << "  --allow-build-type-mismatch\n"
        << "\n"
        << "Execution:\n"
        << "  --dry-run\n"
        << "  --help\n";
}

ExperimentConfig parse_arguments(
    int argc,
    char** argv
) {
    ExperimentConfig config =
        default_experiment_config();

    for (int index = 1;
         index < argc;
         ++index) {
        const std::string option =
            argv[index];

        if (option == "--help") {
            print_usage(
                argc > 0
                    ? argv[0]
                    : "random_graph_classification"
            );

            std::exit(
                0
            );
        }

        if (option == "--root") {
            config.root =
                option_value(
                    index,
                    argc,
                    argv,
                    option
                );

            continue;
        }

        if (option == "--output-root") {
            config.output_root =
                option_value(
                    index,
                    argc,
                    argv,
                    option
                );

            continue;
        }

        if (option == "--seed") {
            config.root_seed_override =
                parse_uint64(
                    option_value(
                        index,
                        argc,
                        argv,
                        option
                    ),
                    option
                );

            continue;
        }

        if (
            option ==
            "--maximum-estimated-output-gib"
        ) {
            config.maximum_estimated_output_gib =
                parse_real(
                    option_value(
                        index,
                        argc,
                        argv,
                        option
                    ),
                    option
                );

            continue;
        }

        if (
            option ==
            "--estimated-diagram-atoms-per-graph"
        ) {
            config.estimated_diagram_atoms_per_graph =
                parse_real(
                    option_value(
                        index,
                        argc,
                        argv,
                        option
                    ),
                    option
                );

            continue;
        }

        if (option == "--resume") {
            config.resume =
                true;

            continue;
        }

        if (option == "--force") {
            config.force =
                true;

            continue;
        }

        if (
            option ==
            "--allow-compiler-mismatch"
        ) {
            config.allow_compiler_mismatch =
                true;

            continue;
        }

        if (
            option ==
            "--allow-build-type-mismatch"
        ) {
            config.allow_build_type_mismatch =
                true;

            continue;
        }

        if (option == "--dry-run") {
            config.dry_run =
                true;

            continue;
        }

        throw std::invalid_argument(
            "Unknown option '" +
            option +
            "'."
        );
    }

    validate_experiment_config(
        config
    );

    return config;
}

}  // namespace vpd