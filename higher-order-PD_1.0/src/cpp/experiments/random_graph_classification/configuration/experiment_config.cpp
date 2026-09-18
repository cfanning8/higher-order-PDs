#include "experiment_config.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace vpd {

namespace {

std::vector<RobustnessCondition>
make_robustness_conditions() {
    std::vector<RobustnessCondition>
        conditions;

    const auto append_conditions =
        [&conditions](
            RobustnessPerturbation perturbation,
            const std::vector<double>& levels
        ) {
            for (double level :
                 levels) {
                conditions.push_back(
                    RobustnessCondition{
                        perturbation,
                        level
                    }
                );
            }
        };

    append_conditions(
        RobustnessPerturbation::EdgeDeletion,
        {
            0.02,
            0.05,
            0.10,
            0.20
        }
    );

    append_conditions(
        RobustnessPerturbation::EdgeInsertion,
        {
            0.02,
            0.05,
            0.10,
            0.20
        }
    );

    append_conditions(
        RobustnessPerturbation::
            DegreePreservingRewiring,
        {
            0.02,
            0.05,
            0.10,
            0.20
        }
    );

    append_conditions(
        RobustnessPerturbation::FiltrationNoise,
        {
            0.01,
            0.025,
            0.05,
            0.10
        }
    );

    return conditions;
}

CanonicalExperimentSpecification
make_canonical_experiment_specification() {
    CanonicalExperimentSpecification
        specification;

    /*
     * Every model uses the same graph size. Each model
     * contributes the same number of batches, distributed
     * uniformly across its canonical parameter conditions.
     * Erdos-Renyi has one implicit parameter condition.
     */
    specification.benchmark =
        BenchmarkDesignSpecification{
            GraphSizeSpecification{
                100,
                200
            },
            100,
            4,
            ModelParameterSpecification{
                {
                    WattsStrogatzParameters{
                        4,
                        0.05
                    },
                    WattsStrogatzParameters{
                        4,
                        0.10
                    },
                    WattsStrogatzParameters{
                        4,
                        0.20
                    },
                    WattsStrogatzParameters{
                        4,
                        0.40
                    },
                    WattsStrogatzParameters{
                        4,
                        0.80
                    }
                },
                {
                    BarabasiAlbertParameters{
                        2,
                        5
                    }
                },
                {
                    ConfigurationModelParameters{
                        4
                    }
                },
                {
                    StochasticBlockModelParameters{
                        4,
                        0.5
                    },
                    StochasticBlockModelParameters{
                        4,
                        1.0
                    },
                    StochasticBlockModelParameters{
                        4,
                        1.5
                    },
                    StochasticBlockModelParameters{
                        4,
                        2.0
                    },
                    StochasticBlockModelParameters{
                        4,
                        3.0
                    }
                },
                {
                    ChungLuParameters{
                        2.2
                    },
                    ChungLuParameters{
                        2.4
                    },
                    ChungLuParameters{
                        2.6
                    },
                    ChungLuParameters{
                        2.8
                    },
                    ChungLuParameters{
                        3.0
                    }
                },
                {
                    KleinbergParameters{
                        10,
                        10,
                        20,
                        1.0
                    },
                    KleinbergParameters{
                        10,
                        10,
                        20,
                        1.5
                    },
                    KleinbergParameters{
                        10,
                        10,
                        20,
                        2.0
                    },
                    KleinbergParameters{
                        10,
                        10,
                        20,
                        2.5
                    },
                    KleinbergParameters{
                        10,
                        10,
                        20,
                        3.0
                    }
                },
                {
                    GirgParameters{
                        2.2,
                        1.0
                    },
                    GirgParameters{
                        2.4,
                        1.0
                    },
                    GirgParameters{
                        2.6,
                        1.0
                    },
                    GirgParameters{
                        2.6,
                        1.5
                    },
                    GirgParameters{
                        2.6,
                        2.0
                    }
                },
                {
                    HyperbolicRandomGraphParameters{
                        2.2,
                        0.3,
                        -1.0,
                        8.0
                    },
                    HyperbolicRandomGraphParameters{
                        2.4,
                        0.3,
                        -1.0,
                        8.0
                    },
                    HyperbolicRandomGraphParameters{
                        2.6,
                        0.3,
                        -1.0,
                        8.0
                    },
                    HyperbolicRandomGraphParameters{
                        2.6,
                        0.5,
                        -1.0,
                        8.0
                    },
                    HyperbolicRandomGraphParameters{
                        2.6,
                        0.7,
                        -1.0,
                        8.0
                    }
                },
                {
                    ExponentialRandomGraphModelParameters{
                        0.25,
                        0.5,
                        100,
                        20
                    },
                    ExponentialRandomGraphModelParameters{
                        0.50,
                        0.5,
                        100,
                        20
                    },
                    ExponentialRandomGraphModelParameters{
                        0.75,
                        0.5,
                        100,
                        20
                    },
                    ExponentialRandomGraphModelParameters{
                        1.00,
                        0.5,
                        100,
                        20
                    },
                    ExponentialRandomGraphModelParameters{
                        1.25,
                        0.5,
                        100,
                        20
                    }
                }
            }
        };

    /*
     * Character-count conditions within each replicate use
     * prefixes of one maximally sampled character bank.
     * Distinct replicate indices receive independent banks.
     */
    specification.random_features =
        RandomFeatureSpecification{
            256,
            10,
            {
                64,
                128,
                256,
                512,
                1024
            }
        };

    /*
     * The clean benchmark observation supplies severity zero,
     * so every explicit robustness condition has positive
     * severity.
     */
    specification.robustness =
        RobustnessSpecification{
            10,
            make_robustness_conditions()
        };

    specification.batch_size_sensitivity =
        BatchSizeSensitivitySpecification{
            {
                1,
                2,
                3,
                4,
                5,
                6,
                7
            }
        };

    /*
     * Scalability varies observations per batch,
     * persistence-diagram input support size, and
     * random-feature character count.
     */
    specification.scalability =
        ScalabilitySpecification{
            {
                1,
                2,
                4,
                8,
                16,
                32
            },
            {
                16U,
                32U,
                64U,
                128U,
                256U
            },
            {
                64,
                128,
                256,
                512,
                1024
            },
            30
        };

    specification.
        cross_observation_ablation_realizations =
            10;

    /*
     * Every stochastic stream derives from this root through
     * the repository seed-derivation API.
     */
    specification.root_seed =
        20260601ULL;

    return specification;
}

}  // namespace

const CanonicalExperimentSpecification&
canonical_experiment_specification() {
    static const
        CanonicalExperimentSpecification
            specification =
                make_canonical_experiment_specification();

    return specification;
}

std::uint64_t experiment_root_seed(
    const ExperimentConfig& config
) {
    if (
        config.root_seed_override.has_value()
    ) {
        return
            *config.root_seed_override;
    }

    return
        canonical_experiment_specification().
            root_seed;
}

ExperimentConfig default_experiment_config() {
    ExperimentConfig config;

    config.root =
        ".";

    config.output_root =
        std::filesystem::path(
            "results"
        ) /
        "raw" /
        "random_graph_classification";

    config.root_seed_override =
        std::nullopt;

    config.maximum_estimated_output_gib =
        20.0;

    config.estimated_diagram_atoms_per_graph =
        100.0;

    config.resume =
        false;

    config.force =
        false;

    config.allow_compiler_mismatch =
        false;

    config.allow_build_type_mismatch =
        false;

    config.dry_run =
        false;

    return config;
}

void validate_experiment_config(
    ExperimentConfig& config
) {
    if (
        !std::isfinite(
            config.maximum_estimated_output_gib
        ) ||
        config.maximum_estimated_output_gib <=
            0.0
    ) {
        throw std::invalid_argument(
            "maximum-estimated-output-gib must be "
            "finite and positive."
        );
    }

    if (
        !std::isfinite(
            config.estimated_diagram_atoms_per_graph
        ) ||
        config.estimated_diagram_atoms_per_graph <
            0.0
    ) {
        throw std::invalid_argument(
            "estimated-diagram-atoms-per-graph must be "
            "finite and nonnegative."
        );
    }

    if (
        config.resume &&
        config.force
    ) {
        throw std::invalid_argument(
            "--resume and --force cannot be combined."
        );
    }

    config.root =
        std::filesystem::absolute(
            config.root
        ).lexically_normal();

    if (
        config.output_root.is_relative()
    ) {
        config.output_root =
            config.root /
            config.output_root;
    }

    config.output_root =
        std::filesystem::absolute(
            config.output_root
        ).lexically_normal();
}

}  // namespace vpd