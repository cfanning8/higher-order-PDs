#include "harmonic_aggregation_ablation.hpp"

#include "../../../core/utils/portable_random.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace vpd {

namespace {

class KahanAccumulator {
public:
    void add(
        double value
    ) {
        const double adjusted =
            value -
            compensation_;

        const double temporary =
            sum_ +
            adjusted;

        compensation_ =
            (
                temporary -
                sum_
            ) -
            adjusted;

        sum_ =
            temporary;
    }

    double value() const {
        return sum_;
    }

private:
    double sum_{};
    double compensation_{};
};

class ComplexKahanAccumulator {
public:
    void add(
        const std::complex<double>& value
    ) {
        real_.add(
            value.real()
        );

        imaginary_.add(
            value.imag()
        );
    }

    std::complex<double> value() const {
        return {
            real_.value(),
            imaginary_.value()
        };
    }

private:
    KahanAccumulator real_;
    KahanAccumulator imaginary_;
};

class ComplexFenwickTree {
public:
    explicit ComplexFenwickTree(
        std::size_t size
    )
        : tree_(
              size + 1U
          ) {
    }

    void add(
        std::size_t index,
        const std::complex<double>& value
    ) {
        for (
            std::size_t tree_index =
                index + 1U;
            tree_index <
                tree_.size();
            tree_index +=
                tree_index &
                (~tree_index + 1U)
        ) {
            tree_[
                tree_index
            ].add(
                value
            );
        }
    }

    std::complex<double> prefix_sum(
        std::size_t index
    ) const {
        ComplexKahanAccumulator result;

        for (
            std::size_t tree_index =
                index + 1U;
            tree_index >
                0U;
            tree_index -=
                tree_index &
                (~tree_index + 1U)
        ) {
            result.add(
                tree_[
                    tree_index
                ].value()
            );
        }

        return result.value();
    }

private:
    std::vector<ComplexKahanAccumulator>
        tree_;
};

struct PreparedAtom {
    const Atom1* atom{};
    double coefficient{};
    std::size_t death_index{};
};

struct PreparedDiagram {
    std::vector<PreparedAtom> atoms;
    std::vector<double> deaths;
};

enum class FeatureFamily {
    marginal_left,
    marginal_right,
    relative_left,
    relative_right,
    joint_left_right,
    joint_right_left
};

struct FeatureSpecification {
    FeatureFamily family{};
    std::size_t character_index{};
};

PreparedDiagram prepare_diagram(
    const VPD1& diagram
) {
    PreparedDiagram prepared;

    if (
        diagram.empty()
    ) {
        return prepared;
    }

    prepared.deaths.reserve(
        diagram.size()
    );

    for (
        const auto& [
            atom,
            coefficient
        ] : diagram
    ) {
        (void)coefficient;

        prepared.deaths.push_back(
            atom.death
        );
    }

    std::sort(
        prepared.deaths.begin(),
        prepared.deaths.end()
    );

    prepared.deaths.erase(
        std::unique(
            prepared.deaths.begin(),
            prepared.deaths.end()
        ),
        prepared.deaths.end()
    );

    prepared.atoms.reserve(
        diagram.size()
    );

    for (
        const auto& [
            atom,
            coefficient
        ] : diagram
    ) {
        const auto death_iterator =
            std::lower_bound(
                prepared.deaths.begin(),
                prepared.deaths.end(),
                atom.death
            );

        prepared.atoms.push_back(
            {
                &atom,
                coefficient,
                static_cast<std::size_t>(
                    death_iterator -
                    prepared.deaths.begin()
                )
            }
        );
    }

    std::sort(
        prepared.atoms.begin(),
        prepared.atoms.end(),
        [](
            const PreparedAtom& first,
            const PreparedAtom& second
        ) {
            if (
                first.atom->birth !=
                second.atom->birth
            ) {
                return
                    first.atom->birth >
                    second.atom->birth;
            }

            return
                first.atom->death <
                second.atom->death;
        }
    );

    return prepared;
}

std::size_t proportional_character_index(
    std::size_t feature_index,
    std::size_t feature_count,
    std::size_t character_count
) {
    if (
        feature_count == 0U ||
        character_count == 0U
    ) {
        return 0U;
    }

    return
        feature_index *
        character_count /
        feature_count;
}

void append_feature_family(
    std::vector<FeatureSpecification>& specifications,
    FeatureFamily family,
    std::size_t feature_count,
    std::size_t character_count
) {
    for (
        std::size_t feature_index = 0U;
        feature_index < feature_count;
        ++feature_index
    ) {
        specifications.push_back(
            {
                family,
                proportional_character_index(
                    feature_index,
                    feature_count,
                    character_count
                )
            }
        );
    }
}

std::vector<FeatureSpecification>
make_feature_specifications(
    const CharacterBank& characters
) {
    const std::size_t character_count =
        characters.size();

    std::vector<FeatureSpecification>
        specifications;

    specifications.reserve(
        character_count
    );

    if (
        character_count == 0U
    ) {
        return specifications;
    }

    const std::size_t marginal_left_end =
        character_count *
        2U /
        8U;

    const std::size_t marginal_right_end =
        character_count *
        4U /
        8U;

    const std::size_t relative_left_end =
        character_count *
        5U /
        8U;

    const std::size_t relative_right_end =
        character_count *
        6U /
        8U;

    const std::size_t joint_left_right_end =
        character_count *
        7U /
        8U;

    append_feature_family(
        specifications,
        FeatureFamily::marginal_left,
        marginal_left_end,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::marginal_right,
        marginal_right_end -
            marginal_left_end,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::relative_left,
        relative_left_end -
            marginal_right_end,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::relative_right,
        relative_right_end -
            relative_left_end,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::joint_left_right,
        joint_left_right_end -
            relative_right_end,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::joint_right_left,
        character_count -
            joint_left_right_end,
        character_count
    );

    return specifications;
}

bool is_marginal_family(
    FeatureFamily family
) {
    return
        family ==
            FeatureFamily::marginal_left ||
        family ==
            FeatureFamily::marginal_right;
}

std::complex<double> source_character_value(
    const Atom1& atom,
    const CharacterSpec& character,
    FeatureFamily family
) {
    switch (
        family
    ) {
        case FeatureFamily::relative_left:
        case FeatureFamily::joint_left_right:
            return
                left_character_value(
                    atom,
                    character
                );

        case FeatureFamily::relative_right:
        case FeatureFamily::joint_right_left:
            return
                right_character_value(
                    atom,
                    character
                );

        default:
            throw std::logic_error(
                "invalid relationship feature family"
            );
    }
}

std::complex<double> target_character_value(
    const Atom1& atom,
    const CharacterSpec& character,
    FeatureFamily family
) {
    switch (
        family
    ) {
        case FeatureFamily::relative_left:
        case FeatureFamily::joint_right_left:
            return
                left_character_value(
                    atom,
                    character
                );

        case FeatureFamily::relative_right:
        case FeatureFamily::joint_left_right:
            return
                right_character_value(
                    atom,
                    character
                );

        default:
            throw std::logic_error(
                "invalid relationship feature family"
            );
    }
}

std::complex<double> marginal_value(
    const VPD1& diagram,
    const CharacterSpec& character,
    FeatureFamily family
) {
    ComplexKahanAccumulator total;

    for (
        const auto& [
            atom,
            coefficient
        ] : diagram
    ) {
        switch (
            family
        ) {
            case FeatureFamily::marginal_left:
                total.add(
                    coefficient *
                    left_character_value(
                        atom,
                        character
                    )
                );
                break;

            case FeatureFamily::marginal_right:
                total.add(
                    coefficient *
                    right_character_value(
                        atom,
                        character
                    )
                );
                break;

            default:
                throw std::logic_error(
                    "invalid marginal feature family"
                );
        }
    }

    return total.value();
}

std::complex<double> unrestricted_relationship_value(
    const VPD1& diagram,
    const CharacterSpec& character,
    FeatureFamily family
) {
    ComplexKahanAccumulator source_sum;
    ComplexKahanAccumulator target_sum;

    for (
        const auto& [
            atom,
            coefficient
        ] : diagram
    ) {
        source_sum.add(
            coefficient *
            std::conj(
                source_character_value(
                    atom,
                    character,
                    family
                )
            )
        );

        target_sum.add(
            coefficient *
            target_character_value(
                atom,
                character,
                family
            )
        );
    }

    return
        source_sum.value() *
        target_sum.value();
}

std::complex<double> cross_observation_relationship_value(
    const PreparedDiagram& source,
    const PreparedDiagram& target,
    const CharacterSpec& character,
    FeatureFamily family
) {
    if (
        source.atoms.empty() ||
        target.atoms.empty()
    ) {
        return {};
    }

    ComplexFenwickTree source_tree(
        source.deaths.size()
    );

    ComplexKahanAccumulator total;

    std::size_t source_index =
        0U;

    for (
        const PreparedAtom& target_atom :
        target.atoms
    ) {
        while (
            source_index <
                source.atoms.size() &&
            source.atoms[
                source_index
            ].atom->birth >=
                target_atom.atom->birth
        ) {
            const PreparedAtom& source_atom =
                source.atoms[
                    source_index
                ];

            source_tree.add(
                source_atom.death_index,
                source_atom.coefficient *
                std::conj(
                    source_character_value(
                        *source_atom.atom,
                        character,
                        family
                    )
                )
            );

            ++source_index;
        }

        const auto upper =
            std::upper_bound(
                source.deaths.begin(),
                source.deaths.end(),
                target_atom.atom->death
            );

        if (
            upper ==
            source.deaths.begin()
        ) {
            continue;
        }

        const std::size_t death_index =
            static_cast<std::size_t>(
                upper -
                source.deaths.begin() -
                1
            );

        const std::complex<double>
            comparable_source_sum =
                source_tree.prefix_sum(
                    death_index
                );

        total.add(
            target_atom.coefficient *
            target_character_value(
                *target_atom.atom,
                character,
                family
            ) *
            comparable_source_sum
        );
    }

    return total.value();
}

std::complex<double> linear_character_value(
    const VPD1& diagram,
    const CharacterSpec& character
) {
    ComplexKahanAccumulator left;
    ComplexKahanAccumulator right;

    for (
        const auto& [
            atom,
            coefficient
        ] : diagram
    ) {
        left.add(
            coefficient *
            left_character_value(
                atom,
                character
            )
        );

        right.add(
            coefficient *
            right_character_value(
                atom,
                character
            )
        );
    }

    return
        0.5 *
        (
            left.value() +
            right.value()
        );
}

void validate_nonempty_diagrams(
    const std::vector<VPD1>& diagrams
) {
    if (
        diagrams.empty()
    ) {
        throw std::invalid_argument(
            "harmonic-aggregation ablation requires at least one diagram"
        );
    }
}

void validate_cross_observation_permutation(
    std::size_t diagram_count,
    const DiagramPermutation& permutation
) {
    if (
        diagram_count < 2U
    ) {
        throw std::invalid_argument(
            "cross-observation ablation requires at least two diagrams"
        );
    }

    if (
        permutation.size() !=
        diagram_count
    ) {
        throw std::invalid_argument(
            "cross-observation permutation has incorrect size"
        );
    }

    std::vector<bool> seen(
        diagram_count,
        false
    );

    for (
        std::size_t index = 0U;
        index < diagram_count;
        ++index
    ) {
        const std::size_t value =
            permutation[
                index
            ];

        if (
            value >=
            diagram_count
        ) {
            throw std::invalid_argument(
                "cross-observation permutation contains an invalid index"
            );
        }

        if (
            value ==
            index
        ) {
            throw std::invalid_argument(
                "cross-observation permutation must be a derangement"
            );
        }

        if (
            seen[
                value
            ]
        ) {
            throw std::invalid_argument(
                "cross-observation mapping is not a permutation"
            );
        }

        seen[
            value
        ] =
            true;
    }
}

template <class Evaluate>
HarmonicValues mean_values(
    const std::vector<VPD1>& diagrams,
    std::size_t value_count,
    const Evaluate& evaluate
) {
    validate_nonempty_diagrams(
        diagrams
    );

    std::vector<ComplexKahanAccumulator>
        accumulators(
            value_count
        );

    for (
        std::size_t diagram_index = 0U;
        diagram_index <
            diagrams.size();
        ++diagram_index
    ) {
        for (
            std::size_t value_index = 0U;
            value_index <
                value_count;
            ++value_index
        ) {
            accumulators[
                value_index
            ].add(
                evaluate(
                    diagram_index,
                    value_index
                )
            );
        }
    }

    const double inverse_count =
        1.0 /
        static_cast<double>(
            diagrams.size()
        );

    HarmonicValues result;

    result.reserve(
        value_count
    );

    for (
        const ComplexKahanAccumulator& accumulator :
        accumulators
    ) {
        result.push_back(
            accumulator.value() *
            inverse_count
        );
    }

    return result;
}

bool is_derangement(
    const DiagramPermutation& permutation
) {
    for (
        std::size_t index = 0U;
        index <
            permutation.size();
        ++index
    ) {
        if (
            permutation[
                index
            ] ==
            index
        ) {
            return false;
        }
    }

    return true;
}

}  // namespace

const char*
harmonic_aggregation_ablation_name(
    HarmonicAggregationAblationKind kind
) {
    switch (
        kind
    ) {
        case HarmonicAggregationAblationKind::
                CrossObservation:
            return
                "cross_observation";

        case HarmonicAggregationAblationKind::
                Linear:
            return
                "linear";

        case HarmonicAggregationAblationKind::
                NoPreorder:
            return
                "no_preorder";
    }

    throw std::logic_error(
        "Unknown harmonic-aggregation ablation kind."
    );
}

DiagramPermutation
make_cross_observation_derangement(
    std::size_t diagram_count,
    std::uint64_t seed
) {
    if (
        diagram_count < 2U
    ) {
        throw std::invalid_argument(
            "cross-observation derangement requires at least two diagrams"
        );
    }

    DiagramPermutation permutation(
        diagram_count
    );

    std::iota(
        permutation.begin(),
        permutation.end(),
        std::size_t{
            0U
        }
    );

    std::mt19937_64 generator(
        seed
    );

    do {
        portable_shuffle(
            permutation,
            generator
        );
    } while (
        !is_derangement(
            permutation
        )
    );

    return permutation;
}

HarmonicValues
evaluate_cross_observation_ablation(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters,
    const DiagramPermutation& permutation
) {
    validate_nonempty_diagrams(
        diagrams
    );

    validate_cross_observation_permutation(
        diagrams.size(),
        permutation
    );

    const std::vector<FeatureSpecification>
        specifications =
            make_feature_specifications(
                characters
            );

    std::vector<PreparedDiagram>
        prepared_diagrams;

    prepared_diagrams.reserve(
        diagrams.size()
    );

    for (
        const VPD1& diagram :
        diagrams
    ) {
        prepared_diagrams.push_back(
            prepare_diagram(
                diagram
            )
        );
    }

    return mean_values(
        diagrams,
        specifications.size(),
        [&](
            std::size_t diagram_index,
            std::size_t feature_index
        ) {
            const FeatureSpecification& specification =
                specifications[
                    feature_index
                ];

            const CharacterSpec& character =
                characters.values[
                    specification.character_index
                ];

            if (
                is_marginal_family(
                    specification.family
                )
            ) {
                return
                    marginal_value(
                        diagrams[
                            diagram_index
                        ],
                        character,
                        specification.family
                    );
            }

            return
                cross_observation_relationship_value(
                    prepared_diagrams[
                        diagram_index
                    ],
                    prepared_diagrams[
                        permutation[
                            diagram_index
                        ]
                    ],
                    character,
                    specification.family
                );
        }
    );
}

HarmonicValues
evaluate_linear_ablation(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters
) {
    return mean_values(
        diagrams,
        characters.size(),
        [&](
            std::size_t diagram_index,
            std::size_t character_index
        ) {
            return
                linear_character_value(
                    diagrams[
                        diagram_index
                    ],
                    characters.values[
                        character_index
                    ]
                );
        }
    );
}

HarmonicValues
evaluate_no_preorder_ablation(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters
) {
    const std::vector<FeatureSpecification>
        specifications =
            make_feature_specifications(
                characters
            );

    return mean_values(
        diagrams,
        specifications.size(),
        [&](
            std::size_t diagram_index,
            std::size_t feature_index
        ) {
            const FeatureSpecification& specification =
                specifications[
                    feature_index
                ];

            const CharacterSpec& character =
                characters.values[
                    specification.character_index
                ];

            if (
                is_marginal_family(
                    specification.family
                )
            ) {
                return
                    marginal_value(
                        diagrams[
                            diagram_index
                        ],
                        character,
                        specification.family
                    );
            }

            return
                unrestricted_relationship_value(
                    diagrams[
                        diagram_index
                    ],
                    character,
                    specification.family
                );
        }
    );
}

}  // namespace vpd