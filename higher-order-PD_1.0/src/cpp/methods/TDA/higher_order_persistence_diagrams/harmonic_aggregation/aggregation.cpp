#include "aggregation.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
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
    std::size_t death_count{};
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

    std::vector<double> deaths;

    deaths.reserve(
        diagram.size()
    );

    for (
        const auto& [
            atom,
            coefficient
        ] : diagram
    ) {
        (void)coefficient;

        deaths.push_back(
            atom.death
        );
    }

    std::sort(
        deaths.begin(),
        deaths.end()
    );

    deaths.erase(
        std::unique(
            deaths.begin(),
            deaths.end()
        ),
        deaths.end()
    );

    prepared.death_count =
        deaths.size();

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
                deaths.begin(),
                deaths.end(),
                atom.death
            );

        prepared.atoms.push_back(
            {
                &atom,
                coefficient,
                static_cast<std::size_t>(
                    death_iterator -
                    deaths.begin()
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

    const std::size_t marginal_left_count =
        marginal_left_end;

    const std::size_t marginal_right_count =
        marginal_right_end -
        marginal_left_end;

    const std::size_t relative_left_count =
        relative_left_end -
        marginal_right_end;

    const std::size_t relative_right_count =
        relative_right_end -
        relative_left_end;

    const std::size_t joint_left_right_count =
        joint_left_right_end -
        relative_right_end;

    const std::size_t joint_right_left_count =
        character_count -
        joint_left_right_end;

    append_feature_family(
        specifications,
        FeatureFamily::marginal_left,
        marginal_left_count,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::marginal_right,
        marginal_right_count,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::relative_left,
        relative_left_count,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::relative_right,
        relative_right_count,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::joint_left_right,
        joint_left_right_count,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::joint_right_left,
        joint_right_left_count,
        character_count
    );

    return specifications;
}

std::complex<double> marginal_fourier(
    const PreparedDiagram& prepared,
    const CharacterSpec& character,
    FeatureFamily family
) {
    ComplexKahanAccumulator total;

    for (
        const PreparedAtom& prepared_atom :
        prepared.atoms
    ) {
        std::complex<double>
            character_value;

        switch (
            family
        ) {
            case FeatureFamily::marginal_left:
                character_value =
                    left_character_value(
                        *prepared_atom.atom,
                        character
                    );
                break;

            case FeatureFamily::marginal_right:
                character_value =
                    right_character_value(
                        *prepared_atom.atom,
                        character
                    );
                break;

            default:
                throw std::logic_error(
                    "invalid marginal feature family"
                );
        }

        total.add(
            prepared_atom.coefficient *
            character_value
        );
    }

    return total.value();
}

std::complex<double> relationship_fourier(
    const PreparedDiagram& prepared,
    const CharacterSpec& character,
    FeatureFamily family
) {
    if (
        prepared.atoms.empty()
    ) {
        return {};
    }

    ComplexFenwickTree source_tree(
        prepared.death_count
    );

    ComplexKahanAccumulator total;

    std::size_t group_begin =
        0U;

    while (
        group_begin <
        prepared.atoms.size()
    ) {
        const double birth =
            prepared.atoms[
                group_begin
            ].atom->birth;

        std::size_t group_end =
            group_begin + 1U;

        while (
            group_end <
                prepared.atoms.size() &&
            prepared.atoms[
                group_end
            ].atom->birth ==
                birth
        ) {
            ++group_end;
        }

        for (
            std::size_t index =
                group_begin;
            index <
                group_end;
            ++index
        ) {
            const PreparedAtom& prepared_atom =
                prepared.atoms[
                    index
                ];

            std::complex<double>
                source_value;

            switch (
                family
            ) {
                case FeatureFamily::relative_left:
                case FeatureFamily::joint_left_right:
                    source_value =
                        left_character_value(
                            *prepared_atom.atom,
                            character
                        );
                    break;

                case FeatureFamily::relative_right:
                case FeatureFamily::joint_right_left:
                    source_value =
                        right_character_value(
                            *prepared_atom.atom,
                            character
                        );
                    break;

                default:
                    throw std::logic_error(
                        "invalid relationship feature family"
                    );
            }

            source_tree.add(
                prepared_atom.death_index,
                prepared_atom.coefficient *
                    std::conj(
                        source_value
                    )
            );
        }

        for (
            std::size_t index =
                group_begin;
            index <
                group_end;
            ++index
        ) {
            const PreparedAtom& prepared_atom =
                prepared.atoms[
                    index
                ];

            std::complex<double>
                target_value;

            switch (
                family
            ) {
                case FeatureFamily::relative_left:
                case FeatureFamily::joint_right_left:
                    target_value =
                        left_character_value(
                            *prepared_atom.atom,
                            character
                        );
                    break;

                case FeatureFamily::relative_right:
                case FeatureFamily::joint_left_right:
                    target_value =
                        right_character_value(
                            *prepared_atom.atom,
                            character
                        );
                    break;

                default:
                    throw std::logic_error(
                        "invalid relationship feature family"
                    );
            }

            const std::complex<double>
                source_sum =
                    source_tree.prefix_sum(
                        prepared_atom.death_index
                    );

            total.add(
                prepared_atom.coefficient *
                target_value *
                source_sum
            );
        }

        group_begin =
            group_end;
    }

    return total.value();
}

std::complex<double> evaluate_feature(
    const PreparedDiagram& prepared,
    const CharacterSpec& character,
    FeatureFamily family
) {
    switch (
        family
    ) {
        case FeatureFamily::marginal_left:
        case FeatureFamily::marginal_right:
            return
                marginal_fourier(
                    prepared,
                    character,
                    family
                );

        case FeatureFamily::relative_left:
        case FeatureFamily::relative_right:
        case FeatureFamily::joint_left_right:
        case FeatureFamily::joint_right_left:
            return
                relationship_fourier(
                    prepared,
                    character,
                    family
                );
    }

    throw std::logic_error(
        "invalid harmonic aggregation feature family"
    );
}

HarmonicValues evaluate_prepared_pair_fourier(
    const PreparedDiagram& prepared,
    const CharacterBank& characters
) {
    const std::vector<FeatureSpecification>
        specifications =
            make_feature_specifications(
                characters
            );

    HarmonicValues result;

    result.reserve(
        characters.size()
    );

    for (
        const FeatureSpecification& specification :
        specifications
    ) {
        result.push_back(
            evaluate_feature(
                prepared,
                characters.values[
                    specification.character_index
                ],
                specification.family
            )
        );
    }

    return result;
}

}  // namespace

std::complex<double> pair_fourier(
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

HarmonicValues evaluate_pair_fourier(
    const VPD1& diagram,
    const CharacterBank& characters
) {
    if (
        characters.size() == 0U
    ) {
        return {};
    }

    if (
        diagram.empty()
    ) {
        return HarmonicValues(
            characters.size(),
            std::complex<double>{}
        );
    }

    const PreparedDiagram prepared =
        prepare_diagram(
            diagram
        );

    return
        evaluate_prepared_pair_fourier(
            prepared,
            characters
        );
}

HarmonicValues mean_pair_fourier(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters
) {
    if (
        diagrams.empty()
    ) {
        throw std::invalid_argument(
            "mean_pair_fourier requires at least one diagram"
        );
    }

    std::vector<ComplexKahanAccumulator>
        accumulators(
            characters.size()
        );

    for (
        const VPD1& diagram :
        diagrams
    ) {
        const HarmonicValues values =
            evaluate_pair_fourier(
                diagram,
                characters
            );

        for (
            std::size_t index = 0U;
            index < values.size();
            ++index
        ) {
            accumulators[
                index
            ].add(
                values[
                    index
                ]
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
        characters.size()
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

}  // namespace vpd