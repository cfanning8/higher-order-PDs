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

class KahanFenwickTree {
public:
    explicit KahanFenwickTree(
        std::size_t size
    )
        : tree_(
              size + 1U
          ) {
    }

    void add(
        std::size_t index,
        double value
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

    double prefix_sum(
        std::size_t index
    ) const {
        KahanAccumulator result;

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
    std::vector<KahanAccumulator>
        tree_;
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
    double weighted_coefficient{};
    std::size_t death_index{};
};

struct PreparedDiagram {
    std::vector<PreparedAtom> atoms;
    std::size_t death_count{};
};

enum class FeatureFamily {
    linear,
    weighted_linear,
    relationship,
    weighted_relationship
};

enum class CharacterSide {
    left,
    right
};

struct HarmonicFeatureSpecification {
    FeatureFamily family{};
    std::size_t character_index{};
    CharacterSide side{};
};

struct SelectedFrequency {
    std::size_t character_index{};
    CharacterSide side{};
};

struct ExactStatistics {
    double mass{};
    double weighted_mass{};
    double relationship_mass{};
    double weighted_relationship_mass{};
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

        const double weighted_coefficient =
            coefficient *
            persistence_length(
                atom
            );

        prepared.atoms.push_back(
            {
                &atom,
                coefficient,
                weighted_coefficient,
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

SelectedFrequency select_frequency(
    std::size_t feature_index,
    std::size_t feature_count,
    std::size_t character_count
) {
    if (
        feature_count == 0U ||
        character_count == 0U
    ) {
        return {};
    }

    const std::size_t character_index =
        feature_index *
        character_count /
        feature_count;

    const std::size_t remainder =
        feature_index *
            character_count %
        feature_count;

    const bool use_right =
        remainder >=
        (
            feature_count + 1U
        ) /
            2U;

    return {
        character_index,
        use_right
            ? CharacterSide::right
            : CharacterSide::left
    };
}

void append_feature_family(
    std::vector<HarmonicFeatureSpecification>& specifications,
    FeatureFamily family,
    std::size_t feature_count,
    std::size_t character_count
) {
    for (
        std::size_t feature_index = 0U;
        feature_index <
            feature_count;
        ++feature_index
    ) {
        const SelectedFrequency frequency =
            select_frequency(
                feature_index,
                feature_count,
                character_count
            );

        specifications.push_back(
            {
                family,
                frequency.character_index,
                frequency.side
            }
        );
    }
}

std::vector<HarmonicFeatureSpecification>
make_feature_specifications(
    const CharacterBank& characters
) {
    const std::size_t character_count =
        characters.size();

    std::vector<HarmonicFeatureSpecification>
        specifications;

    if (
        character_count <= 4U
    ) {
        return specifications;
    }

    const std::size_t harmonic_count =
        character_count -
        4U;

    specifications.reserve(
        harmonic_count
    );

    const std::size_t base_count =
        harmonic_count /
        4U;

    const std::size_t remainder =
        harmonic_count %
        4U;

    const std::size_t linear_count =
        base_count +
        (
            remainder >
                0U
                ? 1U
                : 0U
        );

    const std::size_t weighted_linear_count =
        base_count +
        (
            remainder >
                1U
                ? 1U
                : 0U
        );

    const std::size_t relationship_count =
        base_count +
        (
            remainder >
                2U
                ? 1U
                : 0U
        );

    const std::size_t weighted_relationship_count =
        base_count;

    append_feature_family(
        specifications,
        FeatureFamily::linear,
        linear_count,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::weighted_linear,
        weighted_linear_count,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::relationship,
        relationship_count,
        character_count
    );

    append_feature_family(
        specifications,
        FeatureFamily::weighted_relationship,
        weighted_relationship_count,
        character_count
    );

    return specifications;
}

std::complex<double> selected_character_value(
    const Atom1& atom,
    const CharacterSpec& character,
    CharacterSide side
) {
    switch (
        side
    ) {
        case CharacterSide::left:
            return
                left_character_value(
                    atom,
                    character
                );

        case CharacterSide::right:
            return
                right_character_value(
                    atom,
                    character
                );
    }

    throw std::logic_error(
        "invalid harmonic aggregation character side"
    );
}

std::complex<double> linear_fourier(
    const PreparedDiagram& prepared,
    const CharacterSpec& character,
    CharacterSide side,
    bool weighted
) {
    ComplexKahanAccumulator total;

    for (
        const PreparedAtom& prepared_atom :
        prepared.atoms
    ) {
        const double coefficient =
            weighted
                ? prepared_atom.weighted_coefficient
                : prepared_atom.coefficient;

        total.add(
            coefficient *
            selected_character_value(
                *prepared_atom.atom,
                character,
                side
            )
        );
    }

    return total.value();
}

std::complex<double> relationship_fourier(
    const PreparedDiagram& prepared,
    const CharacterSpec& character,
    CharacterSide side,
    bool weighted
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

    for (
        const PreparedAtom& prepared_atom :
        prepared.atoms
    ) {
        const double coefficient =
            weighted
                ? prepared_atom.weighted_coefficient
                : prepared_atom.coefficient;

        const std::complex<double>
            character_value =
                selected_character_value(
                    *prepared_atom.atom,
                    character,
                    side
                );

        const std::complex<double>
            source_sum =
                source_tree.prefix_sum(
                    prepared_atom.death_index
                );

        total.add(
            coefficient *
            character_value *
            source_sum
        );

        source_tree.add(
            prepared_atom.death_index,
            coefficient *
            std::conj(
                character_value
            )
        );
    }

    return total.value();
}

ExactStatistics exact_statistics(
    const PreparedDiagram& prepared
) {
    KahanAccumulator mass;
    KahanAccumulator weighted_mass;
    KahanAccumulator relationship_mass;
    KahanAccumulator weighted_relationship_mass;

    KahanFenwickTree source_tree(
        prepared.death_count
    );

    KahanFenwickTree weighted_source_tree(
        prepared.death_count
    );

    for (
        const PreparedAtom& prepared_atom :
        prepared.atoms
    ) {
        mass.add(
            prepared_atom.coefficient
        );

        weighted_mass.add(
            prepared_atom.weighted_coefficient
        );

        const double source_sum =
            source_tree.prefix_sum(
                prepared_atom.death_index
            );

        const double weighted_source_sum =
            weighted_source_tree.prefix_sum(
                prepared_atom.death_index
            );

        relationship_mass.add(
            prepared_atom.coefficient *
            source_sum
        );

        weighted_relationship_mass.add(
            prepared_atom.weighted_coefficient *
            weighted_source_sum
        );

        source_tree.add(
            prepared_atom.death_index,
            prepared_atom.coefficient
        );

        weighted_source_tree.add(
            prepared_atom.death_index,
            prepared_atom.weighted_coefficient
        );
    }

    return {
        mass.value(),
        weighted_mass.value(),
        relationship_mass.value(),
        weighted_relationship_mass.value()
    };
}

std::complex<double> evaluate_feature(
    const PreparedDiagram& prepared,
    const CharacterSpec& character,
    const HarmonicFeatureSpecification& specification
) {
    switch (
        specification.family
    ) {
        case FeatureFamily::linear:
            return
                linear_fourier(
                    prepared,
                    character,
                    specification.side,
                    false
                );

        case FeatureFamily::weighted_linear:
            return
                linear_fourier(
                    prepared,
                    character,
                    specification.side,
                    true
                );

        case FeatureFamily::relationship:
            return
                relationship_fourier(
                    prepared,
                    character,
                    specification.side,
                    false
                );

        case FeatureFamily::weighted_relationship:
            return
                relationship_fourier(
                    prepared,
                    character,
                    specification.side,
                    true
                );
    }

    throw std::logic_error(
        "invalid harmonic aggregation feature family"
    );
}

void append_exact_statistics(
    HarmonicValues& result,
    const ExactStatistics& statistics,
    std::size_t output_count
) {
    if (
        result.size() <
        output_count
    ) {
        result.emplace_back(
            statistics.mass,
            0.0
        );
    }

    if (
        result.size() <
        output_count
    ) {
        result.emplace_back(
            statistics.weighted_mass,
            0.0
        );
    }

    if (
        result.size() <
        output_count
    ) {
        result.emplace_back(
            statistics.relationship_mass,
            0.0
        );
    }

    if (
        result.size() <
        output_count
    ) {
        result.emplace_back(
            statistics.weighted_relationship_mass,
            0.0
        );
    }
}

HarmonicValues evaluate_prepared_pair_fourier(
    const PreparedDiagram& prepared,
    const CharacterBank& characters
) {
    const std::size_t character_count =
        characters.size();

    HarmonicValues result;

    result.reserve(
        character_count
    );

    const ExactStatistics statistics =
        exact_statistics(
            prepared
        );

    if (
        character_count <= 4U
    ) {
        append_exact_statistics(
            result,
            statistics,
            character_count
        );

        return result;
    }

    const std::vector<HarmonicFeatureSpecification>
        specifications =
            make_feature_specifications(
                characters
            );

    for (
        const HarmonicFeatureSpecification& specification :
        specifications
    ) {
        result.push_back(
            evaluate_feature(
                prepared,
                characters.values[
                    specification.character_index
                ],
                specification
            )
        );
    }

    append_exact_statistics(
        result,
        statistics,
        character_count
    );

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