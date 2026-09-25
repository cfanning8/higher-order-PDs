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
    std::vector<double> deaths;
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
};

struct NoPreorderStatistics {
    double relationship_mass{};
    double weighted_relationship_mass{};
};

struct CrossObservationStatistics {
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
                coefficient *
                    persistence_length(
                        atom
                    ),
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

    const std::size_t frequency_count =
        2U *
        character_count;

    const std::size_t frequency_index =
        feature_index *
        frequency_count /
        feature_count;

    return {
        frequency_index /
            2U,
        frequency_index %
                    2U ==
                0U
            ? CharacterSide::left
            : CharacterSide::right
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

std::vector<HarmonicFeatureSpecification>
make_linear_feature_specifications(
    const CharacterBank& characters
) {
    const std::size_t character_count =
        characters.size();

    std::vector<HarmonicFeatureSpecification>
        specifications;

    if (
        character_count <= 2U
    ) {
        return specifications;
    }

    const std::size_t harmonic_count =
        character_count -
        2U;

    specifications.reserve(
        harmonic_count
    );

    const std::size_t linear_count =
        (
            harmonic_count +
            1U
        ) /
        2U;

    const std::size_t weighted_linear_count =
        harmonic_count -
        linear_count;

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
        "invalid harmonic-aggregation character side"
    );
}

double selected_coefficient(
    const PreparedAtom& prepared_atom,
    bool weighted
) {
    return
        weighted
            ? prepared_atom.weighted_coefficient
            : prepared_atom.coefficient;
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
        total.add(
            selected_coefficient(
                prepared_atom,
                weighted
            ) *
            selected_character_value(
                *prepared_atom.atom,
                character,
                side
            )
        );
    }

    return total.value();
}

std::complex<double> unrestricted_relationship_value(
    const PreparedDiagram& prepared,
    const CharacterSpec& character,
    CharacterSide side,
    bool weighted
) {
    ComplexKahanAccumulator total;
    KahanAccumulator diagonal;

    for (
        const PreparedAtom& prepared_atom :
        prepared.atoms
    ) {
        const double coefficient =
            selected_coefficient(
                prepared_atom,
                weighted
            );

        total.add(
            coefficient *
            selected_character_value(
                *prepared_atom.atom,
                character,
                side
            )
        );

        diagonal.add(
            coefficient *
            coefficient
        );
    }

    const std::complex<double>
        linear_value =
            total.value();

    return
        std::conj(
            linear_value
        ) *
            linear_value -
        std::complex<double>(
            diagonal.value(),
            0.0
        );
}

const PreparedAtom* find_equivalent_atom(
    const PreparedDiagram& prepared,
    const Atom1& atom
) {
    const auto iterator =
        std::lower_bound(
            prepared.atoms.begin(),
            prepared.atoms.end(),
            atom,
            [](
                const PreparedAtom& prepared_atom,
                const Atom1& target
            ) {
                if (
                    prepared_atom.atom->birth !=
                    target.birth
                ) {
                    return
                        prepared_atom.atom->birth >
                        target.birth;
                }

                return
                    prepared_atom.atom->death <
                    target.death;
            }
        );

    if (
        iterator ==
            prepared.atoms.end() ||
        !(
            *iterator->atom ==
            atom
        )
    ) {
        return nullptr;
    }

    return &*iterator;
}

std::complex<double> cross_observation_relationship_value(
    const PreparedDiagram& source,
    const PreparedDiagram& target,
    const CharacterSpec& character,
    CharacterSide side,
    bool weighted
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
                selected_coefficient(
                    source_atom,
                    weighted
                ) *
                std::conj(
                    selected_character_value(
                        *source_atom.atom,
                        character,
                        side
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

        std::complex<double>
            comparable_source_sum =
                source_tree.prefix_sum(
                    death_index
                );

        const PreparedAtom*
            equivalent_source =
                find_equivalent_atom(
                    source,
                    *target_atom.atom
                );

        if (
            equivalent_source !=
            nullptr
        ) {
            comparable_source_sum -=
                selected_coefficient(
                    *equivalent_source,
                    weighted
                ) *
                std::conj(
                    selected_character_value(
                        *equivalent_source->atom,
                        character,
                        side
                    )
                );
        }

        total.add(
            selected_coefficient(
                target_atom,
                weighted
            ) *
            selected_character_value(
                *target_atom.atom,
                character,
                side
            ) *
            comparable_source_sum
        );
    }

    return total.value();
}

ExactStatistics exact_statistics(
    const PreparedDiagram& prepared
) {
    KahanAccumulator mass;
    KahanAccumulator weighted_mass;

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
    }

    return {
        mass.value(),
        weighted_mass.value()
    };
}

NoPreorderStatistics no_preorder_statistics(
    const PreparedDiagram& prepared,
    const ExactStatistics& statistics
) {
    KahanAccumulator diagonal;
    KahanAccumulator weighted_diagonal;

    for (
        const PreparedAtom& prepared_atom :
        prepared.atoms
    ) {
        diagonal.add(
            prepared_atom.coefficient *
            prepared_atom.coefficient
        );

        weighted_diagonal.add(
            prepared_atom.weighted_coefficient *
            prepared_atom.weighted_coefficient
        );
    }

    return {
        statistics.mass *
            statistics.mass -
            diagonal.value(),
        statistics.weighted_mass *
            statistics.weighted_mass -
            weighted_diagonal.value()
    };
}

CrossObservationStatistics
cross_observation_statistics(
    const PreparedDiagram& source,
    const PreparedDiagram& target
) {
    if (
        source.atoms.empty() ||
        target.atoms.empty()
    ) {
        return {};
    }

    KahanFenwickTree source_tree(
        source.deaths.size()
    );

    KahanFenwickTree weighted_source_tree(
        source.deaths.size()
    );

    KahanAccumulator relationship_mass;
    KahanAccumulator weighted_relationship_mass;

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
                source_atom.coefficient
            );

            weighted_source_tree.add(
                source_atom.death_index,
                source_atom.weighted_coefficient
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

        double comparable_source_sum =
            source_tree.prefix_sum(
                death_index
            );

        double weighted_comparable_source_sum =
            weighted_source_tree.prefix_sum(
                death_index
            );

        const PreparedAtom*
            equivalent_source =
                find_equivalent_atom(
                    source,
                    *target_atom.atom
                );

        if (
            equivalent_source !=
            nullptr
        ) {
            comparable_source_sum -=
                equivalent_source->
                    coefficient;

            weighted_comparable_source_sum -=
                equivalent_source->
                    weighted_coefficient;
        }

        relationship_mass.add(
            target_atom.coefficient *
            comparable_source_sum
        );

        weighted_relationship_mass.add(
            target_atom.weighted_coefficient *
            weighted_comparable_source_sum
        );
    }

    return {
        relationship_mass.value(),
        weighted_relationship_mass.value()
    };
}

std::complex<double> evaluate_linear_feature(
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
        case FeatureFamily::weighted_relationship:
            break;
    }

    throw std::logic_error(
        "invalid linear harmonic-aggregation feature family"
    );
}

std::complex<double>
evaluate_no_preorder_feature(
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
                unrestricted_relationship_value(
                    prepared,
                    character,
                    specification.side,
                    false
                );

        case FeatureFamily::weighted_relationship:
            return
                unrestricted_relationship_value(
                    prepared,
                    character,
                    specification.side,
                    true
                );
    }

    throw std::logic_error(
        "invalid no-preorder feature family"
    );
}

std::complex<double>
evaluate_cross_observation_feature(
    const PreparedDiagram& source,
    const PreparedDiagram& target,
    const CharacterSpec& character,
    const HarmonicFeatureSpecification& specification
) {
    switch (
        specification.family
    ) {
        case FeatureFamily::linear:
            return
                linear_fourier(
                    source,
                    character,
                    specification.side,
                    false
                );

        case FeatureFamily::weighted_linear:
            return
                linear_fourier(
                    source,
                    character,
                    specification.side,
                    true
                );

        case FeatureFamily::relationship:
            return
                cross_observation_relationship_value(
                    source,
                    target,
                    character,
                    specification.side,
                    false
                );

        case FeatureFamily::weighted_relationship:
            return
                cross_observation_relationship_value(
                    source,
                    target,
                    character,
                    specification.side,
                    true
                );
    }

    throw std::logic_error(
        "invalid cross-observation feature family"
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
        index <
            diagram_count;
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

    const std::size_t character_count =
        characters.size();

    if (
        character_count == 0U
    ) {
        return {};
    }

    const std::vector<HarmonicFeatureSpecification>
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

    std::vector<ExactStatistics>
        exact_values;

    exact_values.reserve(
        diagrams.size()
    );

    for (
        const PreparedDiagram& prepared :
        prepared_diagrams
    ) {
        exact_values.push_back(
            exact_statistics(
                prepared
            )
        );
    }

    std::vector<CrossObservationStatistics>
        relationship_values;

    relationship_values.reserve(
        diagrams.size()
    );

    for (
        std::size_t diagram_index = 0U;
        diagram_index <
            diagrams.size();
        ++diagram_index
    ) {
        relationship_values.push_back(
            cross_observation_statistics(
                prepared_diagrams[
                    diagram_index
                ],
                prepared_diagrams[
                    permutation[
                        diagram_index
                    ]
                ]
            )
        );
    }

    if (
        character_count <= 4U
    ) {
        return mean_values(
            diagrams,
            character_count,
            [&](
                std::size_t diagram_index,
                std::size_t value_index
            ) {
                if (
                    value_index == 0U
                ) {
                    return std::complex<double>(
                        exact_values[
                            diagram_index
                        ].mass,
                        0.0
                    );
                }

                if (
                    value_index == 1U
                ) {
                    return std::complex<double>(
                        exact_values[
                            diagram_index
                        ].weighted_mass,
                        0.0
                    );
                }

                if (
                    value_index == 2U
                ) {
                    return std::complex<double>(
                        relationship_values[
                            diagram_index
                        ].relationship_mass,
                        0.0
                    );
                }

                return std::complex<double>(
                    relationship_values[
                        diagram_index
                    ].weighted_relationship_mass,
                    0.0
                );
            }
        );
    }

    const std::size_t harmonic_count =
        specifications.size();

    return mean_values(
        diagrams,
        character_count,
        [&](
            std::size_t diagram_index,
            std::size_t value_index
        ) {
            if (
                value_index <
                harmonic_count
            ) {
                const HarmonicFeatureSpecification&
                    specification =
                        specifications[
                            value_index
                        ];

                return
                    evaluate_cross_observation_feature(
                        prepared_diagrams[
                            diagram_index
                        ],
                        prepared_diagrams[
                            permutation[
                                diagram_index
                            ]
                        ],
                        characters.values[
                            specification.
                                character_index
                        ],
                        specification
                    );
            }

            const std::size_t scalar_index =
                value_index -
                harmonic_count;

            if (
                scalar_index == 0U
            ) {
                return std::complex<double>(
                    exact_values[
                        diagram_index
                    ].mass,
                    0.0
                );
            }

            if (
                scalar_index == 1U
            ) {
                return std::complex<double>(
                    exact_values[
                        diagram_index
                    ].weighted_mass,
                    0.0
                );
            }

            if (
                scalar_index == 2U
            ) {
                return std::complex<double>(
                    relationship_values[
                        diagram_index
                    ].relationship_mass,
                    0.0
                );
            }

            return std::complex<double>(
                relationship_values[
                    diagram_index
                ].weighted_relationship_mass,
                0.0
            );
        }
    );
}

HarmonicValues
evaluate_linear_ablation(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters
) {
    validate_nonempty_diagrams(
        diagrams
    );

    const std::size_t character_count =
        characters.size();

    if (
        character_count == 0U
    ) {
        return {};
    }

    const std::vector<HarmonicFeatureSpecification>
        specifications =
            make_linear_feature_specifications(
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

    std::vector<ExactStatistics>
        exact_values;

    exact_values.reserve(
        diagrams.size()
    );

    for (
        const PreparedDiagram& prepared :
        prepared_diagrams
    ) {
        exact_values.push_back(
            exact_statistics(
                prepared
            )
        );
    }

    if (
        character_count <= 2U
    ) {
        return mean_values(
            diagrams,
            character_count,
            [&](
                std::size_t diagram_index,
                std::size_t value_index
            ) {
                if (
                    value_index == 0U
                ) {
                    return std::complex<double>(
                        exact_values[
                            diagram_index
                        ].mass,
                        0.0
                    );
                }

                return std::complex<double>(
                    exact_values[
                        diagram_index
                    ].weighted_mass,
                    0.0
                );
            }
        );
    }

    const std::size_t harmonic_count =
        specifications.size();

    return mean_values(
        diagrams,
        character_count,
        [&](
            std::size_t diagram_index,
            std::size_t value_index
        ) {
            if (
                value_index <
                harmonic_count
            ) {
                const HarmonicFeatureSpecification&
                    specification =
                        specifications[
                            value_index
                        ];

                return
                    evaluate_linear_feature(
                        prepared_diagrams[
                            diagram_index
                        ],
                        characters.values[
                            specification.
                                character_index
                        ],
                        specification
                    );
            }

            if (
                value_index ==
                harmonic_count
            ) {
                return std::complex<double>(
                    exact_values[
                        diagram_index
                    ].mass,
                    0.0
                );
            }

            return std::complex<double>(
                exact_values[
                    diagram_index
                ].weighted_mass,
                0.0
            );
        }
    );
}

HarmonicValues
evaluate_no_preorder_ablation(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters
) {
    validate_nonempty_diagrams(
        diagrams
    );

    const std::size_t character_count =
        characters.size();

    if (
        character_count == 0U
    ) {
        return {};
    }

    const std::vector<HarmonicFeatureSpecification>
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

    std::vector<ExactStatistics>
        exact_values;

    exact_values.reserve(
        diagrams.size()
    );

    std::vector<NoPreorderStatistics>
        relationship_values;

    relationship_values.reserve(
        diagrams.size()
    );

    for (
        const PreparedDiagram& prepared :
        prepared_diagrams
    ) {
        const ExactStatistics statistics =
            exact_statistics(
                prepared
            );

        exact_values.push_back(
            statistics
        );

        relationship_values.push_back(
            no_preorder_statistics(
                prepared,
                statistics
            )
        );
    }

    if (
        character_count <= 4U
    ) {
        return mean_values(
            diagrams,
            character_count,
            [&](
                std::size_t diagram_index,
                std::size_t value_index
            ) {
                if (
                    value_index == 0U
                ) {
                    return std::complex<double>(
                        exact_values[
                            diagram_index
                        ].mass,
                        0.0
                    );
                }

                if (
                    value_index == 1U
                ) {
                    return std::complex<double>(
                        exact_values[
                            diagram_index
                        ].weighted_mass,
                        0.0
                    );
                }

                if (
                    value_index == 2U
                ) {
                    return std::complex<double>(
                        relationship_values[
                            diagram_index
                        ].relationship_mass,
                        0.0
                    );
                }

                return std::complex<double>(
                    relationship_values[
                        diagram_index
                    ].weighted_relationship_mass,
                    0.0
                );
            }
        );
    }

    const std::size_t harmonic_count =
        specifications.size();

    return mean_values(
        diagrams,
        character_count,
        [&](
            std::size_t diagram_index,
            std::size_t value_index
        ) {
            if (
                value_index <
                harmonic_count
            ) {
                const HarmonicFeatureSpecification&
                    specification =
                        specifications[
                            value_index
                        ];

                return
                    evaluate_no_preorder_feature(
                        prepared_diagrams[
                            diagram_index
                        ],
                        characters.values[
                            specification.
                                character_index
                        ],
                        specification
                    );
            }

            const std::size_t scalar_index =
                value_index -
                harmonic_count;

            if (
                scalar_index == 0U
            ) {
                return std::complex<double>(
                    exact_values[
                        diagram_index
                    ].mass,
                    0.0
                );
            }

            if (
                scalar_index == 1U
            ) {
                return std::complex<double>(
                    exact_values[
                        diagram_index
                    ].weighted_mass,
                    0.0
                );
            }

            if (
                scalar_index == 2U
            ) {
                return std::complex<double>(
                    relationship_values[
                        diagram_index
                    ].relationship_mass,
                    0.0
                );
            }

            return std::complex<double>(
                relationship_values[
                    diagram_index
                ].weighted_relationship_mass,
                0.0
            );
        }
    );
}

}  // namespace vpd