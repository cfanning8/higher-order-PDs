#ifndef GRAPH_BENCHMARK_RANDOM_FOURIER_FEATURES_HPP
#define GRAPH_BENCHMARK_RANDOM_FOURIER_FEATURES_HPP

#include "../../persistent_homology/diagrams.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpd {

struct CharacterSpec {
    int character_index{};

    std::array<double, 2>
        left_frequencies{};

    std::array<double, 2>
        right_frequencies{};
};

struct CharacterBank {
    std::vector<CharacterSpec> values;

    std::size_t size() const;
};

CharacterBank sample_character_bank(
    int character_count,
    std::uint64_t seed
);

std::complex<double> left_character_value(
    const Atom1& atom,
    const CharacterSpec& character
);

std::complex<double> right_character_value(
    const Atom1& atom,
    const CharacterSpec& character
);

}  // namespace vpd

#endif