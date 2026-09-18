#include "random_fourier_features.hpp"

#include "../../../../core/utils/portable_random.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>

namespace vpd {

namespace {

std::complex<double> character_value(
    const Atom1& atom,
    const std::array<double, 2>& frequencies
) {
    const double argument =
        frequencies[0] *
            atom.birth +
        frequencies[1] *
            atom.death;

    return {
        std::cos(
            argument
        ),
        std::sin(
            argument
        )
    };
}

}  // namespace

std::size_t CharacterBank::size() const {
    return values.size();
}

CharacterBank sample_character_bank(
    int character_count,
    std::uint64_t seed
) {
    CharacterBank bank;

    bank.values.reserve(
        static_cast<std::size_t>(
            character_count
        )
    );

    std::mt19937_64 engine(
        seed
    );

    NormalGenerator normal(
        engine
    );

    for (int character_index = 0;
         character_index <
             character_count;
         ++character_index) {
        CharacterSpec character;

        character.character_index =
            character_index;

        character.left_frequencies = {
            normal.sample(),
            normal.sample()
        };

        character.right_frequencies = {
            normal.sample(),
            normal.sample()
        };

        bank.values.push_back(
            character
        );
    }

    return bank;
}

std::complex<double> left_character_value(
    const Atom1& atom,
    const CharacterSpec& character
) {
    return character_value(
        atom,
        character.left_frequencies
    );
}

std::complex<double> right_character_value(
    const Atom1& atom,
    const CharacterSpec& character
) {
    return character_value(
        atom,
        character.right_frequencies
    );
}

}  // namespace vpd