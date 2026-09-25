#include "seed_derivation.hpp"

#include <cstdint>

namespace vpd {

namespace {

constexpr std::uint64_t kCoordinateStride =
    0x9e3779b97f4a7c15ULL;

constexpr std::uint64_t kArityTag =
    0xd1b54a32d192ed03ULL;

}  // namespace

std::uint64_t splitmix64(
    std::uint64_t value
) {
    value +=
        0x9e3779b97f4a7c15ULL;

    value =
        (
            value ^
            (
                value >>
                30U
            )
        ) *
        0xbf58476d1ce4e5b9ULL;

    value =
        (
            value ^
            (
                value >>
                27U
            )
        ) *
        0x94d049bb133111ebULL;

    return
        value ^
        (
            value >>
            31U
        );
}

std::uint64_t derive_seed(
    std::uint64_t base_seed,
    SeedDomain domain,
    std::initializer_list<std::uint64_t> coordinates
) {
    std::uint64_t state =
        splitmix64(
            base_seed
        );

    state =
        splitmix64(
            state ^
            static_cast<std::uint64_t>(
                domain
            )
        );

    std::uint64_t position =
        1U;

    for (std::uint64_t coordinate :
         coordinates) {
        const std::uint64_t tagged_coordinate =
            splitmix64(
                coordinate ^
                (
                    kCoordinateStride *
                    position
                )
            );

        state =
            splitmix64(
                state ^
                tagged_coordinate
            );

        ++position;
    }

    const std::uint64_t coordinate_count =
        static_cast<std::uint64_t>(
            coordinates.size()
        );

    return splitmix64(
        state ^
        splitmix64(
            coordinate_count ^
            kArityTag
        )
    );
}

}  // namespace vpd