#ifndef GRAPH_BENCHMARK_SEED_DERIVATION_HPP
#define GRAPH_BENCHMARK_SEED_DERIVATION_HPP

#include <cstdint>
#include <initializer_list>

namespace vpd {

enum class SeedDomain : std::uint64_t {
    GraphLatent =
        0x243f6a8885a308d3ULL,

    GraphSampling =
        0x13198a2e03707344ULL,

    Robustness =
        0xa4093822299f31d0ULL,

    CharacterBank =
        0x082efa98ec4e6c89ULL,

    CrossObservationAblation =
        0x452821e638d01377ULL,

    Scalability =
        0xbe5466cf34e90c6cULL
};

std::uint64_t splitmix64(
    std::uint64_t value
);

std::uint64_t derive_seed(
    std::uint64_t base_seed,
    SeedDomain domain,
    std::initializer_list<std::uint64_t> coordinates
);

}  // namespace vpd

#endif