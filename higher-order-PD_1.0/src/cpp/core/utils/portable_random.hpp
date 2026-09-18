#ifndef GRAPH_BENCHMARK_PORTABLE_RANDOM_HPP
#define GRAPH_BENCHMARK_PORTABLE_RANDOM_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace vpd {

double uniform_01(
    std::mt19937_64& engine
);

double uniform_open_01(
    std::mt19937_64& engine
);

std::uint64_t bounded_uint64(
    std::mt19937_64& engine,
    std::uint64_t upper_exclusive
);

int uniform_int(
    std::mt19937_64& engine,
    int lower,
    int upper
);

std::size_t uniform_index(
    std::mt19937_64& engine,
    std::size_t size
);

std::size_t weighted_index(
    std::mt19937_64& engine,
    const std::vector<double>& weights
);

template <class RandomAccessIterator>
void portable_shuffle(
    RandomAccessIterator first,
    RandomAccessIterator last,
    std::mt19937_64& engine
) {
    auto size =
        last -
        first;

    for (decltype(size) remaining = size;
         remaining > 1;
         --remaining) {
        const auto offset =
            static_cast<decltype(size)>(
                bounded_uint64(
                    engine,
                    static_cast<std::uint64_t>(
                        remaining
                    )
                )
            );

        std::iter_swap(
            first +
                (
                    remaining -
                    1
                ),
            first +
                offset
        );
    }
}

template <class Container>
void portable_shuffle(
    Container& values,
    std::mt19937_64& engine
) {
    portable_shuffle(
        values.begin(),
        values.end(),
        engine
    );
}

class NormalGenerator {
public:
    explicit NormalGenerator(
        std::mt19937_64& engine
    );

    double sample();

private:
    std::mt19937_64& engine_;

    bool has_cached_{};

    double cached_{};
};

}  // namespace vpd

#endif