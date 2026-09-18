#include "portable_random.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace vpd {

namespace {

constexpr double kTwoPi =
    6.283185307179586476925286766559005768;

constexpr double kTwoToThe53 =
    9007199254740992.0;

}  // namespace

double uniform_01(
    std::mt19937_64& engine
) {
    return
        static_cast<double>(
            engine() >>
            11U
        ) /
        kTwoToThe53;
}

double uniform_open_01(
    std::mt19937_64& engine
) {
    return
        (
            static_cast<double>(
                engine() >>
                11U
            ) +
            0.5
        ) /
        kTwoToThe53;
}

std::uint64_t bounded_uint64(
    std::mt19937_64& engine,
    std::uint64_t upper_exclusive
) {
    const std::uint64_t threshold =
        (
            std::uint64_t{0} -
            upper_exclusive
        ) %
        upper_exclusive;

    for (;;) {
        const std::uint64_t value =
            engine();

        if (value >= threshold) {
            return
                value %
                upper_exclusive;
        }
    }
}

int uniform_int(
    std::mt19937_64& engine,
    int lower,
    int upper
) {
    const std::int64_t lower_wide =
        static_cast<std::int64_t>(
            lower
        );

    const std::int64_t upper_wide =
        static_cast<std::int64_t>(
            upper
        );

    const std::uint64_t width =
        static_cast<std::uint64_t>(
            upper_wide -
            lower_wide
        ) +
        1U;

    const std::int64_t value =
        lower_wide +
        static_cast<std::int64_t>(
            bounded_uint64(
                engine,
                width
            )
        );

    return static_cast<int>(
        value
    );
}

std::size_t uniform_index(
    std::mt19937_64& engine,
    std::size_t size
) {
    return static_cast<std::size_t>(
        bounded_uint64(
            engine,
            static_cast<std::uint64_t>(
                size
            )
        )
    );
}

std::size_t weighted_index(
    std::mt19937_64& engine,
    const std::vector<double>& weights
) {
    double total =
        0.0;

    for (double weight :
         weights) {
        total +=
            weight;
    }

    const double target =
        uniform_01(
            engine
        ) *
        total;

    double cumulative =
        0.0;

    for (std::size_t index = 0U;
         index < weights.size();
         ++index) {
        cumulative +=
            weights[index];

        if (target < cumulative) {
            return index;
        }
    }

    return
        weights.size() -
        1U;
}

NormalGenerator::NormalGenerator(
    std::mt19937_64& engine
)
    : engine_(
          engine
      ) {}

double NormalGenerator::sample() {
    if (has_cached_) {
        has_cached_ =
            false;

        return cached_;
    }

    const double first =
        uniform_open_01(
            engine_
        );

    const double second =
        uniform_open_01(
            engine_
        );

    const double radius =
        std::sqrt(
            -2.0 *
            std::log(
                first
            )
        );

    const double angle =
        kTwoPi *
        second;

    cached_ =
        radius *
        std::sin(
            angle
        );

    has_cached_ =
        true;

    return
        radius *
        std::cos(
            angle
        );
}

}  // namespace vpd