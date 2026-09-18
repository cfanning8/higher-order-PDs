#ifndef GRAPH_BENCHMARK_HARMONIC_AGGREGATION_HPP
#define GRAPH_BENCHMARK_HARMONIC_AGGREGATION_HPP

#include "../../persistent_homology/diagrams.hpp"
#include "random_fourier_features.hpp"

#include <complex>
#include <vector>

namespace vpd {

using HarmonicValues =
    std::vector<std::complex<double>>;

std::complex<double> pair_fourier(
    const VPD1& diagram,
    const CharacterSpec& character
);

HarmonicValues evaluate_pair_fourier(
    const VPD1& diagram,
    const CharacterBank& characters
);

HarmonicValues mean_pair_fourier(
    const std::vector<VPD1>& diagrams,
    const CharacterBank& characters
);

}  // namespace vpd

#endif