#ifndef GRAPH_BENCHMARK_WASSERSTEIN_DISTANCE_HPP
#define GRAPH_BENCHMARK_WASSERSTEIN_DISTANCE_HPP

#include "../higher_order_persistence_diagrams.hpp"

#include <map>

namespace vpd {

struct Atom2PairKey {
    Atom2 first;
    Atom2 second;

    Atom2PairKey() = default;

    Atom2PairKey(
        const Atom2& first_value,
        const Atom2& second_value
    );

    bool operator<(
        const Atom2PairKey& other
    ) const;
};

class WassersteinMemo {
public:
    std::map<Atom2, double>
        atom2_diagonal_cost;

    std::map<Atom2PairKey, double>
        atom2_pair_cost;
};

double atom1_linf_distance(
    const Atom1& first,
    const Atom1& second
);

double atom1_diagonal_distance(
    const Atom1& atom
);

double atom1_basepoint_distance(
    const Atom1& first,
    const Atom1& second
);

double atom2_product_distance(
    const Atom2& first,
    const Atom2& second
);

double atom2_diagonal_distance(
    const Atom2& atom
);

double atom2_basepoint_distance(
    const Atom2& first,
    const Atom2& second
);

double atom2_diagonal_distance_memoized(
    const Atom2& atom,
    WassersteinMemo& memo
);

double atom2_basepoint_distance_memoized(
    const Atom2& first,
    const Atom2& second,
    WassersteinMemo& memo
);

double W1_PD1(
    const PD1& first,
    const PD1& second
);

double W1_PD2_direct(
    const PD2& first,
    const PD2& second
);

double W1_PD2_memoized(
    const PD2& first,
    const PD2& second,
    WassersteinMemo& memo
);

double W1_VPD2_to_zero_direct(
    const VPD2& diagram
);

double W1_VPD2_to_zero_memoized(
    const VPD2& diagram
);

double W1_VPD2_to_zero_memoized(
    const VPD2& diagram,
    WassersteinMemo& memo
);

}  // namespace vpd

#endif