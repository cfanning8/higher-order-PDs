#ifndef GRAPH_BENCHMARK_HIGHER_ORDER_PERSISTENCE_DIAGRAMS_HPP
#define GRAPH_BENCHMARK_HIGHER_ORDER_PERSISTENCE_DIAGRAMS_HPP

#include "../persistent_homology/diagrams.hpp"

#include <ostream>

namespace vpd {

struct Atom2 {
    Atom1 left;
    Atom1 right;

    Atom2() = default;

    Atom2(
        const Atom1& left_value,
        const Atom1& right_value
    );

    bool operator<(
        const Atom2& other
    ) const;

    bool operator==(
        const Atom2& other
    ) const;
};

using PD2 =
    Diagram<Atom2>;

using VPD2 =
    Diagram<Atom2>;

bool preorder_atom2(
    const Atom2& first,
    const Atom2& second
);

bool equivalent_atom2(
    const Atom2& first,
    const Atom2& second
);

bool is_zero_class(
    const Atom2& atom
);

std::ostream& operator<<(
    std::ostream& stream,
    const Atom2& atom
);

}  // namespace vpd

#endif