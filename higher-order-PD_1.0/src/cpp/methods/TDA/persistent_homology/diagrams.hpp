#ifndef GRAPH_BENCHMARK_DIAGRAMS_HPP
#define GRAPH_BENCHMARK_DIAGRAMS_HPP

#include <cmath>
#include <cstddef>
#include <map>
#include <ostream>
#include <vector>

namespace vpd {

inline constexpr double
    kCoefficientTolerance = 1.0e-12;

struct Atom1 {
    double birth{};
    double death{};

    Atom1() = default;

    Atom1(
        double birth_value,
        double death_value
    );

    bool operator<(
        const Atom1& other
    ) const;

    bool operator==(
        const Atom1& other
    ) const;
};

template <class Atom>
using Diagram =
    std::map<Atom, double>;

using PD1 =
    Diagram<Atom1>;

using VPD1 =
    Diagram<Atom1>;

double coefficient_tolerance();

bool preorder_base(
    double first,
    double second
);

bool preorder_atom1(
    const Atom1& first,
    const Atom1& second
);

bool equivalent_atom1(
    const Atom1& first,
    const Atom1& second
);

bool is_zero_class(
    const Atom1& atom
);

double persistence_length(
    const Atom1& atom
);

VPD1 prune_low_persistence(
    const VPD1& diagram,
    double minimum_persistence
);

template <class Atom>
void add_to_diagram(
    Diagram<Atom>& diagram,
    const Atom& atom,
    double coefficient
) {
    if (
        std::abs(
            coefficient
        ) <=
        kCoefficientTolerance
    ) {
        return;
    }

    if (is_zero_class(
            atom
        )) {
        return;
    }

    auto [
        iterator,
        inserted
    ] =
        diagram.try_emplace(
            atom,
            0.0
        );

    (void)inserted;

    iterator->second +=
        coefficient;

    if (
        std::abs(
            iterator->second
        ) <=
        kCoefficientTolerance
    ) {
        diagram.erase(
            iterator
        );
    }
}

template <class Atom>
Diagram<Atom> add_diagrams(
    const Diagram<Atom>& first,
    const Diagram<Atom>& second
) {
    Diagram<Atom> result =
        first;

    for (const auto& [
             atom,
             coefficient
         ] : second) {
        add_to_diagram(
            result,
            atom,
            coefficient
        );
    }

    return result;
}

template <class Atom>
Diagram<Atom> subtract_diagrams(
    const Diagram<Atom>& first,
    const Diagram<Atom>& second
) {
    Diagram<Atom> result =
        first;

    for (const auto& [
             atom,
             coefficient
         ] : second) {
        add_to_diagram(
            result,
            atom,
            -coefficient
        );
    }

    return result;
}

template <class Atom>
Diagram<Atom> scale_diagram(
    const Diagram<Atom>& diagram,
    double scalar
) {
    Diagram<Atom> result;

    for (const auto& [
             atom,
             coefficient
         ] : diagram) {
        add_to_diagram(
            result,
            atom,
            scalar *
                coefficient
        );
    }

    return result;
}

template <class Atom>
void clean_diagram(
    Diagram<Atom>& diagram
) {
    std::vector<Atom>
        atoms_to_remove;

    atoms_to_remove.reserve(
        diagram.size()
    );

    for (const auto& [
             atom,
             coefficient
         ] : diagram) {
        if (
            std::abs(
                coefficient
            ) <=
                kCoefficientTolerance ||
            is_zero_class(
                atom
            )
        ) {
            atoms_to_remove.push_back(
                atom
            );
        }
    }

    for (const Atom& atom :
         atoms_to_remove) {
        diagram.erase(
            atom
        );
    }
}

template <class Atom>
std::size_t support_size(
    const Diagram<Atom>& diagram
) {
    return diagram.size();
}

template <class Atom>
double total_mass(
    const Diagram<Atom>& diagram
) {
    double total =
        0.0;

    for (const auto& [
             atom,
             coefficient
         ] : diagram) {
        (void)atom;

        total +=
            coefficient;
    }

    return total;
}

template <class Atom>
double total_abs_mass(
    const Diagram<Atom>& diagram
) {
    double total =
        0.0;

    for (const auto& [
             atom,
             coefficient
         ] : diagram) {
        (void)atom;

        total +=
            std::abs(
                coefficient
            );
    }

    return total;
}

template <class Atom>
bool is_zero_diagram(
    const Diagram<Atom>& diagram
) {
    return diagram.empty();
}

std::ostream& operator<<(
    std::ostream& stream,
    const Atom1& atom
);

}  // namespace vpd

#endif