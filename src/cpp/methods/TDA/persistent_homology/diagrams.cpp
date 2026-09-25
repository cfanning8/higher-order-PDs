#include "diagrams.hpp"

#include <iomanip>
#include <tuple>

namespace vpd {

Atom1::Atom1(
    double birth_value,
    double death_value
)
    : birth(
          birth_value
      ),
      death(
          death_value
      ) {
}

bool Atom1::operator<(
    const Atom1& other
) const {
    return
        std::tie(
            birth,
            death
        ) <
        std::tie(
            other.birth,
            other.death
        );
}

bool Atom1::operator==(
    const Atom1& other
) const {
    return
        birth ==
            other.birth &&
        death ==
            other.death;
}

double coefficient_tolerance() {
    return kCoefficientTolerance;
}

bool preorder_base(
    double first,
    double second
) {
    return first <= second;
}

bool preorder_atom1(
    const Atom1& first,
    const Atom1& second
) {
    return
        second.birth <=
            first.birth &&
        first.death <=
            second.death;
}

bool equivalent_atom1(
    const Atom1& first,
    const Atom1& second
) {
    return
        preorder_atom1(
            first,
            second
        ) &&
        preorder_atom1(
            second,
            first
        );
}

bool is_zero_class(
    const Atom1& atom
) {
    return
        atom.birth ==
        atom.death;
}

double persistence_length(
    const Atom1& atom
) {
    return
        atom.death -
        atom.birth;
}

VPD1 prune_low_persistence(
    const VPD1& diagram,
    double minimum_persistence
) {
    VPD1 pruned;

    for (const auto& [
             atom,
             coefficient
         ] : diagram) {
        if (
            persistence_length(
                atom
            ) >=
            minimum_persistence
        ) {
            add_to_diagram(
                pruned,
                atom,
                coefficient
            );
        }
    }

    return pruned;
}

std::ostream& operator<<(
    std::ostream& stream,
    const Atom1& atom
) {
    stream
        << "("
        << std::setprecision(
               17
           )
        << atom.birth
        << ","
        << atom.death
        << ")";

    return stream;
}

}  // namespace vpd