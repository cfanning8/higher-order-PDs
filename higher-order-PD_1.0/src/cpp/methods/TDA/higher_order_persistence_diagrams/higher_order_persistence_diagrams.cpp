#include "higher_order_persistence_diagrams.hpp"

#include <tuple>

namespace vpd {

Atom2::Atom2(
    const Atom1& left_value,
    const Atom1& right_value
)
    : left(
          left_value
      ),
      right(
          right_value
      ) {
}

bool Atom2::operator<(
    const Atom2& other
) const {
    return
        std::tie(
            left,
            right
        ) <
        std::tie(
            other.left,
            other.right
        );
}

bool Atom2::operator==(
    const Atom2& other
) const {
    return
        left ==
            other.left &&
        right ==
            other.right;
}

bool preorder_atom2(
    const Atom2& first,
    const Atom2& second
) {
    return
        preorder_atom1(
            second.left,
            first.left
        ) &&
        preorder_atom1(
            first.right,
            second.right
        );
}

bool equivalent_atom2(
    const Atom2& first,
    const Atom2& second
) {
    return
        preorder_atom2(
            first,
            second
        ) &&
        preorder_atom2(
            second,
            first
        );
}

bool is_zero_class(
    const Atom2& atom
) {
    return
        equivalent_atom1(
            atom.left,
            atom.right
        );
}

std::ostream& operator<<(
    std::ostream& stream,
    const Atom2& atom
) {
    stream
        << "("
        << atom.left
        << ","
        << atom.right
        << ")";

    return stream;
}

}  // namespace vpd