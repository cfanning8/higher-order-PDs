from __future__ import annotations

import math
import re
from collections.abc import Hashable, Iterable, Sequence
from itertools import product
from typing import Any, Final, TypeVar

import numpy as np


INT32_MIN: Final[int] = -(2**31)
INT32_MAX: Final[int] = 2**31 - 1
UINT64_MAX: Final[int] = 2**64 - 1

DEFAULT_DIAGNOSTIC_LIMIT: Final[int] = 5

_CANONICAL_SIGNED_INTEGER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:0|-?[1-9][0-9]*)\Z"
)
_CANONICAL_UNSIGNED_INTEGER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:0|[1-9][0-9]*)\Z"
)

T = TypeVar(
    "T",
    bound=Hashable,
)


class ValidationError(ValueError):
    """Raised when serialized experimental data violate their declared schema."""


def _context_label(
    context: str | None,
) -> str:
    if context is None:
        return "data"

    stripped = context.strip()

    return (
        stripped
        if stripped
        else "data"
    )


def _diagnostic_sample(
    values: Sequence[Any],
) -> str:
    if len(values) <= DEFAULT_DIAGNOSTIC_LIMIT:
        return repr(
            tuple(
                values
            )
        )

    sample = tuple(
        values[
            :DEFAULT_DIAGNOSTIC_LIMIT
        ]
    )

    return (
        f"{sample!r} ... "
        f"({len(values)} total)"
    )


def _parse_integer(
    value: object,
    *,
    minimum: int,
    maximum: int,
    unsigned: bool,
    description: str,
) -> int:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        raise ValidationError(
            f"{description} must be an integer."
        )

    if isinstance(
        value,
        (int, np.integer),
    ):
        parsed = int(
            value
        )
    elif isinstance(
        value,
        str,
    ):
        pattern = (
            _CANONICAL_UNSIGNED_INTEGER_PATTERN
            if unsigned
            else _CANONICAL_SIGNED_INTEGER_PATTERN
        )

        if pattern.fullmatch(
            value
        ) is None:
            raise ValidationError(
                f"{description} has a noncanonical integer "
                f"representation {value!r}."
            )

        parsed = int(
            value,
            10,
        )
    else:
        raise ValidationError(
            f"{description} must be an integer."
        )

    if (
        parsed < minimum
        or parsed > maximum
    ):
        raise ValidationError(
            f"{description}={parsed} lies outside "
            f"[{minimum}, {maximum}]."
        )

    return parsed


def parse_int32(
    value: object,
    *,
    description: str,
) -> int:
    return _parse_integer(
        value,
        minimum=INT32_MIN,
        maximum=INT32_MAX,
        unsigned=False,
        description=description,
    )


def parse_uint64(
    value: object,
    *,
    description: str,
) -> int:
    return _parse_integer(
        value,
        minimum=0,
        maximum=UINT64_MAX,
        unsigned=True,
        description=description,
    )


def parse_finite_real(
    value: object,
    *,
    description: str,
) -> float:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        raise ValidationError(
            f"{description} must be numeric."
        )

    if isinstance(
        value,
        str,
    ):
        if value == "":
            raise ValidationError(
                f"{description} must not be empty."
            )

        try:
            parsed = float(
                value
            )
        except ValueError as error:
            raise ValidationError(
                f"{description}={value!r} is not numeric."
            ) from error

    elif isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        parsed = float(
            value
        )
    else:
        raise ValidationError(
            f"{description} must be numeric."
        )

    if not math.isfinite(
        parsed
    ):
        raise ValidationError(
            f"{description}={value!r} is not finite."
        )

    if parsed == 0.0:
        return 0.0

    return parsed


def require_exact_columns(
    actual_columns: Sequence[str],
    expected_columns: Sequence[str],
    *,
    context: str | None = None,
) -> None:
    actual = tuple(
        actual_columns
    )

    expected = tuple(
        expected_columns
    )

    if actual == expected:
        return

    label = _context_label(
        context
    )

    raise ValidationError(
        f"{label} has an incompatible column schema. "
        f"Expected {expected!r}, found {actual!r}."
    )


def require_schema_version(
    value: object,
    *,
    expected_version: int,
    context: str | None = None,
) -> None:
    label = _context_label(
        context
    )

    observed = parse_int32(
        value,
        description=(
            f"{label} schema version"
        ),
    )

    if observed != expected_version:
        raise ValidationError(
            f"{label} uses schema version {observed}; "
            f"expected {expected_version}."
        )


def require_unique_keys(
    keys: Iterable[T],
    *,
    context: str | None = None,
) -> None:
    label = _context_label(
        context
    )

    seen: set[T] = set()

    for key in keys:
        if key in seen:
            raise ValidationError(
                f"{label} contains duplicate key {key!r}."
            )

        seen.add(
            key
        )


def require_contiguous_indices(
    observed: Iterable[int],
    *,
    count: int,
    context: str | None = None,
) -> None:
    if count < 0:
        raise ValueError(
            "Expected index count must be nonnegative."
        )

    observed_set = {
        int(
            index
        )
        for index in observed
    }

    expected_set = set(
        range(
            count
        )
    )

    if observed_set == expected_set:
        return

    missing = tuple(
        sorted(
            expected_set
            - observed_set
        )
    )

    unexpected = tuple(
        sorted(
            observed_set
            - expected_set
        )
    )

    label = _context_label(
        context
    )

    raise ValidationError(
        f"{label} has an incomplete contiguous index domain. "
        f"Missing={_diagnostic_sample(missing)}, "
        f"unexpected={_diagnostic_sample(unexpected)}."
    )


def require_key_sets_equal(
    left_keys: Iterable[T],
    right_keys: Iterable[T],
    *,
    left_context: str,
    right_context: str,
) -> None:
    left = set(
        left_keys
    )

    right = set(
        right_keys
    )

    if left == right:
        return

    missing_from_left = tuple(
        sorted(
            right
            - left,
            key=repr,
        )
    )

    missing_from_right = tuple(
        sorted(
            left
            - right,
            key=repr,
        )
    )

    raise ValidationError(
        f"Key domains differ between {left_context} and "
        f"{right_context}. "
        f"Missing from {left_context}="
        f"{_diagnostic_sample(missing_from_left)}, "
        f"missing from {right_context}="
        f"{_diagnostic_sample(missing_from_right)}."
    )


def require_cartesian_domain(
    observed_keys: Iterable[
        tuple[Any, ...]
    ],
    axes: Sequence[
        Iterable[Any]
    ],
    *,
    context: str | None = None,
) -> None:
    expected = set(
        product(
            *(
                tuple(
                    axis
                )
                for axis in axes
            )
        )
    )

    observed = set(
        observed_keys
    )

    if observed == expected:
        return

    missing = tuple(
        sorted(
            expected
            - observed,
            key=repr,
        )
    )

    unexpected = tuple(
        sorted(
            observed
            - expected,
            key=repr,
        )
    )

    label = _context_label(
        context
    )

    raise ValidationError(
        f"{label} does not match its declared Cartesian domain. "
        f"Missing={_diagnostic_sample(missing)}, "
        f"unexpected={_diagnostic_sample(unexpected)}."
    )


__all__ = [
    "INT32_MAX",
    "INT32_MIN",
    "UINT64_MAX",
    "ValidationError",
    "parse_finite_real",
    "parse_int32",
    "parse_uint64",
    "require_cartesian_domain",
    "require_contiguous_indices",
    "require_exact_columns",
    "require_key_sets_equal",
    "require_schema_version",
    "require_unique_keys",
]