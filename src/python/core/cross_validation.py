from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import StratifiedKFold


IndexArray: TypeAlias = NDArray[np.int64]
LabelArray: TypeAlias = NDArray[np.int64]
StratumArray: TypeAlias = NDArray[np.int64]

CandidateT = TypeVar("CandidateT")

CandidateEvaluator: TypeAlias = Callable[
    [CandidateT, IndexArray, IndexArray],
    float,
]

_UINT32_MASK = (1 << 32) - 1


@dataclass(
    frozen=True,
    slots=True,
)
class InnerFold:
    outer_fold_index: int
    inner_fold_index: int
    train_indices: IndexArray
    validation_indices: IndexArray


@dataclass(
    frozen=True,
    slots=True,
)
class OuterFold:
    fold_index: int
    train_indices: IndexArray
    test_indices: IndexArray
    inner_folds: tuple[InnerFold, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class NestedCrossValidationPlan:
    sample_count: int
    outer_fold_count: int
    inner_fold_count: int
    outer_seed: int
    inner_seed: int
    class_labels: tuple[int, ...]
    class_counts: tuple[int, ...]
    stratum_labels: tuple[int, ...]
    stratum_counts: tuple[int, ...]
    folds: tuple[OuterFold, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class CandidateScore(
    Generic[CandidateT]
):
    candidate: CandidateT
    fold_scores: tuple[float, ...]
    mean_score: float = field(
        init=False
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "mean_score",
            (
                math.fsum(
                    self.fold_scores
                )
                / len(
                    self.fold_scores
                )
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class CandidateSelectionResult(
    Generic[CandidateT]
):
    selected_index: int
    scores: tuple[
        CandidateScore[CandidateT],
        ...,
    ]
    selected_candidate: CandidateT = field(
        init=False
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "selected_candidate",
            self.scores[
                self.selected_index
            ].candidate,
        )


def _labels(
    values: ArrayLike,
) -> LabelArray:
    result = np.asarray(
        values,
        dtype=np.int64,
    )

    if result.ndim != 1:
        raise ValueError(
            "Classification labels must be one-dimensional."
        )

    if result.size == 0:
        raise ValueError(
            "Classification labels must not be empty."
        )

    return result


def _strata(
    values: ArrayLike,
    *,
    sample_count: int,
) -> tuple[
    StratumArray,
    tuple[int, ...],
]:
    domain = np.asarray(
        values,
        dtype=np.int64,
    )

    if domain.ndim == 1:
        if domain.shape[0] != sample_count:
            raise ValueError(
                "Stratification-domain size does not match the "
                "sample count."
            )

        encoded = np.unique(
            domain,
            return_inverse=True,
        )[1]

    elif domain.ndim == 2:
        if domain.shape[0] != sample_count:
            raise ValueError(
                "Stratification-domain size does not match the "
                "sample count."
            )

        encoded = np.unique(
            domain,
            axis=0,
            return_inverse=True,
        )[1]

    else:
        raise ValueError(
            "Stratification domain must be one- or two-dimensional."
        )

    strata = np.asarray(
        encoded,
        dtype=np.int64,
    )

    counts = np.bincount(
        strata
    )

    return (
        strata,
        tuple(
            int(
                count
            )
            for count in counts
        ),
    )


def _random_state(
    seed: int,
    index: int = 0,
) -> int:
    return int(
        (
            int(
                seed
            )
            +
            int(
                index
            )
        )
        & _UINT32_MASK
    )


def make_nested_stratified_plan(
    labels: ArrayLike,
    stratification_domain: ArrayLike,
    *,
    outer_folds: int,
    inner_folds: int,
    outer_seed: int,
    inner_seed: int,
) -> NestedCrossValidationPlan:
    label_array = _labels(
        labels
    )

    sample_count = int(
        label_array.size
    )

    strata, stratum_counts = _strata(
        stratification_domain,
        sample_count=sample_count,
    )

    if min(
        stratum_counts
    ) < outer_folds:
        raise ValueError(
            "The smallest stratification stratum does not contain "
            "enough observations for the requested outer folds."
        )

    class_labels_array, class_counts_array = np.unique(
        label_array,
        return_counts=True,
    )

    all_indices = np.arange(
        sample_count,
        dtype=np.int64,
    )

    outer_splitter = StratifiedKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=_random_state(
            outer_seed
        ),
    )

    fold_results: list[
        OuterFold
    ] = []

    for outer_fold_index, (
        outer_train_local,
        outer_test_local,
    ) in enumerate(
        outer_splitter.split(
            all_indices,
            strata,
        )
    ):
        outer_train_indices = np.asarray(
            all_indices[
                outer_train_local
            ],
            dtype=np.int64,
        )

        outer_test_indices = np.asarray(
            all_indices[
                outer_test_local
            ],
            dtype=np.int64,
        )

        outer_train_strata = strata[
            outer_train_indices
        ]

        inner_stratum_counts = np.bincount(
            outer_train_strata,
            minlength=len(
                stratum_counts
            ),
        )

        if int(
            np.min(
                inner_stratum_counts
            )
        ) < inner_folds:
            raise ValueError(
                f"Outer fold {outer_fold_index} does not contain "
                "enough observations in every stratification "
                "stratum for the requested inner folds."
            )

        inner_splitter = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=_random_state(
                inner_seed,
                outer_fold_index,
            ),
        )

        inner_folds_result: list[
            InnerFold
        ] = []

        outer_train_positions = np.arange(
            outer_train_indices.size,
            dtype=np.int64,
        )

        for inner_fold_index, (
            inner_train_local,
            inner_validation_local,
        ) in enumerate(
            inner_splitter.split(
                outer_train_positions,
                outer_train_strata,
            )
        ):
            inner_folds_result.append(
                InnerFold(
                    outer_fold_index=outer_fold_index,
                    inner_fold_index=inner_fold_index,
                    train_indices=np.asarray(
                        outer_train_indices[
                            inner_train_local
                        ],
                        dtype=np.int64,
                    ),
                    validation_indices=np.asarray(
                        outer_train_indices[
                            inner_validation_local
                        ],
                        dtype=np.int64,
                    ),
                )
            )

        fold_results.append(
            OuterFold(
                fold_index=outer_fold_index,
                train_indices=outer_train_indices,
                test_indices=outer_test_indices,
                inner_folds=tuple(
                    inner_folds_result
                ),
            )
        )

    return NestedCrossValidationPlan(
        sample_count=sample_count,
        outer_fold_count=outer_folds,
        inner_fold_count=inner_folds,
        outer_seed=int(
            outer_seed
        ),
        inner_seed=int(
            inner_seed
        ),
        class_labels=tuple(
            int(
                value
            )
            for value in class_labels_array
        ),
        class_counts=tuple(
            int(
                value
            )
            for value in class_counts_array
        ),
        stratum_labels=tuple(
            range(
                len(
                    stratum_counts
                )
            )
        ),
        stratum_counts=stratum_counts,
        folds=tuple(
            fold_results
        ),
    )


def validate_nested_cross_validation_plan(
    plan: NestedCrossValidationPlan,
    labels: ArrayLike,
    stratification_domain: ArrayLike,
) -> None:
    label_array = _labels(
        labels
    )

    if label_array.size != plan.sample_count:
        raise ValueError(
            "Cross-validation plan and labels have different sample "
            "counts."
        )

    strata, stratum_counts = _strata(
        stratification_domain,
        sample_count=plan.sample_count,
    )

    classes, class_counts = np.unique(
        label_array,
        return_counts=True,
    )

    if tuple(
        int(
            value
        )
        for value in classes
    ) != plan.class_labels:
        raise ValueError(
            "Cross-validation plan uses different classification "
            "labels."
        )

    if tuple(
        int(
            value
        )
        for value in class_counts
    ) != plan.class_counts:
        raise ValueError(
            "Cross-validation plan uses different class counts."
        )

    if stratum_counts != plan.stratum_counts:
        raise ValueError(
            "Cross-validation plan uses different stratification "
            "strata."
        )

    test_counts = np.zeros(
        plan.sample_count,
        dtype=np.int64,
    )

    universe = np.arange(
        plan.sample_count,
        dtype=np.int64,
    )

    for outer_fold in plan.folds:
        if np.intersect1d(
            outer_fold.train_indices,
            outer_fold.test_indices,
        ).size:
            raise ValueError(
                "Outer training and test sets overlap."
            )

        if not np.array_equal(
            np.sort(
                np.concatenate(
                    (
                        outer_fold.train_indices,
                        outer_fold.test_indices,
                    )
                )
            ),
            universe,
        ):
            raise ValueError(
                "An outer train/test split does not partition the "
                "sample domain."
            )

        test_counts[
            outer_fold.test_indices
        ] += 1

        inner_validation_counts = np.zeros(
            plan.sample_count,
            dtype=np.int64,
        )

        for inner_fold in outer_fold.inner_folds:
            if np.intersect1d(
                inner_fold.train_indices,
                inner_fold.validation_indices,
            ).size:
                raise ValueError(
                    "Inner training and validation sets overlap."
                )

            if not np.array_equal(
                np.sort(
                    np.concatenate(
                        (
                            inner_fold.train_indices,
                            inner_fold.validation_indices,
                        )
                    )
                ),
                np.sort(
                    outer_fold.train_indices
                ),
            ):
                raise ValueError(
                    "An inner train/validation split does not "
                    "partition its outer-training domain."
                )

            inner_validation_counts[
                inner_fold.validation_indices
            ] += 1

        if not np.all(
            inner_validation_counts[
                outer_fold.train_indices
            ]
            ==
            1
        ):
            raise ValueError(
                "Inner validation folds do not partition the "
                "outer-training observations exactly once."
            )

    if not np.all(
        test_counts == 1
    ):
        raise ValueError(
            "Outer test folds do not partition the observations "
            "exactly once."
        )

    del strata


def outer_test_fold_by_sample(
    plan: NestedCrossValidationPlan,
) -> IndexArray:
    result = np.empty(
        plan.sample_count,
        dtype=np.int64,
    )

    for fold in plan.folds:
        result[
            fold.test_indices
        ] = fold.fold_index

    return result


def select_candidate(
    candidates: Sequence[CandidateT],
    outer_fold: OuterFold,
    evaluate: CandidateEvaluator[CandidateT],
    *,
    maximize: bool = True,
) -> CandidateSelectionResult[CandidateT]:
    if not candidates:
        raise ValueError(
            "Candidate selection requires at least one candidate."
        )

    candidate_scores: list[
        CandidateScore[CandidateT]
    ] = []

    for candidate in candidates:
        fold_scores = tuple(
            float(
                evaluate(
                    candidate,
                    inner_fold.train_indices,
                    inner_fold.validation_indices,
                )
            )
            for inner_fold in outer_fold.inner_folds
        )

        candidate_scores.append(
            CandidateScore(
                candidate=candidate,
                fold_scores=fold_scores,
            )
        )

    selected_index = 0
    selected_score = candidate_scores[
        0
    ].mean_score

    for candidate_index in range(
        1,
        len(
            candidate_scores
        ),
    ):
        score = candidate_scores[
            candidate_index
        ].mean_score

        if (
            score > selected_score
            if maximize
            else score < selected_score
        ):
            selected_index = candidate_index
            selected_score = score

    return CandidateSelectionResult(
        selected_index=selected_index,
        scores=tuple(
            candidate_scores
        ),
    )