from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable


_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9_-]*$"
)

_WINDOWS_RESERVED_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)

_WINDOWS_FORBIDDEN_FILENAME_CHARACTERS: Final[frozenset[str]] = (
    frozenset('<>:"/\\|?*')
)

_RANDOM_GRAPH_CLASSIFICATION_EXPERIMENT: Final[str] = (
    "random_graph_classification"
)


def _require_path(
    value: str | os.PathLike[str] | Path,
    *,
    description: str,
) -> Path:
    if not isinstance(description, str) or not description:
        raise ValueError(
            "A path description must be a nonempty string."
        )

    if isinstance(value, Path):
        path = value
    elif isinstance(value, (str, os.PathLike)):
        raw_value = os.fspath(value)

        if not isinstance(raw_value, str):
            raise TypeError(
                f"{description} must resolve to a string path."
            )

        if not raw_value.strip():
            raise ValueError(
                f"{description} must not be empty."
            )

        path = Path(raw_value)
    else:
        raise TypeError(
            f"{description} must be path-like."
        )

    return path


def _normalize_path(
    value: str | os.PathLike[str] | Path,
    *,
    description: str,
) -> Path:
    path = _require_path(
        value,
        description=description,
    )

    try:
        return path.expanduser().resolve(
            strict=False
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Could not resolve {description} '{path}'."
        ) from error


def _path_identity(
    path: Path,
) -> str:
    normalized = _normalize_path(
        path,
        description="filesystem path",
    )

    return os.path.normcase(
        os.fspath(
            normalized
        )
    )


def _same_path(
    first: Path,
    second: Path,
) -> bool:
    return (
        _path_identity(
            first
        )
        ==
        _path_identity(
            second
        )
    )


def _is_within(
    candidate: Path,
    parent: Path,
    *,
    allow_equal: bool = True,
) -> bool:
    normalized_candidate = _normalize_path(
        candidate,
        description="candidate path",
    )

    normalized_parent = _normalize_path(
        parent,
        description="parent path",
    )

    candidate_identity = _path_identity(
        normalized_candidate
    )

    parent_identity = _path_identity(
        normalized_parent
    )

    if candidate_identity == parent_identity:
        return allow_equal

    try:
        relative = normalized_candidate.relative_to(
            normalized_parent
        )
    except ValueError:
        if os.name != "nt":
            return False

        candidate_parts = tuple(
            os.path.normcase(
                part
            )
            for part in normalized_candidate.parts
        )

        parent_parts = tuple(
            os.path.normcase(
                part
            )
            for part in normalized_parent.parts
        )

        return (
            len(
                candidate_parts
            )
            >
            len(
                parent_parts
            )
            and candidate_parts[
                :len(
                    parent_parts
                )
            ]
            ==
            parent_parts
        )

    return relative != Path(".")


def _paths_overlap(
    first: Path,
    second: Path,
) -> bool:
    return (
        _is_within(
            first,
            second,
            allow_equal=True,
        )
        or _is_within(
            second,
            first,
            allow_equal=True,
        )
    )


def _require_nonoverlapping(
    first: Path,
    first_description: str,
    second: Path,
    second_description: str,
) -> None:
    if _paths_overlap(
        first,
        second,
    ):
        raise ValueError(
            f"{first_description} '{first}' overlaps "
            f"{second_description} '{second}'."
        )


def _contains_control_character(
    value: str,
) -> bool:
    return any(
        ord(
            character
        )
        <
        32
        for character in value
    )


def _has_windows_reserved_basename(
    value: str,
) -> bool:
    basename = value.split(
        ".",
        1,
    )[0].upper()

    return (
        basename
        in
        _WINDOWS_RESERVED_BASENAMES
    )


def _validate_component_common(
    value: str,
    *,
    description: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{description} must be a string."
        )

    if not value:
        raise ValueError(
            f"{description} must not be empty."
        )

    if value in {".", ".."}:
        raise ValueError(
            f"{description} must not be '.' or '..'."
        )

    if "\x00" in value:
        raise ValueError(
            f"{description} contains a NUL character."
        )

    if _contains_control_character(
        value
    ):
        raise ValueError(
            f"{description} contains a control character."
        )

    if "/" in value or "\\" in value:
        raise ValueError(
            f"{description} must contain exactly one "
            "filesystem component."
        )

    if value.endswith(
        (
            " ",
            ".",
        )
    ):
        raise ValueError(
            f"{description} must not end with a space "
            "or period."
        )

    candidate = Path(
        value
    )

    if (
        candidate.is_absolute()
        or candidate.anchor
        or candidate.drive
        or candidate.root
        or len(
            candidate.parts
        )
        !=
        1
    ):
        raise ValueError(
            f"{description} must be a relative single "
            "filesystem component."
        )

    if _has_windows_reserved_basename(
        value
    ):
        raise ValueError(
            f"{description} uses a reserved Windows "
            f"device name: '{value}'."
        )

    return value


def validate_identifier(
    value: str,
    *,
    description: str = "identifier",
) -> str:
    validated = _validate_component_common(
        value,
        description=description,
    )

    if _IDENTIFIER_PATTERN.fullmatch(
        validated
    ) is None:
        raise ValueError(
            f"{description} '{validated}' must match "
            "[a-z0-9][a-z0-9_-]*."
        )

    return validated


def validate_filename(
    value: str,
    *,
    description: str = "filename",
) -> str:
    validated = _validate_component_common(
        value,
        description=description,
    )

    forbidden = sorted(
        character
        for character in validated
        if character
        in
        _WINDOWS_FORBIDDEN_FILENAME_CHARACTERS
    )

    if forbidden:
        rendered = "".join(
            sorted(
                set(
                    forbidden
                )
            )
        )

        raise ValueError(
            f"{description} '{validated}' contains "
            "characters forbidden in Windows filenames: "
            f"'{rendered}'."
        )

    return validated


def _safe_child(
    owner: Path,
    component: str,
    *,
    component_description: str,
) -> Path:
    validated = validate_identifier(
        component,
        description=component_description,
    )

    candidate = _normalize_path(
        owner
        /
        validated,
        description=component_description,
    )

    if not _is_within(
        candidate,
        owner,
        allow_equal=False,
    ):
        raise ValueError(
            f"{component_description} '{component}' "
            f"escapes owner directory '{owner}'."
        )

    return candidate


def _safe_file(
    owner: Path,
    filename: str,
    *,
    filename_description: str,
) -> Path:
    validated = validate_filename(
        filename,
        description=filename_description,
    )

    candidate = _normalize_path(
        owner
        /
        validated,
        description=filename_description,
    )

    if not _is_within(
        candidate,
        owner,
        allow_equal=False,
    ):
        raise ValueError(
            f"{filename_description} '{filename}' "
            f"escapes owner directory '{owner}'."
        )

    return candidate


def _require_directory_if_present(
    path: Path,
    *,
    description: str,
) -> None:
    if (
        path.exists()
        and not path.is_dir()
    ):
        raise RuntimeError(
            f"{description} exists but is not a directory: "
            f"'{path}'."
        )


@dataclass(
    frozen=True,
    slots=True,
)
class ProjectPaths:
    python_root: Path
    src_root: Path
    cpp_root: Path
    cpp_raw_root: Path

    results_root: Path
    cache_root: Path
    processed_root: Path
    figures_root: Path
    tables_root: Path

    def __post_init__(
        self,
    ) -> None:
        normalized_python_root = _normalize_path(
            self.python_root,
            description="Python source root",
        )

        normalized_src_root = _normalize_path(
            self.src_root,
            description="source root",
        )

        normalized_cpp_root = _normalize_path(
            self.cpp_root,
            description="C++ source root",
        )

        normalized_cpp_raw_root = _normalize_path(
            self.cpp_raw_root,
            description="C++ raw-data root",
        )

        normalized_results_root = _normalize_path(
            self.results_root,
            description="Python results root",
        )

        normalized_cache_root = _normalize_path(
            self.cache_root,
            description="Python cache root",
        )

        normalized_processed_root = _normalize_path(
            self.processed_root,
            description="Python processed-data root",
        )

        normalized_figures_root = _normalize_path(
            self.figures_root,
            description="Python figures root",
        )

        normalized_tables_root = _normalize_path(
            self.tables_root,
            description="Python tables root",
        )

        object.__setattr__(
            self,
            "python_root",
            normalized_python_root,
        )

        object.__setattr__(
            self,
            "src_root",
            normalized_src_root,
        )

        object.__setattr__(
            self,
            "cpp_root",
            normalized_cpp_root,
        )

        object.__setattr__(
            self,
            "cpp_raw_root",
            normalized_cpp_raw_root,
        )

        object.__setattr__(
            self,
            "results_root",
            normalized_results_root,
        )

        object.__setattr__(
            self,
            "cache_root",
            normalized_cache_root,
        )

        object.__setattr__(
            self,
            "processed_root",
            normalized_processed_root,
        )

        object.__setattr__(
            self,
            "figures_root",
            normalized_figures_root,
        )

        object.__setattr__(
            self,
            "tables_root",
            normalized_tables_root,
        )

        self._validate_repository_relationships()
        self._validate_output_relationships()

    @classmethod
    def discover(
        cls,
        *,
        results_root: (
            str
            | os.PathLike[str]
            | Path
            | None
        ) = None,
        cpp_raw_root: (
            str
            | os.PathLike[str]
            | Path
            | None
        ) = None,
    ) -> ProjectPaths:
        module_path = _normalize_path(
            Path(
                __file__
            ),
            description="paths module",
        )

        core_root = module_path.parent
        python_root = core_root.parent

        if python_root.name != "python":
            raise RuntimeError(
                "Repository discovery failed because "
                f"the Python source root resolved to "
                f"'{python_root}', whose final component "
                "is not 'python'."
            )

        src_root = python_root.parent

        if src_root.name != "src":
            raise RuntimeError(
                "Repository discovery failed because "
                f"the source root resolved to "
                f"'{src_root}', whose final component "
                "is not 'src'."
            )

        cpp_root = _normalize_path(
            src_root
            /
            "cpp",
            description="C++ source root",
        )

        default_cpp_raw_root = _normalize_path(
            cpp_root
            /
            "results"
            /
            "raw",
            description="C++ raw-data root",
        )

        resolved_cpp_raw_root = (
            default_cpp_raw_root
            if cpp_raw_root is None
            else _normalize_path(
                cpp_raw_root,
                description="C++ raw-data root",
            )
        )

        default_results_root = _normalize_path(
            python_root
            /
            "results",
            description="Python results root",
        )

        resolved_results_root = (
            default_results_root
            if results_root is None
            else _normalize_path(
                results_root,
                description="Python results root",
            )
        )

        return cls(
            python_root=python_root,
            src_root=src_root,
            cpp_root=cpp_root,
            cpp_raw_root=resolved_cpp_raw_root,
            results_root=resolved_results_root,
            cache_root=(
                resolved_results_root
                /
                "cache"
            ),
            processed_root=(
                resolved_results_root
                /
                "processed"
            ),
            figures_root=(
                resolved_results_root
                /
                "figures"
            ),
            tables_root=(
                resolved_results_root
                /
                "tables"
            ),
        )

    def _validate_repository_relationships(
        self,
    ) -> None:
        expected_src_root = _normalize_path(
            self.python_root.parent,
            description="expected source root",
        )

        if not _same_path(
            self.src_root,
            expected_src_root,
        ):
            raise ValueError(
                "Python source root and source root are "
                "inconsistent: "
                f"python_root='{self.python_root}', "
                f"src_root='{self.src_root}'."
            )

        expected_cpp_root = _normalize_path(
            self.src_root
            /
            "cpp",
            description="expected C++ source root",
        )

        if not _same_path(
            self.cpp_root,
            expected_cpp_root,
        ):
            raise ValueError(
                "C++ source root is inconsistent with "
                "the source tree: "
                f"cpp_root='{self.cpp_root}', "
                f"expected='{expected_cpp_root}'."
            )

        if not _is_within(
            self.cpp_raw_root,
            self.src_root,
            allow_equal=False,
        ):
            raise ValueError(
                "C++ raw-data root must lie below the "
                "source root: "
                f"cpp_raw_root='{self.cpp_raw_root}', "
                f"src_root='{self.src_root}'."
            )

        if _is_within(
            self.cpp_raw_root,
            self.python_root,
            allow_equal=True,
        ):
            raise ValueError(
                "C++ raw-data root must not lie inside the "
                "Python source tree: "
                f"cpp_raw_root='{self.cpp_raw_root}', "
                f"python_root='{self.python_root}'."
            )

    def _validate_output_relationships(
        self,
    ) -> None:
        expected_children = {
            "Python cache root": (
                self.cache_root,
                self.results_root
                /
                "cache",
            ),
            "Python processed-data root": (
                self.processed_root,
                self.results_root
                /
                "processed",
            ),
            "Python figures root": (
                self.figures_root,
                self.results_root
                /
                "figures",
            ),
            "Python tables root": (
                self.tables_root,
                self.results_root
                /
                "tables",
            ),
        }

        for description, (
            actual,
            expected,
        ) in expected_children.items():
            normalized_expected = _normalize_path(
                expected,
                description=(
                    f"expected {description}"
                ),
            )

            if not _same_path(
                actual,
                normalized_expected,
            ):
                raise ValueError(
                    f"{description} must be the fixed "
                    "child of the Python results root: "
                    f"actual='{actual}', "
                    f"expected='{normalized_expected}'."
                )

        writable_roots = (
            (
                "Python cache root",
                self.cache_root,
            ),
            (
                "Python processed-data root",
                self.processed_root,
            ),
            (
                "Python figures root",
                self.figures_root,
            ),
            (
                "Python tables root",
                self.tables_root,
            ),
        )

        for description, root in writable_roots:
            if not _is_within(
                root,
                self.results_root,
                allow_equal=False,
            ):
                raise ValueError(
                    f"{description} '{root}' must lie "
                    "strictly below Python results root "
                    f"'{self.results_root}'."
                )

            _require_nonoverlapping(
                root,
                description,
                self.cpp_raw_root,
                "immutable C++ raw-data root",
            )

        _require_nonoverlapping(
            self.results_root,
            "Python results root",
            self.cpp_raw_root,
            "immutable C++ raw-data root",
        )

        for first_index in range(
            len(
                writable_roots
            )
        ):
            (
                first_description,
                first_root,
            ) = writable_roots[
                first_index
            ]

            for second_index in range(
                first_index
                +
                1,
                len(
                    writable_roots
                ),
            ):
                (
                    second_description,
                    second_root,
                ) = writable_roots[
                    second_index
                ]

                _require_nonoverlapping(
                    first_root,
                    first_description,
                    second_root,
                    second_description,
                )

        protected_python_roots = (
            (
                "Python core source directory",
                self.python_root
                /
                "core",
            ),
            (
                "Python methods source directory",
                self.python_root
                /
                "methods",
            ),
            (
                "Python experiments source directory",
                self.python_root
                /
                "experiments",
            ),
            (
                "Python tests source directory",
                self.python_root
                /
                "tests",
            ),
        )

        for (
            protected_description,
            protected_root,
        ) in protected_python_roots:
            normalized_protected = _normalize_path(
                protected_root,
                description=protected_description,
            )

            _require_nonoverlapping(
                self.results_root,
                "Python results root",
                normalized_protected,
                protected_description,
            )

    @property
    def random_graph_classification_raw_root(
        self,
    ) -> Path:
        return _safe_child(
            self.cpp_raw_root,
            _RANDOM_GRAPH_CLASSIFICATION_EXPERIMENT,
            component_description=(
                "random-graph-classification raw "
                "experiment name"
            ),
        )

    @property
    def random_graph_classification_cache_root(
        self,
    ) -> Path:
        return self.cache_experiment_dir(
            _RANDOM_GRAPH_CLASSIFICATION_EXPERIMENT
        )

    @property
    def random_graph_classification_processed_root(
        self,
    ) -> Path:
        return self.processed_experiment_dir(
            _RANDOM_GRAPH_CLASSIFICATION_EXPERIMENT
        )

    @property
    def random_graph_classification_figures_root(
        self,
    ) -> Path:
        return self.figures_experiment_dir(
            _RANDOM_GRAPH_CLASSIFICATION_EXPERIMENT
        )

    @property
    def random_graph_classification_tables_root(
        self,
    ) -> Path:
        return self.tables_experiment_dir(
            _RANDOM_GRAPH_CLASSIFICATION_EXPERIMENT
        )

    def raw_experiment_dir(
        self,
        experiment: str,
    ) -> Path:
        return _safe_child(
            self.cpp_raw_root,
            experiment,
            component_description=(
                "raw experiment name"
            ),
        )

    def raw_study_dir(
        self,
        experiment: str,
        study: str,
    ) -> Path:
        experiment_root = self.raw_experiment_dir(
            experiment
        )

        return _safe_child(
            experiment_root,
            study,
            component_description=(
                "raw study name"
            ),
        )

    def raw_experiment_file(
        self,
        experiment: str,
        filename: str,
    ) -> Path:
        owner = self.raw_experiment_dir(
            experiment
        )

        return _safe_file(
            owner,
            filename,
            filename_description=(
                "raw experiment filename"
            ),
        )

    def raw_study_file(
        self,
        experiment: str,
        study: str,
        filename: str,
    ) -> Path:
        owner = self.raw_study_dir(
            experiment,
            study,
        )

        return _safe_file(
            owner,
            filename,
            filename_description=(
                "raw study filename"
            ),
        )

    def cache_experiment_dir(
        self,
        experiment: str,
    ) -> Path:
        return _safe_child(
            self.cache_root,
            experiment,
            component_description=(
                "cache experiment name"
            ),
        )

    def cache_study_dir(
        self,
        experiment: str,
        study: str,
    ) -> Path:
        experiment_root = (
            self.cache_experiment_dir(
                experiment
            )
        )

        return _safe_child(
            experiment_root,
            study,
            component_description=(
                "cache study name"
            ),
        )

    def cache_file(
        self,
        experiment: str,
        filename: str,
        *,
        study: str | None = None,
    ) -> Path:
        owner = (
            self.cache_experiment_dir(
                experiment
            )
            if study is None
            else self.cache_study_dir(
                experiment,
                study,
            )
        )

        return _safe_file(
            owner,
            filename,
            filename_description=(
                "cache filename"
            ),
        )

    def processed_experiment_dir(
        self,
        experiment: str,
    ) -> Path:
        return _safe_child(
            self.processed_root,
            experiment,
            component_description=(
                "processed experiment name"
            ),
        )

    def processed_study_dir(
        self,
        experiment: str,
        study: str,
    ) -> Path:
        experiment_root = (
            self.processed_experiment_dir(
                experiment
            )
        )

        return _safe_child(
            experiment_root,
            study,
            component_description=(
                "processed study name"
            ),
        )

    def processed_file(
        self,
        experiment: str,
        filename: str,
        *,
        study: str | None = None,
    ) -> Path:
        owner = (
            self.processed_experiment_dir(
                experiment
            )
            if study is None
            else self.processed_study_dir(
                experiment,
                study,
            )
        )

        return _safe_file(
            owner,
            filename,
            filename_description=(
                "processed-data filename"
            ),
        )

    def figures_experiment_dir(
        self,
        experiment: str,
    ) -> Path:
        return _safe_child(
            self.figures_root,
            experiment,
            component_description=(
                "figures experiment name"
            ),
        )

    def figure_file(
        self,
        experiment: str,
        filename: str,
    ) -> Path:
        owner = self.figures_experiment_dir(
            experiment
        )

        return _safe_file(
            owner,
            filename,
            filename_description=(
                "figure filename"
            ),
        )

    def tables_experiment_dir(
        self,
        experiment: str,
    ) -> Path:
        return _safe_child(
            self.tables_root,
            experiment,
            component_description=(
                "tables experiment name"
            ),
        )

    def table_file(
        self,
        experiment: str,
        filename: str,
    ) -> Path:
        owner = self.tables_experiment_dir(
            experiment
        )

        return _safe_file(
            owner,
            filename,
            filename_description=(
                "table filename"
            ),
        )

    def _require_python_owned_target(
        self,
        target: Path,
    ) -> None:
        normalized_target = _normalize_path(
            target,
            description="Python output target",
        )

        owners = (
            self.cache_root,
            self.processed_root,
            self.figures_root,
            self.tables_root,
        )

        if not any(
            _is_within(
                normalized_target,
                owner,
                allow_equal=True,
            )
            for owner in owners
        ):
            raise ValueError(
                "Refusing to create a directory outside "
                "the Python-owned output roots: "
                f"'{normalized_target}'."
            )

        if _paths_overlap(
            normalized_target,
            self.cpp_raw_root,
        ):
            raise ValueError(
                "Refusing to create a Python output "
                "directory that overlaps immutable C++ "
                "raw data: "
                f"target='{normalized_target}', "
                f"raw_root='{self.cpp_raw_root}'."
            )

    def create_directory(
        self,
        target: Path,
    ) -> Path:
        normalized_target = _normalize_path(
            target,
            description="Python output directory",
        )

        self._require_python_owned_target(
            normalized_target
        )

        _require_directory_if_present(
            normalized_target,
            description=(
                "Python output directory"
            ),
        )

        normalized_target.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not normalized_target.is_dir():
            raise RuntimeError(
                "Could not create Python output "
                f"directory '{normalized_target}'."
            )

        return normalized_target

    def create_output_directories(
        self,
        *,
        experiments: Iterable[str] = (),
    ) -> None:
        roots = (
            self.cache_root,
            self.processed_root,
            self.figures_root,
            self.tables_root,
        )

        for root in roots:
            self.create_directory(
                root
            )

        validated_experiments = tuple(
            validate_identifier(
                experiment,
                description="experiment name",
            )
            for experiment in experiments
        )

        for experiment in validated_experiments:
            self.create_directory(
                self.cache_experiment_dir(
                    experiment
                )
            )

            self.create_directory(
                self.processed_experiment_dir(
                    experiment
                )
            )

            self.create_directory(
                self.figures_experiment_dir(
                    experiment
                )
            )

            self.create_directory(
                self.tables_experiment_dir(
                    experiment
                )
            )

    def create_processed_study_directory(
        self,
        experiment: str,
        study: str,
    ) -> Path:
        return self.create_directory(
            self.processed_study_dir(
                experiment,
                study,
            )
        )

    def create_cache_study_directory(
        self,
        experiment: str,
        study: str,
    ) -> Path:
        return self.create_directory(
            self.cache_study_dir(
                experiment,
                study,
            )
        )


def default_project_paths(
) -> ProjectPaths:
    return ProjectPaths.discover()