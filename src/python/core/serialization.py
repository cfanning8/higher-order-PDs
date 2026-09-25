from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
from numpy.typing import NDArray


PathInput = str | PathLike[str]
JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)

_CHECKSUM_CHUNK_SIZE = 1024 * 1024


class SerializationError(RuntimeError):
    pass


def _as_path(
    path: PathInput,
    *,
    description: str = "path",
) -> Path:
    if path is None:
        raise TypeError(
            f"{description.capitalize()} must not be None."
        )

    try:
        raw_path = os.fspath(path)
    except TypeError as error:
        raise TypeError(
            f"{description.capitalize()} must be path-like."
        ) from error

    if isinstance(raw_path, bytes):
        if not raw_path:
            raise ValueError(
                f"{description.capitalize()} must not be empty."
            )

        try:
            raw_path = os.fsdecode(raw_path)
        except Exception as error:
            raise ValueError(
                f"{description.capitalize()} could not be decoded."
            ) from error

    if not isinstance(raw_path, str):
        raise TypeError(
            f"{description.capitalize()} must resolve to a string path."
        )

    if raw_path == "":
        raise ValueError(
            f"{description.capitalize()} must not be empty."
        )

    return Path(raw_path)


def _require_existing_regular_file(
    path: Path,
    *,
    description: str = "file",
) -> None:
    try:
        exists = path.exists()
    except OSError as error:
        raise SerializationError(
            f"Could not inspect {description} '{path}': {error}"
        ) from error

    if not exists:
        raise FileNotFoundError(
            f"{description.capitalize()} does not exist: '{path}'."
        )

    try:
        regular = path.is_file()
    except OSError as error:
        raise SerializationError(
            f"Could not inspect {description} '{path}': {error}"
        ) from error

    if not regular:
        raise SerializationError(
            f"{description.capitalize()} is not a regular file: "
            f"'{path}'."
        )


def _require_existing_parent(
    path: Path,
) -> None:
    parent = path.parent

    try:
        exists = parent.exists()
    except OSError as error:
        raise SerializationError(
            f"Could not inspect output parent '{parent}': {error}"
        ) from error

    if not exists:
        raise FileNotFoundError(
            f"Output parent directory does not exist: '{parent}'."
        )

    try:
        directory = parent.is_dir()
    except OSError as error:
        raise SerializationError(
            f"Could not inspect output parent '{parent}': {error}"
        ) from error

    if not directory:
        raise SerializationError(
            f"Output parent path is not a directory: '{parent}'."
        )


def _require_output_target(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    _require_existing_parent(
        path
    )

    try:
        exists = path.exists()
    except OSError as error:
        raise SerializationError(
            f"Could not inspect output target '{path}': {error}"
        ) from error

    if not exists:
        return

    try:
        regular = path.is_file()
    except OSError as error:
        raise SerializationError(
            f"Could not inspect output target '{path}': {error}"
        ) from error

    if not regular:
        raise SerializationError(
            f"Output target is not a regular file: '{path}'."
        )

    if not overwrite:
        raise FileExistsError(
            f"Output target already exists: '{path}'."
        )


def _temporary_path(
    target: Path,
) -> Path:
    _require_existing_parent(
        target
    )

    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
    except OSError as error:
        raise SerializationError(
            f"Could not create a temporary file beside "
            f"'{target}': {error}"
        ) from error

    try:
        os.close(
            descriptor
        )
    except OSError:
        try:
            os.unlink(
                name
            )
        except OSError:
            pass
        raise

    return Path(
        name
    )


def _remove_if_present(
    path: Path,
) -> None:
    try:
        path.unlink(
            missing_ok=True
        )
    except OSError:
        pass


def _publish_without_overwrite(
    temporary_path: Path,
    target_path: Path,
) -> None:
    try:
        os.link(
            temporary_path,
            target_path,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"Output target already exists: '{target_path}'."
        ) from error
    except OSError as error:
        raise SerializationError(
            "Could not atomically publish output without overwriting "
            f"'{target_path}': {error}"
        ) from error

    _remove_if_present(
        temporary_path
    )


def _publish(
    temporary_path: Path,
    target_path: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        try:
            os.replace(
                temporary_path,
                target_path,
            )
        except OSError as error:
            raise SerializationError(
                f"Could not atomically replace '{target_path}': {error}"
            ) from error

        return

    _publish_without_overwrite(
        temporary_path,
        target_path,
    )


def _write_atomic(
    path: PathInput,
    writer: Any,
    *,
    overwrite: bool,
) -> Path:
    target = _as_path(
        path,
        description="output path",
    )

    _require_output_target(
        target,
        overwrite=overwrite,
    )

    temporary = _temporary_path(
        target
    )

    try:
        writer(
            temporary
        )

        _require_existing_regular_file(
            temporary,
            description="temporary output file",
        )

        _publish(
            temporary,
            target,
            overwrite=overwrite,
        )
    except Exception:
        _remove_if_present(
            temporary
        )
        raise

    return target


def _open_text_for_write(
    path: Path,
) -> TextIO:
    try:
        return path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
    except OSError as error:
        raise SerializationError(
            f"Could not open '{path}' for text output: {error}"
        ) from error


def _open_binary_for_write(
    path: Path,
) -> BinaryIO:
    try:
        return path.open(
            "wb"
        )
    except OSError as error:
        raise SerializationError(
            f"Could not open '{path}' for binary output: {error}"
        ) from error


def _flush_and_sync(
    stream: TextIO | BinaryIO,
    path: Path,
) -> None:
    try:
        stream.flush()
        os.fsync(
            stream.fileno()
        )
    except OSError as error:
        raise SerializationError(
            f"Could not flush and synchronize '{path}': {error}"
        ) from error


def read_text(
    path: PathInput,
) -> str:
    source = _as_path(
        path,
        description="text input path",
    )

    _require_existing_regular_file(
        source,
        description="text input file",
    )

    try:
        with source.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            return stream.read()
    except (OSError, UnicodeError) as error:
        raise SerializationError(
            f"Could not read UTF-8 text file '{source}': {error}"
        ) from error


def write_text(
    path: PathInput,
    text: str,
    *,
    overwrite: bool = False,
) -> Path:
    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "Text output must be a string."
        )

    def writer(
        temporary: Path,
    ) -> None:
        stream = _open_text_for_write(
            temporary
        )

        try:
            stream.write(
                text
            )

            _flush_and_sync(
                stream,
                temporary,
            )
        finally:
            stream.close()

    return _write_atomic(
        path,
        writer,
        overwrite=overwrite,
    )


def _reject_json_constant(
    value: str,
) -> None:
    raise ValueError(
        f"JSON contains nonfinite numeric constant '{value}'."
    )


def _unique_json_object(
    pairs: list[tuple[str, JsonValue]],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"JSON object contains duplicate key '{key}'."
            )

        result[key] = value

    return result


def _validate_json_value(
    value: Any,
    *,
    location: str = "$",
) -> None:
    if value is None:
        return

    if isinstance(
        value,
        bool,
    ):
        return

    if isinstance(
        value,
        int,
    ):
        return

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            raise ValueError(
                f"JSON value at {location} is nonfinite."
            )

        return

    if isinstance(
        value,
        str,
    ):
        return

    if isinstance(
        value,
        list,
    ):
        for index, element in enumerate(
            value
        ):
            _validate_json_value(
                element,
                location=f"{location}[{index}]",
            )

        return

    if isinstance(
        value,
        dict,
    ):
        for key, element in value.items():
            if not isinstance(
                key,
                str,
            ):
                raise TypeError(
                    f"JSON object key at {location} is not a string."
                )

            _validate_json_value(
                element,
                location=f"{location}.{key}",
            )

        return

    raise TypeError(
        f"Unsupported JSON value at {location}: "
        f"{type(value).__name__}."
    )


def read_json(
    path: PathInput,
) -> JsonValue:
    source = _as_path(
        path,
        description="JSON input path",
    )

    text = read_text(
        source
    )

    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise SerializationError(
            f"Could not parse JSON file '{source}': {error}"
        ) from error

    _validate_json_value(
        value
    )

    return value


def write_json(
    path: PathInput,
    value: JsonValue,
    *,
    overwrite: bool = False,
    indent: int = 2,
) -> Path:
    if (
        isinstance(
            indent,
            bool,
        )
        or not isinstance(
            indent,
            int,
        )
        or indent < 0
    ):
        raise ValueError(
            "JSON indentation must be a nonnegative integer."
        )

    _validate_json_value(
        value
    )

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise SerializationError(
            f"Could not serialize JSON value: {error}"
        ) from error

    serialized += "\n"

    return write_text(
        path,
        serialized,
        overwrite=overwrite,
    )


def read_csv_rows(
    path: PathInput,
    *,
    expected_header: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    source = _as_path(
        path,
        description="CSV input path",
    )

    _require_existing_regular_file(
        source,
        description="CSV input file",
    )

    expected: tuple[str, ...] | None = None

    if expected_header is not None:
        expected = tuple(
            expected_header
        )

        if not expected:
            raise ValueError(
                "Expected CSV header must not be empty."
            )

        if any(
            not isinstance(column, str) or column == ""
            for column in expected
        ):
            raise ValueError(
                "Expected CSV header contains an invalid column name."
            )

        if len(
            set(expected)
        ) != len(
            expected
        ):
            raise ValueError(
                "Expected CSV header contains duplicate column names."
            )

    try:
        with source.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            reader = csv.reader(
                stream
            )

            try:
                first = next(
                    reader
                )
            except StopIteration as error:
                raise SerializationError(
                    f"CSV file is empty: '{source}'."
                ) from error

            header = tuple(
                first
            )

            if not header:
                raise SerializationError(
                    f"CSV file has an empty header: '{source}'."
                )

            if any(
                column == ""
                for column in header
            ):
                raise SerializationError(
                    f"CSV file has an empty header column: '{source}'."
                )

            if len(
                set(header)
            ) != len(
                header
            ):
                raise SerializationError(
                    f"CSV file has duplicate header columns: '{source}'."
                )

            if (
                expected is not None
                and header != expected
            ):
                raise SerializationError(
                    f"CSV header mismatch in '{source}'. "
                    f"Expected {expected}, found {header}."
                )

            rows: list[tuple[str, ...]] = []

            for row_index, raw_row in enumerate(
                reader,
                start=2,
            ):
                row = tuple(
                    raw_row
                )

                if len(
                    row
                ) != len(
                    header
                ):
                    raise SerializationError(
                        f"CSV row {row_index} in '{source}' has "
                        f"{len(row)} fields; expected {len(header)}."
                    )

                rows.append(
                    row
                )

    except SerializationError:
        raise
    except (
        csv.Error,
        OSError,
        UnicodeError,
    ) as error:
        raise SerializationError(
            f"Could not read CSV file '{source}': {error}"
        ) from error

    return header, rows


def read_csv_dicts(
    path: PathInput,
    *,
    expected_header: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    header, rows = read_csv_rows(
        path,
        expected_header=expected_header,
    )

    return [
        dict(
            zip(
                header,
                row,
                strict=True,
            )
        )
        for row in rows
    ]


def write_csv_rows(
    path: PathInput,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    overwrite: bool = False,
) -> Path:
    normalized_header = tuple(
        header
    )

    if not normalized_header:
        raise ValueError(
            "CSV header must not be empty."
        )

    if any(
        not isinstance(column, str) or column == ""
        for column in normalized_header
    ):
        raise ValueError(
            "CSV header contains an invalid column name."
        )

    if len(
        set(normalized_header)
    ) != len(
        normalized_header
    ):
        raise ValueError(
            "CSV header contains duplicate column names."
        )

    materialized_rows: list[tuple[str, ...]] = []

    for row_index, row in enumerate(
        rows,
        start=1,
    ):
        normalized_row = tuple(
            str(value)
            for value in row
        )

        if len(
            normalized_row
        ) != len(
            normalized_header
        ):
            raise ValueError(
                f"CSV output row {row_index} has "
                f"{len(normalized_row)} fields; expected "
                f"{len(normalized_header)}."
            )

        materialized_rows.append(
            normalized_row
        )

    def writer(
        temporary: Path,
    ) -> None:
        stream = _open_text_for_write(
            temporary
        )

        try:
            csv_writer = csv.writer(
                stream,
                lineterminator="\n",
            )

            csv_writer.writerow(
                normalized_header
            )

            csv_writer.writerows(
                materialized_rows
            )

            _flush_and_sync(
                stream,
                temporary,
            )
        except csv.Error as error:
            raise SerializationError(
                f"Could not serialize CSV output '{temporary}': "
                f"{error}"
            ) from error
        finally:
            stream.close()

    return _write_atomic(
        path,
        writer,
        overwrite=overwrite,
    )


def write_csv_dicts(
    path: PathInput,
    header: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    normalized_header = tuple(
        header
    )

    if not normalized_header:
        raise ValueError(
            "CSV header must not be empty."
        )

    if any(
        not isinstance(column, str) or column == ""
        for column in normalized_header
    ):
        raise ValueError(
            "CSV header contains an invalid column name."
        )

    if len(
        set(normalized_header)
    ) != len(
        normalized_header
    ):
        raise ValueError(
            "CSV header contains duplicate column names."
        )

    expected_keys = set(
        normalized_header
    )

    normalized_rows: list[tuple[str, ...]] = []

    for row_index, row in enumerate(
        rows,
        start=1,
    ):
        actual_keys = set(
            row.keys()
        )

        if actual_keys != expected_keys:
            missing = sorted(
                expected_keys - actual_keys
            )

            extra = sorted(
                actual_keys - expected_keys
            )

            raise ValueError(
                f"CSV output row {row_index} has an incompatible "
                f"schema. Missing={missing}, extra={extra}."
            )

        normalized_rows.append(
            tuple(
                str(
                    row[column]
                )
                for column in normalized_header
            )
        )

    return write_csv_rows(
        path,
        normalized_header,
        normalized_rows,
        overwrite=overwrite,
    )


def _validate_array(
    array: NDArray[Any],
    *,
    description: str,
    require_finite: bool,
) -> None:
    if not isinstance(
        array,
        np.ndarray,
    ):
        raise TypeError(
            f"{description.capitalize()} must be a NumPy array."
        )

    if array.dtype.hasobject:
        raise TypeError(
            f"{description.capitalize()} must not use object dtype."
        )

    if require_finite and np.issubdtype(
        array.dtype,
        np.number,
    ):
        try:
            finite = np.isfinite(
                array
            )
        except TypeError as error:
            raise TypeError(
                f"Could not test finiteness of {description}."
            ) from error

        if not bool(
            np.all(
                finite
            )
        ):
            raise ValueError(
                f"{description.capitalize()} contains nonfinite values."
            )


def read_npy(
    path: PathInput,
    *,
    require_finite: bool = False,
) -> NDArray[Any]:
    source = _as_path(
        path,
        description="NumPy input path",
    )

    _require_existing_regular_file(
        source,
        description="NumPy input file",
    )

    try:
        loaded = np.load(
            source,
            allow_pickle=False,
        )
    except (
        OSError,
        ValueError,
        TypeError,
    ) as error:
        raise SerializationError(
            f"Could not load NumPy file '{source}': {error}"
        ) from error

    if not isinstance(
        loaded,
        np.ndarray,
    ):
        raise SerializationError(
            f"NumPy file '{source}' does not contain one array."
        )

    _validate_array(
        loaded,
        description=f"array loaded from '{source}'",
        require_finite=require_finite,
    )

    return loaded


def write_npy(
    path: PathInput,
    array: NDArray[Any],
    *,
    overwrite: bool = False,
    require_finite: bool = False,
) -> Path:
    _validate_array(
        array,
        description="NumPy output array",
        require_finite=require_finite,
    )

    def writer(
        temporary: Path,
    ) -> None:
        stream = _open_binary_for_write(
            temporary
        )

        try:
            np.save(
                stream,
                array,
                allow_pickle=False,
            )

            _flush_and_sync(
                stream,
                temporary,
            )
        except (
            OSError,
            ValueError,
            TypeError,
        ) as error:
            raise SerializationError(
                f"Could not serialize NumPy array to "
                f"'{temporary}': {error}"
            ) from error
        finally:
            stream.close()

    return _write_atomic(
        path,
        writer,
        overwrite=overwrite,
    )


def read_npz(
    path: PathInput,
    *,
    require_finite: bool = False,
) -> dict[str, NDArray[Any]]:
    source = _as_path(
        path,
        description="NumPy archive input path",
    )

    _require_existing_regular_file(
        source,
        description="NumPy archive input file",
    )

    try:
        with np.load(
            source,
            allow_pickle=False,
        ) as archive:
            names = tuple(
                archive.files
            )

            if len(
                set(names)
            ) != len(
                names
            ):
                raise SerializationError(
                    f"NumPy archive '{source}' contains duplicate names."
                )

            arrays: dict[str, NDArray[Any]] = {}

            for name in names:
                if not name:
                    raise SerializationError(
                        f"NumPy archive '{source}' contains an empty "
                        "array name."
                    )

                array = np.asarray(
                    archive[name]
                )

                _validate_array(
                    array,
                    description=(
                        f"array '{name}' loaded from '{source}'"
                    ),
                    require_finite=require_finite,
                )

                arrays[name] = array.copy()

    except SerializationError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
    ) as error:
        raise SerializationError(
            f"Could not load NumPy archive '{source}': {error}"
        ) from error

    return arrays


def write_npz(
    path: PathInput,
    arrays: Mapping[str, NDArray[Any]],
    *,
    overwrite: bool = False,
    compressed: bool = True,
    require_finite: bool = False,
) -> Path:
    if not arrays:
        raise ValueError(
            "NumPy archive output requires at least one array."
        )

    normalized: dict[str, NDArray[Any]] = {}

    for name, array in arrays.items():
        if not isinstance(
            name,
            str,
        ) or not name:
            raise ValueError(
                "Every NumPy archive array name must be a nonempty "
                "string."
            )

        if (
            "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise ValueError(
                f"Invalid NumPy archive array name '{name}'."
            )

        _validate_array(
            array,
            description=f"NumPy archive array '{name}'",
            require_finite=require_finite,
        )

        normalized[name] = array

    def writer(
        temporary: Path,
    ) -> None:
        stream = _open_binary_for_write(
            temporary
        )

        try:
            save_function = (
                np.savez_compressed
                if compressed
                else np.savez
            )

            save_function(
                stream,
                **normalized,
            )

            _flush_and_sync(
                stream,
                temporary,
            )
        except (
            OSError,
            ValueError,
            TypeError,
        ) as error:
            raise SerializationError(
                f"Could not serialize NumPy archive to "
                f"'{temporary}': {error}"
            ) from error
        finally:
            stream.close()

    return _write_atomic(
        path,
        writer,
        overwrite=overwrite,
    )


def file_checksum(
    path: PathInput,
    *,
    algorithm: str = "sha256",
    chunk_size: int = _CHECKSUM_CHUNK_SIZE,
) -> str:
    source = _as_path(
        path,
        description="checksum input path",
    )

    _require_existing_regular_file(
        source,
        description="checksum input file",
    )

    if (
        isinstance(
            chunk_size,
            bool,
        )
        or not isinstance(
            chunk_size,
            int,
        )
        or chunk_size <= 0
    ):
        raise ValueError(
            "Checksum chunk size must be a positive integer."
        )

    try:
        digest = hashlib.new(
            algorithm
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Unsupported checksum algorithm '{algorithm}'."
        ) from error

    try:
        with source.open(
            "rb"
        ) as stream:
            while True:
                chunk = stream.read(
                    chunk_size
                )

                if not chunk:
                    break

                digest.update(
                    chunk
                )
    except OSError as error:
        raise SerializationError(
            f"Could not checksum file '{source}': {error}"
        ) from error

    return digest.hexdigest()


def sha256_file(
    path: PathInput,
) -> str:
    return file_checksum(
        path,
        algorithm="sha256",
    )