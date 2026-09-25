from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray


SINGLE_COLUMN_WIDTH_INCHES: Final[float] = 3.35
DOUBLE_COLUMN_WIDTH_INCHES: Final[float] = 7.00

DEFAULT_ASPECT_RATIO: Final[float] = 1.45

DEFAULT_DPI: Final[int] = 300
PREVIEW_DPI: Final[int] = 180

DEFAULT_LINE_WIDTH: Final[float] = 1.6
DEFAULT_MARKER_SIZE: Final[float] = 5.0
DEFAULT_CAP_SIZE: Final[float] = 3.0
DEFAULT_ERROR_LINE_WIDTH: Final[float] = 1.1

DEFAULT_GRID_ALPHA: Final[float] = 0.20
DEFAULT_BAND_ALPHA: Final[float] = 0.18

COLOR_BLACK: Final[str] = "#000000"
COLOR_GRAY: Final[str] = "#999999"
COLOR_ORANGE: Final[str] = "#E69F00"
COLOR_SKY_BLUE: Final[str] = "#56B4E9"
COLOR_GREEN: Final[str] = "#009E73"
COLOR_YELLOW: Final[str] = "#F0E442"
COLOR_BLUE: Final[str] = "#0072B2"
COLOR_RED: Final[str] = "#D55E00"
COLOR_PURPLE: Final[str] = "#CC79A7"
COLOR_BROWN: Final[str] = "#8C564B"

DEFAULT_COLOR_CYCLE: Final[tuple[str, ...]] = (
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_PURPLE,
    COLOR_SKY_BLUE,
    COLOR_BROWN,
    COLOR_GRAY,
)

DEFAULT_MARKERS: Final[tuple[str, ...]] = (
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "*",
)

DEFAULT_LINESTYLES: Final[tuple[str, ...]] = (
    "-",
    "--",
    "-.",
    ":",
)

SUPPORTED_VECTOR_FORMATS: Final[frozenset[str]] = frozenset(
    {
        "pdf",
        "svg",
    }
)

SUPPORTED_RASTER_FORMATS: Final[frozenset[str]] = frozenset(
    {
        "png",
    }
)

SUPPORTED_FIGURE_FORMATS: Final[frozenset[str]] = (
    SUPPORTED_VECTOR_FORMATS
    | SUPPORTED_RASTER_FORMATS
)


class PlottingError(RuntimeError):
    pass


def _as_finite_scalar(
    value: Any,
    *,
    description: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{description.capitalize()} must be numeric."
        )

    try:
        parsed = float(
            value
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{description.capitalize()} must be numeric."
        ) from error

    if not math.isfinite(
        parsed
    ):
        raise ValueError(
            f"{description.capitalize()} must be finite."
        )

    if positive and parsed <= 0.0:
        raise ValueError(
            f"{description.capitalize()} must be positive."
        )

    if nonnegative and parsed < 0.0:
        raise ValueError(
            f"{description.capitalize()} must be nonnegative."
        )

    return parsed


def _as_finite_1d(
    values: ArrayLike,
    *,
    description: str,
    require_nonempty: bool = True,
) -> NDArray[np.float64]:
    try:
        array = np.asarray(
            values,
            dtype=np.float64,
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{description.capitalize()} must be numeric."
        ) from error

    if array.ndim != 1:
        raise ValueError(
            f"{description.capitalize()} must be one-dimensional."
        )

    if (
        require_nonempty
        and array.size == 0
    ):
        raise ValueError(
            f"{description.capitalize()} must not be empty."
        )

    if not bool(
        np.all(
            np.isfinite(
                array
            )
        )
    ):
        raise ValueError(
            f"{description.capitalize()} contains nonfinite values."
        )

    return np.array(
        array,
        dtype=np.float64,
        copy=True,
    )


def _require_matching_shapes(
    arrays: Sequence[NDArray[np.float64]],
    *,
    description: str,
) -> None:
    if not arrays:
        raise ValueError(
            f"{description.capitalize()} requires at least one array."
        )

    shape = arrays[0].shape

    for array in arrays[1:]:
        if array.shape != shape:
            raise ValueError(
                f"{description.capitalize()} arrays must have matching "
                "shapes."
            )


def _require_axis(
    axis: Axes,
) -> None:
    if not isinstance(
        axis,
        Axes,
    ):
        raise TypeError(
            "axis must be a matplotlib.axes.Axes instance."
        )


def _require_figure(
    figure: Figure,
) -> None:
    if not isinstance(
        figure,
        Figure,
    ):
        raise TypeError(
            "figure must be a matplotlib.figure.Figure instance."
        )


def _require_string_or_none(
    value: str | None,
    *,
    description: str,
) -> None:
    if (
        value is not None
        and not isinstance(
            value,
            str,
        )
    ):
        raise TypeError(
            f"{description.capitalize()} must be a string or None."
        )


def _require_positive_integer(
    value: int,
    *,
    description: str,
) -> int:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"{description.capitalize()} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{description.capitalize()} must be positive."
        )

    return value


def _as_output_path(
    path: str | os.PathLike[str],
) -> Path:
    if path is None:
        raise TypeError(
            "Figure output path must not be None."
        )

    try:
        raw = os.fspath(
            path
        )
    except TypeError as error:
        raise TypeError(
            "Figure output path must be path-like."
        ) from error

    if isinstance(
        raw,
        bytes,
    ):
        if not raw:
            raise ValueError(
                "Figure output path must not be empty."
            )

        try:
            raw = os.fsdecode(
                raw
            )
        except Exception as error:
            raise ValueError(
                "Figure output path could not be decoded."
            ) from error

    if not isinstance(
        raw,
        str,
    ):
        raise TypeError(
            "Figure output path must resolve to a string path."
        )

    if raw == "":
        raise ValueError(
            "Figure output path must not be empty."
        )

    result = Path(
        raw
    )

    suffix = result.suffix.lower()

    if not suffix:
        raise ValueError(
            "Figure output path must include a file extension."
        )

    format_name = suffix.lstrip(
        "."
    )

    if format_name not in SUPPORTED_FIGURE_FORMATS:
        raise ValueError(
            f"Unsupported figure output format '.{format_name}'."
        )

    return result


def _require_output_parent(
    path: Path,
) -> None:
    parent = path.parent

    try:
        exists = parent.exists()
    except OSError as error:
        raise PlottingError(
            f"Could not inspect figure output parent '{parent}': "
            f"{error}"
        ) from error

    if not exists:
        raise FileNotFoundError(
            f"Figure output parent directory does not exist: "
            f"'{parent}'."
        )

    try:
        directory = parent.is_dir()
    except OSError as error:
        raise PlottingError(
            f"Could not inspect figure output parent '{parent}': "
            f"{error}"
        ) from error

    if not directory:
        raise PlottingError(
            f"Figure output parent path is not a directory: "
            f"'{parent}'."
        )


def _require_output_target(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    _require_output_parent(
        path
    )

    try:
        exists = path.exists()
    except OSError as error:
        raise PlottingError(
            f"Could not inspect figure output target '{path}': "
            f"{error}"
        ) from error

    if not exists:
        return

    try:
        regular = path.is_file()
    except OSError as error:
        raise PlottingError(
            f"Could not inspect figure output target '{path}': "
            f"{error}"
        ) from error

    if not regular:
        raise PlottingError(
            f"Figure output target is not a regular file: '{path}'."
        )

    if not overwrite:
        raise FileExistsError(
            f"Figure output target already exists: '{path}'."
        )


def _temporary_figure_path(
    target: Path,
) -> Path:
    _require_output_parent(
        target
    )

    suffix = target.suffix.lower()

    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.stem}.",
            suffix=f".tmp{suffix}",
            dir=target.parent,
        )
    except OSError as error:
        raise PlottingError(
            f"Could not create a temporary figure file beside "
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
            f"Figure output target already exists: '{target_path}'."
        ) from error
    except OSError as error:
        raise PlottingError(
            "Could not atomically publish figure without overwriting "
            f"'{target_path}': {error}"
        ) from error

    _remove_if_present(
        temporary_path
    )


def _publish_figure(
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
            raise PlottingError(
                f"Could not atomically replace figure "
                f"'{target_path}': {error}"
            ) from error

        return

    _publish_without_overwrite(
        temporary_path,
        target_path,
    )


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
            ],
            "mathtext.fontset": "dejavuserif",
            "text.usetex": False,
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.prop_cycle": mpl.cycler(
                color=DEFAULT_COLOR_CYCLE
            ),
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.top": False,
            "ytick.right": False,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "lines.linewidth": DEFAULT_LINE_WIDTH,
            "lines.markersize": DEFAULT_MARKER_SIZE,
            "figure.dpi": PREVIEW_DPI,
            "savefig.dpi": DEFAULT_DPI,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure_size(
    *,
    width: float = SINGLE_COLUMN_WIDTH_INCHES,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> tuple[float, float]:
    parsed_width = _as_finite_scalar(
        width,
        description="figure width",
        positive=True,
    )

    parsed_aspect_ratio = _as_finite_scalar(
        aspect_ratio,
        description="figure aspect ratio",
        positive=True,
    )

    height = (
        parsed_width
        / parsed_aspect_ratio
    )

    return (
        parsed_width,
        height,
    )


def create_figure(
    *,
    width: float = SINGLE_COLUMN_WIDTH_INCHES,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    constrained_layout: bool = True,
) -> tuple[Figure, Axes]:
    if not isinstance(
        constrained_layout,
        bool,
    ):
        raise TypeError(
            "constrained_layout must be boolean."
        )

    configure_matplotlib()

    size = figure_size(
        width=width,
        aspect_ratio=aspect_ratio,
    )

    figure, axis = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=size,
        constrained_layout=constrained_layout,
        squeeze=True,
    )

    return (
        figure,
        axis,
    )


def create_subplots(
    rows: int,
    columns: int,
    *,
    width: float = DOUBLE_COLUMN_WIDTH_INCHES,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    sharex: bool = False,
    sharey: bool = False,
    constrained_layout: bool = True,
) -> tuple[Figure, NDArray[np.object_]]:
    parsed_rows = _require_positive_integer(
        rows,
        description="subplot row count",
    )

    parsed_columns = _require_positive_integer(
        columns,
        description="subplot column count",
    )

    if not isinstance(
        sharex,
        bool,
    ):
        raise TypeError(
            "sharex must be boolean."
        )

    if not isinstance(
        sharey,
        bool,
    ):
        raise TypeError(
            "sharey must be boolean."
        )

    if not isinstance(
        constrained_layout,
        bool,
    ):
        raise TypeError(
            "constrained_layout must be boolean."
        )

    configure_matplotlib()

    size = figure_size(
        width=width,
        aspect_ratio=aspect_ratio,
    )

    figure, axes = plt.subplots(
        nrows=parsed_rows,
        ncols=parsed_columns,
        figsize=size,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=constrained_layout,
        squeeze=False,
    )

    result = np.asarray(
        axes,
        dtype=object,
    )

    if result.shape != (
        parsed_rows,
        parsed_columns,
    ):
        raise PlottingError(
            "Matplotlib returned an unexpected subplot array shape."
        )

    return (
        figure,
        result,
    )


def style_axis(
    axis: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = False,
    xgrid: bool | None = None,
    ygrid: bool | None = None,
) -> None:
    _require_axis(
        axis
    )

    _require_string_or_none(
        xlabel,
        description="x-axis label",
    )

    _require_string_or_none(
        ylabel,
        description="y-axis label",
    )

    _require_string_or_none(
        title,
        description="axis title",
    )

    if not isinstance(
        grid,
        bool,
    ):
        raise TypeError(
            "grid must be boolean."
        )

    if (
        xgrid is not None
        and not isinstance(
            xgrid,
            bool,
        )
    ):
        raise TypeError(
            "xgrid must be boolean or None."
        )

    if (
        ygrid is not None
        and not isinstance(
            ygrid,
            bool,
        )
    ):
        raise TypeError(
            "ygrid must be boolean or None."
        )

    if xlabel is not None:
        axis.set_xlabel(
            xlabel
        )

    if ylabel is not None:
        axis.set_ylabel(
            ylabel
        )

    if title is not None:
        axis.set_title(
            title
        )

    resolved_xgrid = (
        grid
        if xgrid is None
        else xgrid
    )

    resolved_ygrid = (
        grid
        if ygrid is None
        else ygrid
    )

    axis.set_axisbelow(
        True
    )

    axis.xaxis.grid(
        resolved_xgrid,
        alpha=DEFAULT_GRID_ALPHA,
        linewidth=0.6,
    )

    axis.yaxis.grid(
        resolved_ygrid,
        alpha=DEFAULT_GRID_ALPHA,
        linewidth=0.6,
    )


def add_panel_label(
    axis: Axes,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.05,
) -> Any:
    _require_axis(
        axis
    )

    if not isinstance(
        label,
        str,
    ):
        raise TypeError(
            "Panel label must be a string."
        )

    if not label:
        raise ValueError(
            "Panel label must not be empty."
        )

    parsed_x = _as_finite_scalar(
        x,
        description="panel-label x coordinate",
    )

    parsed_y = _as_finite_scalar(
        y,
        description="panel-label y coordinate",
    )

    return axis.text(
        parsed_x,
        parsed_y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        clip_on=False,
    )


def add_legend(
    axis: Axes,
    *,
    location: str = "best",
    columns: int = 1,
    frame: bool = False,
) -> Any:
    _require_axis(
        axis
    )

    if not isinstance(
        location,
        str,
    ):
        raise TypeError(
            "Legend location must be a string."
        )

    if not location:
        raise ValueError(
            "Legend location must not be empty."
        )

    parsed_columns = _require_positive_integer(
        columns,
        description="legend column count",
    )

    if not isinstance(
        frame,
        bool,
    ):
        raise TypeError(
            "frame must be boolean."
        )

    return axis.legend(
        loc=location,
        ncols=parsed_columns,
        frameon=frame,
    )


def add_figure_legend(
    figure: Figure,
    handles: Sequence[Any],
    labels: Sequence[str],
    *,
    location: str = "outside lower center",
    columns: int = 1,
    frame: bool = False,
) -> Any:
    _require_figure(
        figure
    )

    parsed_columns = _require_positive_integer(
        columns,
        description="legend column count",
    )

    return figure.legend(
        handles,
        labels,
        loc=location,
        ncols=parsed_columns,
        frameon=frame,
    )


def add_reference_line(
    axis: Axes,
    value: float,
    *,
    orientation: str,
    color: str = COLOR_GRAY,
    linestyle: str = "--",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    label: str | None = None,
    zorder: float | None = None,
) -> Any:
    _require_axis(
        axis
    )

    parsed_value = _as_finite_scalar(
        value,
        description="reference-line value",
    )

    if not isinstance(
        orientation,
        str,
    ):
        raise TypeError(
            "Reference-line orientation must be a string."
        )

    normalized_orientation = orientation.strip().lower()

    if normalized_orientation not in {
        "horizontal",
        "vertical",
    }:
        raise ValueError(
            "Reference-line orientation must be 'horizontal' "
            "or 'vertical'."
        )

    parsed_linewidth = _as_finite_scalar(
        linewidth,
        description="reference-line width",
        positive=True,
    )

    parsed_alpha = _as_finite_scalar(
        alpha,
        description="reference-line alpha",
        nonnegative=True,
    )

    if parsed_alpha > 1.0:
        raise ValueError(
            "Reference-line alpha must not exceed 1."
        )

    _require_string_or_none(
        label,
        description="reference-line label",
    )

    kwargs: dict[str, Any] = {
        "color": color,
        "linestyle": linestyle,
        "linewidth": parsed_linewidth,
        "alpha": parsed_alpha,
        "label": label,
    }

    if zorder is not None:
        kwargs["zorder"] = _as_finite_scalar(
            zorder,
            description="reference-line z-order",
        )

    if normalized_orientation == "horizontal":
        return axis.axhline(
            parsed_value,
            **kwargs,
        )

    return axis.axvline(
        parsed_value,
        **kwargs,
    )


def plot_interval_band(
    axis: Axes,
    x: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    color: str,
    alpha: float = DEFAULT_BAND_ALPHA,
    label: str | None = None,
    zorder: float | None = None,
) -> Any:
    _require_axis(
        axis
    )

    x_values = _as_finite_1d(
        x,
        description="interval-band x values",
    )

    lower_values = _as_finite_1d(
        lower,
        description="interval-band lower values",
    )

    upper_values = _as_finite_1d(
        upper,
        description="interval-band upper values",
    )

    _require_matching_shapes(
        (
            x_values,
            lower_values,
            upper_values,
        ),
        description="interval band",
    )

    if bool(
        np.any(
            lower_values
            >
            upper_values
        )
    ):
        raise ValueError(
            "Interval-band lower values must not exceed upper values."
        )

    parsed_alpha = _as_finite_scalar(
        alpha,
        description="interval-band alpha",
        nonnegative=True,
    )

    if parsed_alpha > 1.0:
        raise ValueError(
            "Interval-band alpha must not exceed 1."
        )

    _require_string_or_none(
        label,
        description="interval-band label",
    )

    kwargs: dict[str, Any] = {
        "color": color,
        "alpha": parsed_alpha,
        "label": label,
        "linewidth": 0.0,
    }

    if zorder is not None:
        kwargs["zorder"] = _as_finite_scalar(
            zorder,
            description="interval-band z-order",
        )

    return axis.fill_between(
        x_values,
        lower_values,
        upper_values,
        **kwargs,
    )


def plot_line_with_interval(
    axis: Axes,
    x: ArrayLike,
    center: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    color: str,
    label: str | None = None,
    marker: str | None = None,
    linestyle: str = "-",
    linewidth: float = DEFAULT_LINE_WIDTH,
    markersize: float = DEFAULT_MARKER_SIZE,
    alpha: float = DEFAULT_BAND_ALPHA,
    zorder: float | None = None,
) -> tuple[Any, Any]:
    _require_axis(
        axis
    )

    x_values = _as_finite_1d(
        x,
        description="line x values",
    )

    center_values = _as_finite_1d(
        center,
        description="line center values",
    )

    lower_values = _as_finite_1d(
        lower,
        description="line lower values",
    )

    upper_values = _as_finite_1d(
        upper,
        description="line upper values",
    )

    _require_matching_shapes(
        (
            x_values,
            center_values,
            lower_values,
            upper_values,
        ),
        description="line-with-interval",
    )

    if bool(
        np.any(
            lower_values
            >
            upper_values
        )
    ):
        raise ValueError(
            "Line interval lower values must not exceed upper values."
        )

    parsed_linewidth = _as_finite_scalar(
        linewidth,
        description="line width",
        positive=True,
    )

    parsed_markersize = _as_finite_scalar(
        markersize,
        description="marker size",
        positive=True,
    )

    parsed_alpha = _as_finite_scalar(
        alpha,
        description="interval alpha",
        nonnegative=True,
    )

    if parsed_alpha > 1.0:
        raise ValueError(
            "Interval alpha must not exceed 1."
        )

    _require_string_or_none(
        label,
        description="line label",
    )

    if (
        marker is not None
        and not isinstance(
            marker,
            str,
        )
    ):
        raise TypeError(
            "marker must be a string or None."
        )

    if not isinstance(
        linestyle,
        str,
    ):
        raise TypeError(
            "linestyle must be a string."
        )

    base_zorder = (
        2.0
        if zorder is None
        else _as_finite_scalar(
            zorder,
            description="line z-order",
        )
    )

    band = plot_interval_band(
        axis,
        x_values,
        lower_values,
        upper_values,
        color=color,
        alpha=parsed_alpha,
        label=None,
        zorder=base_zorder - 1.0,
    )

    line, = axis.plot(
        x_values,
        center_values,
        color=color,
        label=label,
        marker=marker,
        linestyle=linestyle,
        linewidth=parsed_linewidth,
        markersize=parsed_markersize,
        zorder=base_zorder,
    )

    return (
        line,
        band,
    )


def plot_point_intervals(
    axis: Axes,
    x: ArrayLike,
    center: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    color: str,
    label: str | None = None,
    marker: str = "o",
    markersize: float = DEFAULT_MARKER_SIZE,
    interval_linewidth: float = DEFAULT_ERROR_LINE_WIDTH,
    zorder: float | None = None,
) -> tuple[Any, Any]:
    _require_axis(
        axis
    )

    x_values = _as_finite_1d(
        x,
        description="point-interval x values",
    )

    center_values = _as_finite_1d(
        center,
        description="point-interval center values",
    )

    lower_values = _as_finite_1d(
        lower,
        description="point-interval lower values",
    )

    upper_values = _as_finite_1d(
        upper,
        description="point-interval upper values",
    )

    _require_matching_shapes(
        (
            x_values,
            center_values,
            lower_values,
            upper_values,
        ),
        description="point-interval plot",
    )

    if bool(
        np.any(
            lower_values
            >
            upper_values
        )
    ):
        raise ValueError(
            "Point-interval lower values must not exceed upper values."
        )

    parsed_markersize = _as_finite_scalar(
        markersize,
        description="marker size",
        positive=True,
    )

    parsed_interval_linewidth = _as_finite_scalar(
        interval_linewidth,
        description="interval line width",
        positive=True,
    )

    base_zorder = (
        2.0
        if zorder is None
        else _as_finite_scalar(
            zorder,
            description="point-interval z-order",
        )
    )

    intervals = axis.vlines(
        x_values,
        lower_values,
        upper_values,
        color=color,
        linewidth=parsed_interval_linewidth,
        zorder=base_zorder - 1.0,
    )

    points = axis.scatter(
        x_values,
        center_values,
        color=color,
        marker=marker,
        s=parsed_markersize
        *
        parsed_markersize,
        label=label,
        zorder=base_zorder,
    )

    return (
        points,
        intervals,
    )


def plot_errorbars(
    axis: Axes,
    x: ArrayLike,
    center: ArrayLike,
    errors: ArrayLike,
    *,
    color: str,
    label: str | None = None,
    marker: str = "o",
    linestyle: str = "-",
    linewidth: float = DEFAULT_LINE_WIDTH,
    markersize: float = DEFAULT_MARKER_SIZE,
    capsize: float = DEFAULT_CAP_SIZE,
    error_linewidth: float = DEFAULT_ERROR_LINE_WIDTH,
    zorder: float | None = None,
) -> Any:
    _require_axis(
        axis
    )

    x_values = _as_finite_1d(
        x,
        description="error-bar x values",
    )

    center_values = _as_finite_1d(
        center,
        description="error-bar center values",
    )

    error_values = _as_finite_1d(
        errors,
        description="error-bar magnitudes",
    )

    _require_matching_shapes(
        (
            x_values,
            center_values,
            error_values,
        ),
        description="error-bar plot",
    )

    if bool(
        np.any(
            error_values
            <
            0.0
        )
    ):
        raise ValueError(
            "Error-bar magnitudes must be nonnegative."
        )

    parsed_linewidth = _as_finite_scalar(
        linewidth,
        description="line width",
        positive=True,
    )

    parsed_markersize = _as_finite_scalar(
        markersize,
        description="marker size",
        positive=True,
    )

    parsed_capsize = _as_finite_scalar(
        capsize,
        description="error-bar cap size",
        nonnegative=True,
    )

    parsed_error_linewidth = _as_finite_scalar(
        error_linewidth,
        description="error-bar line width",
        positive=True,
    )

    _require_string_or_none(
        label,
        description="error-bar label",
    )

    if not isinstance(
        marker,
        str,
    ):
        raise TypeError(
            "marker must be a string."
        )

    if not isinstance(
        linestyle,
        str,
    ):
        raise TypeError(
            "linestyle must be a string."
        )

    kwargs: dict[str, Any] = {
        "color": color,
        "label": label,
        "marker": marker,
        "linestyle": linestyle,
        "linewidth": parsed_linewidth,
        "markersize": parsed_markersize,
        "capsize": parsed_capsize,
        "elinewidth": parsed_error_linewidth,
    }

    if zorder is not None:
        kwargs[
            "zorder"
        ] = _as_finite_scalar(
            zorder,
            description="error-bar z-order",
        )

    return axis.errorbar(
        x_values,
        center_values,
        yerr=error_values,
        **kwargs,
    )


def use_scientific_notation(
    axis: Axes,
    *,
    axis_name: str = "y",
    limits: tuple[int, int] = (
        -3,
        4,
    ),
) -> None:
    _require_axis(
        axis
    )

    if not isinstance(
        axis_name,
        str,
    ):
        raise TypeError(
            "Scientific-notation axis name must be a string."
        )

    normalized_axis = axis_name.strip().lower()

    if normalized_axis not in {
        "x",
        "y",
        "both",
    }:
        raise ValueError(
            "Scientific-notation axis name must be 'x', 'y', "
            "or 'both'."
        )

    if (
        not isinstance(
            limits,
            tuple,
        )
        or len(
            limits
        )
        != 2
    ):
        raise TypeError(
            "Scientific-notation limits must be a two-integer tuple."
        )

    lower, upper = limits

    if (
        isinstance(
            lower,
            bool,
        )
        or isinstance(
            upper,
            bool,
        )
        or not isinstance(
            lower,
            int,
        )
        or not isinstance(
            upper,
            int,
        )
    ):
        raise TypeError(
            "Scientific-notation limits must contain integers."
        )

    if lower > upper:
        raise ValueError(
            "Scientific-notation lower limit must not exceed "
            "the upper limit."
        )

    axis.ticklabel_format(
        axis=normalized_axis,
        style="sci",
        scilimits=(
            lower,
            upper,
        ),
        useMathText=True,
    )


def save_figure(
    figure: Figure,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    dpi: int = DEFAULT_DPI,
    transparent: bool = False,
    close: bool = False,
) -> Path:
    _require_figure(
        figure
    )

    if not isinstance(
        overwrite,
        bool,
    ):
        raise TypeError(
            "overwrite must be boolean."
        )

    parsed_dpi = _require_positive_integer(
        dpi,
        description="figure DPI",
    )

    if not isinstance(
        transparent,
        bool,
    ):
        raise TypeError(
            "transparent must be boolean."
        )

    if not isinstance(
        close,
        bool,
    ):
        raise TypeError(
            "close must be boolean."
        )

    target = _as_output_path(
        path
    )

    _require_output_target(
        target,
        overwrite=overwrite,
    )

    temporary = _temporary_figure_path(
        target
    )

    format_name = target.suffix.lower().lstrip(
        "."
    )

    try:
        try:
            figure.savefig(
                temporary,
                format=format_name,
                dpi=parsed_dpi,
                transparent=transparent,
                bbox_inches="tight",
            )
        except Exception as error:
            raise PlottingError(
                f"Could not render figure to temporary output "
                f"'{temporary}': {error}"
            ) from error

        try:
            exists = temporary.exists()
        except OSError as error:
            raise PlottingError(
                f"Could not inspect temporary figure output "
                f"'{temporary}': {error}"
            ) from error

        if not exists:
            raise PlottingError(
                f"Matplotlib did not create temporary figure output "
                f"'{temporary}'."
            )

        try:
            regular = temporary.is_file()
        except OSError as error:
            raise PlottingError(
                f"Could not inspect temporary figure output "
                f"'{temporary}': {error}"
            ) from error

        if not regular:
            raise PlottingError(
                f"Temporary figure output is not a regular file: "
                f"'{temporary}'."
            )

        try:
            size = temporary.stat().st_size
        except OSError as error:
            raise PlottingError(
                f"Could not inspect temporary figure size "
                f"'{temporary}': {error}"
            ) from error

        if size <= 0:
            raise PlottingError(
                f"Temporary figure output is empty: '{temporary}'."
            )

        _publish_figure(
            temporary,
            target,
            overwrite=overwrite,
        )
    except Exception:
        _remove_if_present(
            temporary
        )
        raise

    if close:
        plt.close(
            figure
        )

    return target