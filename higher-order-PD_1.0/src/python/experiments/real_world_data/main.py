from __future__ import annotations

import sys

from core.paths import (
    default_project_paths,
)

from .configuration import (
    EXPERIMENT_NAME,
    default_real_world_data_config,
)
from .data import (
    load_real_world_data,
)
from .experiment import (
    run_real_world_experiment,
)
from .output import (
    AnalysisOutputLayout,
    write_analysis_configuration,
    write_real_world_result,
)


def main(
) -> int:
    paths = default_project_paths()

    config = (
        default_real_world_data_config()
    )

    raw_root = paths.raw_experiment_dir(
        EXPERIMENT_NAME
    )

    tu_root = (
        paths.cpp_root
        /
        "experiments"
        /
        "real_world_data"
        /
        "data"
    )

    output_root = (
        paths.processed_experiment_dir(
            EXPERIMENT_NAME
        )
    )

    print(
        "[Real world] Loading C++ cache and TU topology",
        flush=True,
    )

    data = load_real_world_data(
        raw_root,
        tu_root,
    )

    print(
        "[Real world] Running four-method MIL benchmark",
        flush=True,
    )

    result = run_real_world_experiment(
        data,
        config,
    )

    layout = AnalysisOutputLayout(
        output_root
    )

    print(
        "[Real world] Writing processed outputs",
        flush=True,
    )

    write_analysis_configuration(
        layout,
        data,
        config,
    )

    write_real_world_result(
        layout,
        result,
    )

    print(
        "[Real world] Complete",
        flush=True,
    )

    print(
        f"Output root: {output_root}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )