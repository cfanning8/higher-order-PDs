cmake_minimum_required(
    VERSION 3.20
)

project(
    graph_benchmark
    VERSION 1.0.0
    LANGUAGES CXX
)

set(
    CMAKE_CXX_STANDARD
    17
)

set(
    CMAKE_CXX_STANDARD_REQUIRED
    ON
)

set(
    CMAKE_CXX_EXTENSIONS
    OFF
)

option(
    GRAPH_BENCHMARK_ENABLE_WARNINGS_AS_ERRORS
    "Treat compiler warnings as errors."
    OFF
)

add_executable(
    random_graph_classification

    experiments/random_graph_classification/main.cpp

    core/graph/graph.cpp

    core/random_network_models/random_network_models_common.cpp
    core/random_network_models/random_network_models_registry.cpp

    core/random_network_models/models/barabasi_albert.cpp
    core/random_network_models/models/chung_lu.cpp
    core/random_network_models/models/configuration_model.cpp
    core/random_network_models/models/erdos_renyi.cpp
    core/random_network_models/models/exponential_random_graph_model.cpp
    core/random_network_models/models/girg.cpp
    core/random_network_models/models/hyperbolic_random_graph.cpp
    core/random_network_models/models/kleinberg.cpp
    core/random_network_models/models/stochastic_block_model.cpp
    core/random_network_models/models/watts_strogatz.cpp

    core/utils/portable_random.cpp
    core/utils/seed_derivation.cpp

    methods/TDA/persistent_homology/diagrams.cpp
    methods/TDA/persistent_homology/persistence.cpp

    methods/TDA/higher_order_persistence_diagrams/higher_order_persistence_diagrams.cpp

    methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/aggregation.cpp
    methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/random_fourier_features.cpp

    methods/TDA/higher_order_persistence_diagrams/wasserstein_distance/hungarian_algorithm.cpp
    methods/TDA/higher_order_persistence_diagrams/wasserstein_distance/wasserstein_distance.cpp

    experiments/random_graph_classification/configuration/command_line.cpp
    experiments/random_graph_classification/configuration/experiment_config.cpp

    experiments/random_graph_classification/experiment_design/batch_size_sensitivity.cpp
    experiments/random_graph_classification/experiment_design/benchmark.cpp
    experiments/random_graph_classification/experiment_design/harmonic_aggregation_ablation.cpp
    experiments/random_graph_classification/experiment_design/rff_sensitivity.cpp
    experiments/random_graph_classification/experiment_design/robustness.cpp
    experiments/random_graph_classification/experiment_design/scalability.cpp

    experiments/random_graph_classification/execution/batch_size_sensitivity_runner.cpp
    experiments/random_graph_classification/execution/benchmark.cpp
    experiments/random_graph_classification/execution/execution_plan_runner.cpp
    experiments/random_graph_classification/execution/harmonic_aggregation_ablation_runner.cpp
    experiments/random_graph_classification/execution/rff_sensitivity_runner.cpp
    experiments/random_graph_classification/execution/robustness_runner.cpp
    experiments/random_graph_classification/execution/scalability_runner.cpp

    experiments/random_graph_classification/output/csv_writer.cpp
    experiments/random_graph_classification/output/experiment_output.cpp
    experiments/random_graph_classification/output/metadata_writer.cpp
    experiments/random_graph_classification/output/output_layout.cpp
)

add_executable(
    real_world_data

    experiments/real_world_data/main.cpp
    experiments/real_world_data/tu_dataset.cpp

    core/graph/graph.cpp

    core/utils/portable_random.cpp
    core/utils/seed_derivation.cpp

    methods/TDA/persistent_homology/diagrams.cpp
    methods/TDA/persistent_homology/persistence.cpp

    methods/TDA/higher_order_persistence_diagrams/higher_order_persistence_diagrams.cpp

    methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/aggregation.cpp
    methods/TDA/higher_order_persistence_diagrams/harmonic_aggregation/random_fourier_features.cpp

    experiments/random_graph_classification/output/csv_writer.cpp
)

target_include_directories(
    random_graph_classification
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    real_world_data
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(
    random_graph_classification
    PRIVATE
        cxx_std_17
)

target_compile_features(
    real_world_data
    PRIVATE
        cxx_std_17
)

target_compile_definitions(
    random_graph_classification
    PRIVATE
        GRAPH_BENCHMARK_BUILD_TYPE="$<CONFIG>"
)

target_compile_definitions(
    real_world_data
    PRIVATE
        GRAPH_BENCHMARK_BUILD_TYPE="$<CONFIG>"
)

if(
    MSVC
)
    target_compile_options(
        random_graph_classification
        PRIVATE
            /W4
            /permissive-
            /EHsc
            /Zc:__cplusplus
    )

    target_compile_options(
        real_world_data
        PRIVATE
            /W4
            /permissive-
            /EHsc
            /Zc:__cplusplus
    )

    if(
        GRAPH_BENCHMARK_ENABLE_WARNINGS_AS_ERRORS
    )
        target_compile_options(
            random_graph_classification
            PRIVATE
                /WX
        )

        target_compile_options(
            real_world_data
            PRIVATE
                /WX
        )
    endif()
else()
    target_compile_options(
        random_graph_classification
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wshadow
    )

    target_compile_options(
        real_world_data
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wshadow
    )

    if(
        GRAPH_BENCHMARK_ENABLE_WARNINGS_AS_ERRORS
    )
        target_compile_options(
            random_graph_classification
            PRIVATE
                -Werror
        )

        target_compile_options(
            real_world_data
            PRIVATE
                -Werror
        )
    endif()
endif()

set_target_properties(
    random_graph_classification
    PROPERTIES
        OUTPUT_NAME
            "random_graph_classification"
        RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    real_world_data
    PROPERTIES
        OUTPUT_NAME
            "real_world_data"
        RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/bin"
)

install(
    TARGETS
        random_graph_classification
        real_world_data
    RUNTIME DESTINATION
        bin
)