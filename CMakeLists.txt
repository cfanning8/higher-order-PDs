cmake_minimum_required(VERSION 3.16)

project(
    NeurIPS2026
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Build type"
        FORCE
    )
endif()

# =========================
# Build metadata
# =========================

find_package(Git QUIET)

set(VPD_GIT_COMMIT "unknown")
set(VPD_GIT_DIRTY "unknown")

if(Git_FOUND)
    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}"
            rev-parse
            HEAD
        WORKING_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE
            VPD_GIT_COMMIT_RESULT
        OUTPUT_VARIABLE
            VPD_GIT_COMMIT_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(VPD_GIT_COMMIT_RESULT EQUAL 0)
        set(
            VPD_GIT_COMMIT
            "${VPD_GIT_COMMIT_OUTPUT}"
        )
    endif()

    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}"
            status
            --porcelain
        WORKING_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE
            VPD_GIT_DIRTY_RESULT
        OUTPUT_VARIABLE
            VPD_GIT_DIRTY_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(VPD_GIT_DIRTY_RESULT EQUAL 0)
        if(VPD_GIT_DIRTY_OUTPUT STREQUAL "")
            set(VPD_GIT_DIRTY "0")
        else()
            set(VPD_GIT_DIRTY "1")
        endif()
    endif()
endif()

file(
    SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/main_generate_classification_data.cpp"
    VPD_SOURCE_HASH
)

if(CMAKE_CONFIGURATION_TYPES)
    set(VPD_BUILD_TYPE "$<CONFIG>")
else()
    set(VPD_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
endif()

# =========================
# Compiler warnings
# =========================

function(vpd_enable_warnings target_name)
    if(MSVC)
        target_compile_options(
            "${target_name}"
            PRIVATE
                /W4
                /permissive-
        )
    else()
        target_compile_options(
            "${target_name}"
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()
endfunction()

# =========================
# Utils library
# =========================

add_library(
    utils
    utils/diagrams.cpp
    utils/aggregation.cpp
    utils/hungarian.cpp
    utils/persistence.cpp
    utils/wasserstein.cpp
    utils/diagrams.hpp
    utils/aggregation.hpp
    utils/hungarian.hpp
    utils/persistence.hpp
    utils/wasserstein.hpp
    utils/io.hpp
    utils/timing.hpp
)

target_include_directories(
    utils
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

vpd_enable_warnings(utils)

# =========================
# Data library
# =========================

add_library(
    data
    data/graph.cpp
    data/random_graph_models.cpp
    data/graph.hpp
    data/random_graph_models.hpp
)

target_include_directories(
    data
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

vpd_enable_warnings(data)

# =========================
# Classification-data executable
# =========================

add_executable(
    generate_classification_data
    main_generate_classification_data.cpp
)

target_link_libraries(
    generate_classification_data
    PRIVATE
        utils
        data
)

target_compile_definitions(
    generate_classification_data
    PRIVATE
        VPD_GIT_COMMIT="${VPD_GIT_COMMIT}"
        VPD_GIT_DIRTY="${VPD_GIT_DIRTY}"
        VPD_SOURCE_HASH="${VPD_SOURCE_HASH}"
        VPD_BUILD_TYPE="${VPD_BUILD_TYPE}"
        VPD_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        VPD_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

vpd_enable_warnings(generate_classification_data)

# =========================
# Tests
# =========================

add_executable(
    test_diagrams
    tests/test_diagrams.cpp
)

target_link_libraries(
    test_diagrams
    PRIVATE
        utils
)

vpd_enable_warnings(test_diagrams)

add_executable(
    test_aggregation
    tests/test_aggregation.cpp
)

target_link_libraries(
    test_aggregation
    PRIVATE
        utils
)

vpd_enable_warnings(test_aggregation)

add_executable(
    test_wasserstein
    tests/test_wasserstein.cpp
)

target_link_libraries(
    test_wasserstein
    PRIVATE
        utils
)

vpd_enable_warnings(test_wasserstein)

# =========================
# Testing
# =========================

enable_testing()

add_test(
    NAME diagrams_test
    COMMAND test_diagrams
)

add_test(
    NAME aggregation_test
    COMMAND test_aggregation
)

add_test(
    NAME wasserstein_test
    COMMAND test_wasserstein
)