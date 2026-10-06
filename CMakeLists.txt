cmake_minimum_required(VERSION 4.4)

project(VoidCLcompute
    VERSION 0.1.0
    DESCRIPTION "Lightweight OpenCL wrapper for elementwise GPU array math"
    LANGUAGES CXX
)

# ------------------------------------------------------------
# C++ configuration
# ------------------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

    set_property(
        CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS
            Debug
            Release
            RelWithDebInfo
            MinSizeRel
    )
endif()

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

option(
    VOIDCL_BUILD_BENCHMARK
    "Build the CPU-vs-GPU benchmark example"
    ON
)

option(
    VOIDCL_ENABLE_AVX2
    "Compile with AVX2 for CPU-side benchmark math"
    ON
)

# ------------------------------------------------------------
# OpenCL
# ------------------------------------------------------------

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL")
    list(
        APPEND
        CMAKE_PREFIX_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL"
    )
endif()

find_package(OpenCL REQUIRED)

# ------------------------------------------------------------
# IPO / LTO
# ------------------------------------------------------------

include(CheckIPOSupported)

check_ipo_supported(
    RESULT VOIDCL_IPO_SUPPORTED
    OUTPUT VOIDCL_IPO_MESSAGE
    LANGUAGES CXX
)

function(voidcl_configure_target target)

    if(VOIDCL_ENABLE_AVX2)

        if(MSVC)
            target_compile_options(
                ${target}
                PRIVATE
                    /arch:AVX2
            )
        else()
            target_compile_options(
                ${target}
                PRIVATE
                    -mavx2
            )
        endif()

    endif()

    if(VOIDCL_IPO_SUPPORTED)
        set_target_properties(
            ${target}
            PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
        )
    endif()

endfunction()

# ------------------------------------------------------------
# VoidCLcompute library
# ------------------------------------------------------------

add_library(
    VoidCLcompute
    SHARED
        src/VoidCLcompute.cpp
        include/VoidCLcompute.h
)

target_include_directories(
    VoidCLcompute
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    VoidCLcompute
    PRIVATE
        OpenCL::OpenCL
)

set_target_properties(
    VoidCLcompute
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

voidcl_configure_target(VoidCLcompute)

# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

if(VOIDCL_BUILD_BENCHMARK)

    find_package(Threads REQUIRED)

    add_executable(
        benchmark
        examples/benchmark/app.cpp
    )

    target_link_libraries(
        benchmark
        PRIVATE
            VoidCLcompute
            Threads::Threads
    )

    voidcl_configure_target(benchmark)

endif()

# ------------------------------------------------------------
# Installation
# ------------------------------------------------------------

include(GNUInstallDirs)

# DLL + LIB on Windows
# SO + import/static libraries on other platforms
install(
    TARGETS VoidCLcompute
    EXPORT VoidCLcomputeTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Public header
install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/VoidCLcompute.h
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export CMake target information
install(
    EXPORT VoidCLcomputeTargets
    FILE
        VoidCLcomputeTargets.cmake
    NAMESPACE
        VoidCL::
    DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/VoidCLcompute
)
