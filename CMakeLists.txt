cmake_minimum_required(VERSION 3.24)
project(AdaptiveGraphEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Compiler Flags (C++17, O3 Optimization, OpenMP, AVX2 SIMD)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fopenmp -mavx2")

# Code Coverage Option
option(COVERAGE "Enable code coverage reporting" OFF)
if(COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Target the native GPU architecture to prevent PTX JIT compilation errors
set(CMAKE_CUDA_ARCHITECTURES native)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP CXX")
endif()

# Find MPI
find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    message(STATUS "Found MPI CXX")
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Include directories for the whole project
include_directories(${CMAKE_SOURCE_DIR}/include)


# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# ---------------------------------------------------------
# PyBind11 Native Python Interface
# ---------------------------------------------------------
# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Ensure Python3 headers are found
find_package(Python3 COMPONENTS Development.Module REQUIRED)

pybind11_add_module(adaptive_graph
    src/python_bindings.cpp
    src/io.cpp
    src/page_rank.cpp
    src/bfs.cpp
    src/connected_components.cpp
    src/sssp.cpp
    src/query_parser.cpp
    src/gnn_layer.cpp
    src/quantum_simulator.cpp
    src/fhe_simulator.cpp
    src/zkp_verifier.cpp
    src/differential_privacy.cpp
    src/federated_learning.cpp
)

# Link requirements for Python Module
target_link_libraries(adaptive_graph PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(adaptive_graph PRIVATE ${CMAKE_SOURCE_DIR}/include)

