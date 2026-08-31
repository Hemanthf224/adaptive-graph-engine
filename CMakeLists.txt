cmake_minimum_required(VERSION 3.24)
project(AdaptiveGraphEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Set common compiler flags for optimizations and warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -mavx2")
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
