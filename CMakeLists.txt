cmake_minimum_required(VERSION 3.20)
project(toy-lang VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CPM)

# --- Dependencies ---

# GoogleTest
# CPMAddPackage(
#   NAME googletest
#   GITHUB_REPOSITORY google/googletest
#   VERSION 1.14.0
#   OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON"
# )

# Google Benchmark
# CPMAddPackage(
#   NAME benchmark
#   GITHUB_REPOSITORY google/benchmark
#   VERSION 1.8.3
#   OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_INSTALL_DOCS OFF" "BENCHMARK_ENABLE_GTEST_TESTS OFF"
# )

# MLIR / LLVM
# Note: Users must set MLIR_DIR (or LLVM_DIR) to the installation path of MLIR/LLVM
# e.g. cmake -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir ..
find_package(MLIR CONFIG REQUIRED)
message(STATUS "Found MLIR: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# --- Subdirectories ---
add_subdirectory(src)
# add_subdirectory(tests)
# add_subdirectory(benchmarks)
