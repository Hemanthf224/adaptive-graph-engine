# Adaptive GPU-Accelerated Graph Processing Engine

A High-Performance Computing (HPC) Graph Processing Engine written in C++ and CUDA, designed specifically to optimize out-of-core graph analytics on consumer hardware. 

This engine implements an **Adaptive Heuristic Scheduler** that dynamically dispatches execution to Sequential CPU, OpenMP CPU, or CUDA GPU backends based on graph topology and memory boundaries.

## Features

- **Compressed Sparse Row (CSR):** Optimized contiguous memory layout for high-speed graph traversal.
- **Heterogeneous Execution:** Implements Graph Algorithms (BFS, PageRank) across three architectures:
  - Single-threaded CPU (Sequential)
  - Multi-threaded CPU (OpenMP `atomic` lock-free synchronization)
  - Massively Parallel GPU (CUDA `atomicCAS` and `atomicAdd`)
- **CUDA Unified Memory (UVM):** Overcomes the "Memory Wall" (VRAM limits on consumer GPUs) by utilizing `cudaMallocManaged` and a custom C++ `UVMAllocator` to page data dynamically between System DDR RAM and GPU VRAM without explicit PCI-E memory copies.
- **Adaptive Scheduler:** Heuristically analyzes graph degree and density to prevent GPU latency bottlenecks on memory-bound algorithms.
- **Python Interoperability:** Exposes the C++ engine to Python via `pybind11` for seamless Data Science integration.

## Architecture

1. **Core API (`graph_core`)**: Handles Matrix Market parsing (3-pass high-speed loader), CSR Structuring, and UVM Allocation.
2. **Algorithms API (`graph_algorithms`)**: Contains the `__global__` CUDA device kernels and OpenMP `#pragma` directives.
3. **Adaptive Engine**: The intelligent runtime that analyzes the loaded `CSRGraph` and automatically routes the calculation to the fastest hardware.

## Benchmarks

Tested on the **Amazon Product Co-purchasing Network** (262,111 Vertices, 1,234,877 Edges). Hardware: Intel CPU / NVIDIA RTX 5060 Laptop GPU.

### PageRank (20 Iterations, Compute-Bound)
*CUDA Unified Memory (Zero-Copy) completely eliminates the 130ms PCI-E bottleneck, allowing the GPU to crunch 20 iterations of floating-point math in just 25ms!*

| Hardware | Time (ms) | Speedup vs OpenMP |
|---|---|---|
| CPU Sequential (L3 Cache hit) | 16.12 | 2.5x |
| CPU OpenMP (24 Threads) | 40.77 | 1.0x |
| GPU CUDA (UVM Zero-Copy) | **25.20** | **1.6x** |

*(Note: Because the Amazon graph fits entirely inside the CPU's L3 Cache (~5MB), the Sequential CPU is incredibly fast as it avoids Main Memory latency. The GPU's massive GDDR6 bandwidth dominance will activate on out-of-core graphs >100M edges).*

## Building from Source

### Prerequisites
- CMake >= 3.20
- GCC with OpenMP support
- NVIDIA CUDA Toolkit (`nvcc`)
- OpenMPI (`libopenmpi-dev`)
- Python 3 Headers (`python3-dev`)

### Compilation

```bash
mkdir build
cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make -j
```

### Running the Engine

Run the benchmark suite against a dataset:
```bash
./src/graph_engine ../data/amazon0302.txt --benchmark
```

Run the automated GTest suite to verify mathematical accuracy across hardware architectures:
```bash
ctest --output-on-failure
```

Drive the engine via Python:
```bash
python3 test_python_api.py
```

## Contributors

- **Hemanth Reddy** ([@Hemanthf224](https://github.com/Hemanthf224)) - Lead Engineer & Architect
