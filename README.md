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

Tested on the **Stanford SNAP** datasets. Hardware: Intel CPU (24 Threads) / NVIDIA RTX 5060 Laptop GPU (8GB VRAM).

### LiveJournal Social Network (4,847,571 Vertices, 68,993,773 Edges)
*PageRank (20 Iterations, Compute-Bound)*

| Hardware | Time (ms) | Speedup vs Seq |
|---|---|---|
| CPU Sequential | 2046.10 | 1.0x |
| CPU OpenMP (24 Threads) | **1070.97** | **1.91x** |
| GPU CUDA (UVM Zero-Copy) | 2654.01 | 0.77x |

### Amazon Product Co-purchasing Network (262,111 Vertices, 1,234,877 Edges)
*PageRank (20 Iterations, Compute-Bound)*

| Hardware | Time (ms) | Speedup vs OpenMP |
|---|---|---|
| CPU Sequential (L3 Cache hit) | **16.12** | **2.5x** |
| CPU OpenMP (24 Threads) | 40.77 | 1.0x |
| GPU CUDA (UVM Zero-Copy) | 25.20 | 1.6x |

## Experimental Setup & Limitations

> [!WARNING]
> **The High-Degree Vertex Contention Problem**

In our benchmark, CUDA Unified Memory successfully mitigated PCIe transfer latency via `cudaMemPrefetchAsync` warm-starts, easily fitting the 300MB LiveJournal graph into VRAM. However, the OpenMP CPU dramatically outperformed the CUDA GPU on the 68M Edge LiveJournal dataset.

**Why?** Power-law graphs (like Social Networks) contain incredibly high-degree vertices (e.g., users with millions of followers). During the GPU PageRank kernel, thousands of CUDA threads attempt to write to the exact same memory address simultaneously using `atomicAdd`. This causes catastrophic memory contention and warp serialization. Conversely, the CPU's sophisticated hierarchical L1/L2/L3 caches and branch predictors handle this uncoalesced memory access pattern far better. 

This proves that **GPU execution is not universally superior**—performance depends heavily on graph topology, atomic contention, and Unified Memory behavior.

## Resume Bullet Points
If you are discussing this project in a systems engineering interview, these are the recommended talking points:
- Engineered a high-performance heterogeneous graph processing engine in C++ and CUDA, implementing a Compressed Sparse Row (CSR) structure to optimize memory locality for large-scale graph analytics.
- Implemented lock-free parallel Breadth-First Search and PageRank algorithms utilizing OpenMP CPU threads and massively parallel CUDA GPU kernels with `atomicCAS` synchronization.
- Leveraged CUDA Unified Virtual Memory (UVM) and `cudaMallocManaged` to build an intelligent zero-copy memory allocator, successfully processing massive out-of-core graphs while mitigating PCIe transfer latency.
- Developed a full-stack technical telemetry dashboard using React, Recharts, and a FastAPI backend wrapped with `pybind11` to dynamically plot execution times and visualize VRAM allocation grids.

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
./src/graph_engine ../data/soc-LiveJournal1.txt --benchmark
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
