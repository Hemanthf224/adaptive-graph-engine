#pragma once

#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// CUDA PageRank (UVM)
// Runs the algorithm using CUDA Unified Memory page faulting
std::vector<float> pagerank_cuda(const core::CSRGraph& host_graph, int iterations = 20, float damping = 0.85f);

// CUDA PageRank (Explicit Copy)
// Runs the algorithm by manually copying the entire CSR graph across the PCI-E bus
std::vector<float> pagerank_cuda_explicit(const core::CSRGraph& host_graph, int iterations = 20, float damping = 0.85f);

// Warm-up function to force PCI-E data transfer into VRAM before benchmarking
void prefetch_graph_to_gpu(const core::CSRGraph& graph, int device_id = 0);

} // namespace algorithms
} // namespace graph_engine
