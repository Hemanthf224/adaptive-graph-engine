#pragma once

#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// CUDA PageRank
// Runs the algorithm entirely in VRAM for the specified number of iterations
std::vector<float> pagerank_cuda(const core::CSRGraph& host_graph, int iterations = 20, float damping = 0.85f);

} // namespace algorithms
} // namespace graph_engine
