#pragma once

#include "core/graph.hpp"
#include <vector>
#include <cstdint>

namespace graph_engine {
namespace algorithms {

// Top-Down Breadth-First Search (CUDA Parallel)
// Explicitly copies the graph to the GPU, runs a level-synchronous 
// CUDA kernel, and copies the resulting distances back to the host.
// Returns a vector of distances from the source vertex.
std::vector<int32_t> bfs_cuda(const core::CSRGraph& host_graph, core::vertex_id_t source);

} // namespace algorithms
} // namespace graph_engine
