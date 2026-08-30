#pragma once

#include "core/graph.hpp"
#include <vector>
#include <cstdint>

namespace graph_engine {
namespace algorithms {

// Top-Down Breadth-First Search (Sequential)
// Returns a vector of distances from the source vertex.
// Unreachable vertices will have a distance of -1.
std::vector<int32_t> bfs_sequential(const core::CSRGraph& graph, core::vertex_id_t source);

// Top-Down Breadth-First Search (OpenMP Parallel)
// Returns a vector of distances from the source vertex.
std::vector<int32_t> bfs_openmp(const core::CSRGraph& graph, core::vertex_id_t source);

} // namespace algorithms
} // namespace graph_engine
