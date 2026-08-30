#pragma once

#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// Sequential PageRank (Push-based)
std::vector<float> pagerank_sequential(const core::CSRGraph& graph, int iterations = 20, float damping = 0.85f);

// OpenMP Parallel PageRank (Push-based)
std::vector<float> pagerank_openmp(const core::CSRGraph& graph, int iterations = 20, float damping = 0.85f);

} // namespace algorithms
} // namespace graph_engine
