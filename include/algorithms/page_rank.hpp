#pragma once

#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// CPU Sequential PageRank
std::vector<float> pagerank_sequential(const core::CSRGraph& graph, int iterations = 20, float damping = 0.85f);

// Multi-Core OpenMP PageRank
std::vector<float> pagerank_openmp(const core::CSRGraph& graph, int iterations = 20, float damping = 0.85f);

// Multi-Node Distributed MPI PageRank (1D Vertex Cut)
std::vector<float> pagerank_mpi(const core::CSRGraph& graph, int iterations = 20, float damping = 0.85f);

} // namespace algorithms
} // namespace graph_engine
