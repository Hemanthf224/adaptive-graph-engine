#pragma once

#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

/**
 * @brief Computes the PageRank of all vertices in a graph using a single CPU core.
 * 
 * @param graph The input CSRGraph to analyze.
 * @param iterations The number of power iteration steps to perform.
 * @param damping The damping factor (probability of clicking a link vs teleporting).
 * @return A std::vector of float containing the PageRank score for each vertex.
 */
std::vector<float> pagerank_sequential(const core::CSRGraph& graph, int iterations = 20, float damping = 0.85f);

/**
 * @brief Computes the PageRank of all vertices utilizing multi-core Thread Pools (OpenMP).
 * 
 * Employs std::atomic<float> to perform lock-free synchronization across threads when 
 * updating vertex scores. Exhibits strong scaling according to Amdahl's Law.
 * 
 * @param graph The input CSRGraph to analyze.
 * @param iterations The number of power iteration steps to perform.
 * @param damping The damping factor (probability of clicking a link vs teleporting).
 * @return A std::vector of float containing the PageRank score for each vertex.
 */
std::vector<float> pagerank_openmp(const core::CSRGraph& graph, int iterations = 20, float damping = 0.85f);

/**
 * @brief Computes the PageRank across a distributed cluster utilizing MPI.
 * 
 * The algorithm uses a 1D Vertex Partitioning scheme, where each physical node processes
 * a disjoint subset of vertices. `MPI_Allreduce` is used to synchronize the global probability
 * vector at the end of each iteration.
 * 
 * @param graph The input CSRGraph to analyze (Only required fully on Rank 0).
 * @param iterations The number of power iteration steps to perform.
 * @param damping The damping factor (probability of clicking a link vs teleporting).
 * @return A std::vector of float containing the PageRank score for each vertex (Valid on Rank 0).
 */
std::vector<float> pagerank_mpi(const core::CSRGraph& graph, int iterations = 20, float damping = 0.85f);

} // namespace algorithms
} // namespace graph_engine
