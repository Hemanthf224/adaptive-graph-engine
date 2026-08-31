#include "algorithms/page_rank.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

namespace graph_engine {
namespace algorithms {

std::vector<float> pagerank_mpi(const core::CSRGraph& graph, int iterations, float damping) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    size_t V = graph.num_vertices;
    float base_score = (1.0f - damping) / V;
    
    // Global scores (maintained by all nodes)
    std::vector<float> global_scores(V, 1.0f / V);
    // Local scores computed by this specific MPI rank
    std::vector<float> local_next_scores(V, 0.0f);
    // Global aggregated next scores
    std::vector<float> global_next_scores(V, 0.0f);

    // 1D Vertex Partitioning: Determine which chunk of vertices this rank owns
    size_t chunk_size = (V + mpi_size - 1) / mpi_size;
    size_t start_v = std::min((size_t)(mpi_rank * chunk_size), V);
    size_t end_v = std::min(start_v + chunk_size, V);

    if (mpi_rank == 0) {
        std::cout << "[MPI_CLUSTER] Distributing " << V << " vertices across " << mpi_size << " nodes.\n";
        std::cout << "[MPI_CLUSTER] Node " << mpi_rank << " handling vertices " << start_v << " to " << end_v << ".\n";
    }

    // Barrier to ensure all nodes are ready to begin the distributed algorithm
    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < iterations; ++iter) {
        // Reset local scores
        std::fill(local_next_scores.begin(), local_next_scores.end(), 0.0f);
        
        // Add the base teleportation score for the vertices we own
        for (size_t u = start_v; u < end_v; ++u) {
            local_next_scores[u] = base_score;
        }

        // Compute Push PageRank ONLY for the vertices assigned to this MPI Rank
        for (size_t u = start_v; u < end_v; ++u) {
            core::edge_id_t start = graph.row_offsets[u];
            core::edge_id_t end = graph.row_offsets[u + 1];
            core::edge_id_t degree = end - start;

            if (degree > 0) {
                float push_val = damping * (global_scores[u] / (float)degree);
                for (core::edge_id_t e = start; e < end; ++e) {
                    core::vertex_id_t v = graph.column_indices[e];
                    local_next_scores[v] += push_val;
                }
            }
        }

        // Distributed Synchronization: Combine all local next_scores into global_next_scores over the network
        MPI_Allreduce(local_next_scores.data(), global_next_scores.data(), V, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // Update the global scores for the next iteration
        global_scores = global_next_scores;
    }

    return global_scores;
}

} // namespace algorithms
} // namespace graph_engine
