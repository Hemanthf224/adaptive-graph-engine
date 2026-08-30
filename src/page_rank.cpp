#include "algorithms/page_rank.hpp"
#include <omp.h>
#include <numeric>

namespace graph_engine {
namespace algorithms {

std::vector<float> pagerank_sequential(const core::CSRGraph& graph, int iterations, float damping) {
    size_t V = graph.num_vertices;
    std::vector<float> scores(V, 1.0f / V);
    std::vector<float> next_scores(V, 0.0f);
    float base_score = (1.0f - damping) / V;

    for (int iter = 0; iter < iterations; ++iter) {
        // Reset next_scores to base_score
        std::fill(next_scores.begin(), next_scores.end(), base_score);

        // Push scores to neighbors
        for (core::vertex_id_t u = 0; u < V; ++u) {
            core::edge_id_t start = graph.row_offsets[u];
            core::edge_id_t end = graph.row_offsets[u + 1];
            core::edge_id_t degree = end - start;

            if (degree > 0) {
                float push_val = damping * (scores[u] / degree);
                for (core::edge_id_t e = start; e < end; ++e) {
                    core::vertex_id_t v = graph.column_indices[e];
                    next_scores[v] += push_val;
                }
            } else {
                // Handling dangling nodes (nodes with 0 out-degree)
                // In standard PR, their score is distributed equally to everyone.
                // For simplicity, we'll just let them leak score, or we can add it to all.
                // We'll ignore dangling nodes here for standard benchmarking.
            }
        }
        scores = next_scores;
    }

    return scores;
}

std::vector<float> pagerank_openmp(const core::CSRGraph& graph, int iterations, float damping) {
    size_t V = graph.num_vertices;
    std::vector<float> scores(V, 1.0f / V);
    std::vector<float> next_scores(V, 0.0f);
    float base_score = (1.0f - damping) / V;

    for (int iter = 0; iter < iterations; ++iter) {
        
        #pragma omp parallel for
        for (core::vertex_id_t i = 0; i < V; ++i) {
            next_scores[i] = base_score;
        }

        #pragma omp parallel for schedule(dynamic, 1024)
        for (core::vertex_id_t u = 0; u < V; ++u) {
            core::edge_id_t start = graph.row_offsets[u];
            core::edge_id_t end = graph.row_offsets[u + 1];
            core::edge_id_t degree = end - start;

            if (degree > 0) {
                float push_val = damping * (scores[u] / degree);
                for (core::edge_id_t e = start; e < end; ++e) {
                    core::vertex_id_t v = graph.column_indices[e];
                    #pragma omp atomic
                    next_scores[v] += push_val;
                }
            }
        }
        scores = next_scores;
    }

    return scores;
}

} // namespace algorithms
} // namespace graph_engine
