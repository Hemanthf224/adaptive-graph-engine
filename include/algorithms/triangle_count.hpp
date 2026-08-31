#pragma once

#include "core/graph.hpp"
#include "core/profiler.hpp"
#include <vector>
#include <omp.h>
#include <iostream>

namespace graph_engine {
namespace algorithms {

/**
 * @brief Computes the number of triangles in the graph using CPU OpenMP.
 * 
 * Triangle counting is a core graph mining algorithm used to calculate clustering coefficients.
 * This implementation iterates over all edges (u, v) where u < v, and computes the intersection
 * of their neighborhood sets to find common neighbors (triangles).
 * 
 * @param graph The input CSRGraph to analyze.
 * @return The total number of triangles in the graph.
 */
inline uint64_t triangle_counting_openmp(const core::CSRGraph& graph) {
    PROFILE_FUNCTION();
    uint64_t total_triangles = 0;
    size_t V = graph.num_vertices;

    std::cout << "[ALGO] Starting OpenMP Triangle Counting on " << V << " vertices...\n";

    #pragma omp parallel for reduction(+:total_triangles) schedule(dynamic, 64)
    for (core::vertex_id_t u = 0; u < V; ++u) {
        core::edge_id_t u_start = graph.row_offsets[u];
        core::edge_id_t u_end = graph.row_offsets[u + 1];

        for (core::edge_id_t e = u_start; e < u_end; ++e) {
            core::vertex_id_t v = graph.column_indices[e];
            
            // To avoid overcounting, only consider edges where u < v
            if (u >= v) continue;

            core::edge_id_t v_start = graph.row_offsets[v];
            core::edge_id_t v_end = graph.row_offsets[v + 1];

            // Intersection of neighborhoods N(u) and N(v)
            core::edge_id_t i = u_start;
            core::edge_id_t j = v_start;

            while (i < u_end && j < v_end) {
                core::vertex_id_t nu = graph.column_indices[i];
                core::vertex_id_t nv = graph.column_indices[j];

                if (nu == nv) {
                    // Avoid counting if nu <= v (guarantees u < v < nu uniqueness)
                    if (v < nu) {
                        total_triangles++;
                    }
                    i++;
                    j++;
                } else if (nu < nv) {
                    i++;
                } else {
                    j++;
                }
            }
        }
    }

    return total_triangles;
}

} // namespace algorithms
} // namespace graph_engine
