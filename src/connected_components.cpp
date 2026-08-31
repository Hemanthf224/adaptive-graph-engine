#include "algorithms/connected_components.hpp"
#include <omp.h>
#include <atomic>
#include <algorithm>

namespace graph_engine {
namespace algorithms {

std::vector<int> connected_components_sequential(const core::CSRGraph& graph) {
    size_t V = graph.num_vertices;
    std::vector<int> labels(V);
    
    // Initialize labels to vertex IDs
    for (size_t i = 0; i < V; ++i) {
        labels[i] = i;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t u = 0; u < V; ++u) {
            core::edge_id_t start = graph.row_offsets[u];
            core::edge_id_t end = graph.row_offsets[u + 1];

            for (core::edge_id_t e = start; e < end; ++e) {
                core::vertex_id_t v = graph.column_indices[e];
                // Propagate the smaller label
                if (labels[u] < labels[v]) {
                    labels[v] = labels[u];
                    changed = true;
                } else if (labels[v] < labels[u]) {
                    labels[u] = labels[v];
                    changed = true;
                }
            }
        }
    }

    return labels;
}

std::vector<int> connected_components_openmp(const core::CSRGraph& graph) {
    size_t V = graph.num_vertices;
    std::vector<std::atomic<int>> labels(V);
    
    #pragma omp parallel for
    for (size_t i = 0; i < V; ++i) {
        labels[i].store(i, std::memory_order_relaxed);
    }

    bool changed = true;
    while (changed) {
        changed = false;
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t u = 0; u < V; ++u) {
            core::edge_id_t start = graph.row_offsets[u];
            core::edge_id_t end = graph.row_offsets[u + 1];
            
            int label_u = labels[u].load(std::memory_order_relaxed);

            for (core::edge_id_t e = start; e < end; ++e) {
                core::vertex_id_t v = graph.column_indices[e];
                int label_v = labels[v].load(std::memory_order_relaxed);

                if (label_u < label_v) {
                    // Atomic Min for v
                    int current_v = label_v;
                    while (label_u < current_v && !labels[v].compare_exchange_weak(current_v, label_u, std::memory_order_relaxed)) {
                        // Loop until successful or condition fails
                    }
                    if (label_u < current_v) changed = true;
                } else if (label_v < label_u) {
                    // Atomic Min for u
                    int current_u = label_u;
                    while (label_v < current_u && !labels[u].compare_exchange_weak(current_u, label_v, std::memory_order_relaxed)) {
                        // Loop
                    }
                    if (label_v < current_u) changed = true;
                    // Update local label_u since it just shrank
                    label_u = labels[u].load(std::memory_order_relaxed);
                }
            }
        }
    }

    std::vector<int> final_labels(V);
    #pragma omp parallel for
    for (size_t i = 0; i < V; ++i) {
        final_labels[i] = labels[i].load(std::memory_order_relaxed);
    }
    return final_labels;
}

} // namespace algorithms
} // namespace graph_engine
