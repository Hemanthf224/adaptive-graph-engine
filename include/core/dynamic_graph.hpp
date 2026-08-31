#pragma once
#include "core/graph.hpp"
#include <vector>
#include <shared_mutex>
#include <omp.h>

namespace graph_engine {
namespace core {

// A dynamic graph that supports O(1) amortized edge insertions
class DynamicGraph {
public:
    size_t num_vertices;
    std::atomic<size_t> num_edges;
    
    // Instead of a flat CSR array, we use an array of dynamic vectors
    // This allows fast insertions but has higher memory overhead than CSR
    std::vector<std::vector<vertex_id_t>> adjacency_list;
    
    // Read-Write lock for safe streaming ingestion
    mutable std::shared_mutex rw_lock;

    DynamicGraph(size_t vertices) : num_vertices(vertices), num_edges(0) {
        adjacency_list.resize(vertices);
    }

    void add_edge(vertex_id_t u, vertex_id_t v) {
        if (u >= num_vertices || v >= num_vertices) return;
        
        std::unique_lock<std::shared_mutex> lock(rw_lock);
        adjacency_list[u].push_back(v);
        num_edges++;
    }

    std::vector<vertex_id_t> get_neighbors(vertex_id_t u) const {
        if (u >= num_vertices) return {};
        
        std::shared_lock<std::shared_mutex> lock(rw_lock);
        return adjacency_list[u];
    }
    
    // Converts the dynamic graph into a static CSR graph for ultra-fast analytics
    CSRGraph freeze() const {
        std::shared_lock<std::shared_mutex> lock(rw_lock);
        CSRGraph static_graph;
        static_graph.num_vertices = num_vertices;
        static_graph.num_edges = num_edges.load();
        
        static_graph.row_offsets.resize(num_vertices + 1, 0);
        static_graph.column_indices.reserve(static_graph.num_edges);
        
        core::edge_id_t current_offset = 0;
        for (size_t i = 0; i < num_vertices; ++i) {
            static_graph.row_offsets[i] = current_offset;
            const auto& neighbors = adjacency_list[i];
            static_graph.column_indices.insert(static_graph.column_indices.end(), neighbors.begin(), neighbors.end());
            current_offset += neighbors.size();
        }
        static_graph.row_offsets[num_vertices] = current_offset;
        
        return static_graph;
    }
};

} // namespace core
} // namespace graph_engine
