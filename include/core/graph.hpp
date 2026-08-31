#pragma once

#include <vector>
#include <iostream>
#include "types.hpp"
#include "uvm_allocator.hpp"

namespace graph_engine {
namespace core {

// Compressed Sparse Row (CSR) representation of a graph.
// Optimized for contiguous memory access on CPUs and GPUs.
struct CSRGraph {
public:
    // Number of vertices and edges
    size_t num_vertices = 0;
    size_t num_edges = 0;

    // CSR Arrays backed by Unified Memory (Accessible by both CPU and GPU!)
    uvm_vector<edge_id_t> row_offsets;
    uvm_vector<vertex_id_t> column_indices;

    // Optional: edge weights (if weighted graph)
    uvm_vector<weight_t> edge_weights;
    bool is_weighted = false;

    // Constructor
    CSRGraph() = default;

    // Helper to print graph stats
    void print() const;
};

// Creates a tiny hardcoded graph for testing and debugging
CSRGraph create_tiny_test_graph();

} // namespace core
} // namespace graph_engine
