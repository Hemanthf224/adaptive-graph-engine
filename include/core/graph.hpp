#pragma once

#include <vector>
#include <iostream>
#include "types.hpp"
#include "uvm_allocator.hpp"

namespace graph_engine {
namespace core {

/**
 * @struct CSRGraph
 * @brief Compressed Sparse Row (CSR) representation of a graph.
 * 
 * The CSRGraph is optimized for contiguous memory access on CPUs and GPUs.
 * It utilizes NVIDIA Unified Virtual Memory (UVM) via the `uvm_allocator`
 * to allow identical C++ STL containers to be accessed on the Host (CPU) 
 * and the Device (GPU) with zero-copy page faulting.
 */
struct CSRGraph {
public:
    /** @brief Total number of vertices in the graph. */
    size_t num_vertices = 0;
    
    /** @brief Total number of directed edges in the graph. */
    size_t num_edges = 0;

    /** 
     * @brief Row offsets array (Size: V + 1).
     * Backed by Unified Memory. Points to the starting index in `column_indices` for a vertex's neighbors.
     */
    uvm_vector<edge_id_t> row_offsets;
    
    /** 
     * @brief Column indices array (Size: E).
     * Backed by Unified Memory. Stores the destination vertex ID for each edge.
     */
    uvm_vector<vertex_id_t> column_indices;

    /** 
     * @brief Edge weights array (Size: E).
     * Backed by Unified Memory. Stores the floating point weight for each edge.
     */
    uvm_vector<weight_t> edge_weights;
    
    /** @brief Flag indicating if the graph contains valid edge weights. */
    bool is_weighted = false;

    /** @brief Default constructor. */
    CSRGraph() = default;

    /** 
     * @brief Helper utility to print graph statistics to stdout.
     */
    void print() const;
};

/**
 * @brief Creates a tiny hardcoded graph for testing and debugging.
 * @return A CSRGraph containing a 5-vertex toy topology.
 */
CSRGraph create_tiny_test_graph();

} // namespace core
} // namespace graph_engine
