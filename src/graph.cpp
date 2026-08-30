#include "core/graph.hpp"
#include <iomanip>

namespace graph_engine {
namespace core {

void CSRGraph::print() const {
    std::cout << "--- CSR Graph ---\n";
    std::cout << "Vertices: " << num_vertices << "\n";
    std::cout << "Edges:    " << num_edges << "\n\n";

    std::cout << "Row Offsets:\n[ ";
    for (size_t i = 0; i < row_offsets.size(); ++i) {
        std::cout << row_offsets[i] << " ";
    }
    std::cout << "]\n\n";

    std::cout << "Column Indices (Edges):\n[ ";
    for (size_t i = 0; i < column_indices.size(); ++i) {
        std::cout << column_indices[i] << " ";
    }
    std::cout << "]\n\n";

    std::cout << "Adjacency List Interpretation:\n";
    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        std::cout << "Vertex " << std::setw(2) << v << " -> [ ";
        edge_id_t start = row_offsets[v];
        edge_id_t end = row_offsets[v+1];
        for (edge_id_t e = start; e < end; ++e) {
            std::cout << column_indices[e] << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "-----------------\n";
}

CSRGraph create_tiny_test_graph() {
    CSRGraph graph;
    
    // A simple directed graph with 5 vertices (0 to 4) and 6 edges
    // 0 -> 1, 2
    // 1 -> 3
    // 2 -> 1, 4
    // 3 -> 4
    // 4 -> (none)
    
    graph.num_vertices = 5;
    graph.num_edges = 6;
    
    // Row offsets:
    // v0 has 2 edges, so offsets are 0, 2
    // v1 has 1 edge,  so offsets are 2, 3
    // v2 has 2 edges, so offsets are 3, 5
    // v3 has 1 edge,  so offsets are 5, 6
    // v4 has 0 edges, so offsets are 6, 6
    graph.row_offsets = {0, 2, 3, 5, 6, 6};
    
    // Column indices (destinations):
    // v0's edges: 1, 2
    // v1's edges: 3
    // v2's edges: 1, 4
    // v3's edges: 4
    graph.column_indices = {1, 2, 3, 1, 4, 4};
    
    return graph;
}

} // namespace core
} // namespace graph_engine
