#include "core/io.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace graph_engine {
namespace core {

CSRGraph load_graph(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string line;
    vertex_id_t max_vertex_id = 0;
    edge_id_t edge_count = 0;

    std::cout << "Pass 1: Discovering graph dimensions..." << std::endl;
    // Pass 1: Find max vertex ID to size our arrays
    while (std::getline(file, line)) {
        // Skip comments (Matrix Market uses %)
        if (line.empty() || line[0] == '%' || line[0] == '#') continue;

        std::stringstream ss(line);
        vertex_id_t u, v;
        if (ss >> u >> v) {
            // Some formats are 1-indexed, we assume 0-indexed for simplicity.
            // If the max ID is N, we need N+1 vertices.
            max_vertex_id = std::max({max_vertex_id, u, v});
            edge_count++;
        }
    }

    CSRGraph graph;
    graph.num_vertices = max_vertex_id + 1;
    graph.num_edges = edge_count;

    // Initialize degrees array
    std::vector<edge_id_t> degrees(graph.num_vertices, 0);

    std::cout << "Pass 2: Counting degrees..." << std::endl;
    // Reset file for Pass 2
    file.clear();
    file.seekg(0, std::ios::beg);

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%' || line[0] == '#') continue;
        std::stringstream ss(line);
        vertex_id_t u, v;
        if (ss >> u >> v) {
            degrees[u]++;
        }
    }

    // Prefix sum to compute row_offsets
    graph.row_offsets.resize(graph.num_vertices + 1, 0);
    for (vertex_id_t i = 0; i < graph.num_vertices; ++i) {
        graph.row_offsets[i + 1] = graph.row_offsets[i] + degrees[i];
    }

    // Allocate column_indices
    graph.column_indices.resize(graph.num_edges);

    // We will use 'current_offsets' to keep track of where to insert the next edge for each vertex
    std::vector<edge_id_t> current_offsets(graph.row_offsets.begin(), graph.row_offsets.end());

    std::cout << "Pass 3: Populating edges..." << std::endl;
    // Reset file for Pass 3
    file.clear();
    file.seekg(0, std::ios::beg);

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%' || line[0] == '#') continue;
        std::stringstream ss(line);
        vertex_id_t u, v;
        if (ss >> u >> v) {
            edge_id_t pos = current_offsets[u]++;
            graph.column_indices[pos] = v;
        }
    }

    std::cout << "Graph loaded successfully. V=" << graph.num_vertices << " E=" << graph.num_edges << std::endl;
    return graph;
}

} // namespace core
} // namespace graph_engine
