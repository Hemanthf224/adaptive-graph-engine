#include "core/io.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace graph_engine {
namespace core {

void save_graph_binary(const CSRGraph& graph, const std::string& filepath) {
    std::string bin_filepath = filepath + ".bin";
    std::cout << "[IO] Saving binary graph cache to: " << bin_filepath << "..." << std::endl;
    
    FILE* f = std::fopen(bin_filepath.c_str(), "wb");
    if (!f) throw std::runtime_error("Could not create binary file: " + bin_filepath);

    // Write header
    std::fwrite(&graph.num_vertices, sizeof(vertex_id_t), 1, f);
    std::fwrite(&graph.num_edges, sizeof(edge_id_t), 1, f);

    // Write arrays
    std::fwrite(graph.row_offsets.data(), sizeof(edge_id_t), graph.num_vertices + 1, f);
    std::fwrite(graph.column_indices.data(), sizeof(vertex_id_t), graph.num_edges, f);
    std::fwrite(graph.edge_weights.data(), sizeof(float), graph.num_edges, f);

    std::fclose(f);
    std::cout << "[IO] Binary cache saved successfully." << std::endl;
}

CSRGraph load_graph_binary(const std::string& filepath) {
    std::cout << "[IO] Loading binary graph from cache: " << filepath << "..." << std::endl;
    
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) throw std::runtime_error("Could not open binary file: " + filepath);

    CSRGraph graph;
    
    // Read header
    if (std::fread(&graph.num_vertices, sizeof(vertex_id_t), 1, f) != 1) throw std::runtime_error("IO Error");
    if (std::fread(&graph.num_edges, sizeof(edge_id_t), 1, f) != 1) throw std::runtime_error("IO Error");

    // Allocate RAM
    graph.row_offsets.resize(graph.num_vertices + 1);
    graph.column_indices.resize(graph.num_edges);
    graph.edge_weights.resize(graph.num_edges);
    graph.is_weighted = true;

    // Read raw bytes straight into RAM (Instantaneous)
    if (std::fread(graph.row_offsets.data(), sizeof(edge_id_t), graph.num_vertices + 1, f) != (graph.num_vertices + 1)) {
        throw std::runtime_error("IO Error reading row_offsets");
    }
    
    if (std::fread(graph.column_indices.data(), sizeof(vertex_id_t), graph.num_edges, f) != graph.num_edges) {
        throw std::runtime_error("IO Error reading column_indices");
    }

    if (std::fread(graph.edge_weights.data(), sizeof(float), graph.num_edges, f) != graph.num_edges) {
        throw std::runtime_error("IO Error reading edge_weights");
    }

    std::fclose(f);
    std::cout << "[SUCCESS] Binary graph loaded instantly. V=" << graph.num_vertices << " E=" << graph.num_edges << std::endl;
    return graph;
}

CSRGraph load_graph(const std::string& filepath) {
    // 1. Check if a binary cache exists
    std::string bin_filepath = filepath + ".bin";
    FILE* test_f = std::fopen(bin_filepath.c_str(), "rb");
    if (test_f) {
        std::fclose(test_f);
        return load_graph_binary(bin_filepath);
    }

    // 2. If no cache, perform the slow ASCII parsing
    std::cout << "[IO] No binary cache found for " << filepath << ". Falling back to ASCII parsing..." << std::endl;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string line;
    vertex_id_t max_vertex_id = 0;
    edge_id_t edge_count = 0;

    std::cout << "Pass 1: Discovering graph dimensions..." << std::endl;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%' || line[0] == '#') continue;
        std::stringstream ss(line);
        vertex_id_t u, v;
        if (ss >> u >> v) {
            max_vertex_id = std::max({max_vertex_id, u, v});
            edge_count++;
        }
    }

    CSRGraph graph;
    graph.num_vertices = max_vertex_id + 1;
    graph.num_edges = edge_count;

    std::vector<edge_id_t> degrees(graph.num_vertices, 0);

    std::cout << "Pass 2: Counting degrees..." << std::endl;
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

    graph.row_offsets.resize(graph.num_vertices + 1, 0);
    for (vertex_id_t i = 0; i < graph.num_vertices; ++i) {
        graph.row_offsets[i + 1] = graph.row_offsets[i] + degrees[i];
    }

    graph.column_indices.resize(graph.num_edges);
    graph.edge_weights.resize(graph.num_edges);
    graph.is_weighted = true;
    std::vector<edge_id_t> current_offsets(graph.row_offsets.begin(), graph.row_offsets.end());

    std::cout << "Pass 3: Populating edges and weights..." << std::endl;
    file.clear();
    file.seekg(0, std::ios::beg);

    // simple seeded random for consistent edge weights if dataset lacks them
    srand(42); 

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%' || line[0] == '#') continue;
        std::stringstream ss(line);
        vertex_id_t u, v;
        float w;
        if (ss >> u >> v) {
            edge_id_t pos = current_offsets[u]++;
            graph.column_indices[pos] = v;
            
            if (ss >> w) {
                graph.edge_weights[pos] = w;
            } else {
                // Generate random weight between 1.0 and 10.0
                graph.edge_weights[pos] = 1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 9.0f));
            }
        }
    }

    std::cout << "ASCII Graph loaded successfully. V=" << graph.num_vertices << " E=" << graph.num_edges << std::endl;
    
    // 3. Save the cache for next time
    save_graph_binary(graph, filepath);
    
    return graph;
}

} // namespace core
} // namespace graph_engine
