#include <emscripten/bind.h>
#include "algorithms/page_rank.hpp"
#include "algorithms/bfs.hpp"
#include "core/graph.hpp"

using namespace emscripten;
using namespace graph_engine;

// WebAssembly Wrapper for CSRGraph
class WasmGraph {
public:
    core::CSRGraph graph;

    WasmGraph(int num_vertices) {
        graph.num_vertices = num_vertices;
        graph.row_offsets.resize(num_vertices + 1, 0);
    }

    void add_edge(int u, int v) {
        // Simplified for Wasm demo - in reality, CSR needs batch building
        // For dynamic insertion in Wasm, we would use the DynamicGraph structure.
        graph.num_edges++;
        graph.column_indices.push_back(v);
        for(int i = u + 1; i <= graph.num_vertices; i++) {
            graph.row_offsets[i]++;
        }
    }
    
    int get_num_vertices() const { return graph.num_vertices; }
    int get_num_edges() const { return graph.num_edges; }
};

// WebAssembly Analytics Runner
std::vector<float> wasm_pagerank(const WasmGraph& wgraph, int iterations) {
    return algorithms::pagerank_sequential(wgraph.graph, iterations, 0.85f);
}

std::vector<int32_t> wasm_bfs(const WasmGraph& wgraph, int source) {
    return algorithms::bfs_sequential(wgraph.graph, source);
}

// Bind C++ constructs to JavaScript
EMSCRIPTEN_BINDINGS(adaptive_graph_module) {
    class_<WasmGraph>("Graph")
        .constructor<int>()
        .function("add_edge", &WasmGraph::add_edge)
        .function("get_num_vertices", &WasmGraph::get_num_vertices)
        .function("get_num_edges", &WasmGraph::get_num_edges);

    function("pagerank", &wasm_pagerank);
    function("bfs", &wasm_bfs);
    
    // Register std::vector<float> and std::vector<int> for JS interop
    register_vector<float>("VectorFloat");
    register_vector<int32_t>("VectorInt");
}
