#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic std::vector <-> python list conversion
#include "core/graph.hpp"
#include "core/io.hpp"
#include "algorithms/page_rank.hpp"
#include "algorithms/bfs.hpp"
#include "algorithms/connected_components.hpp"
#include "algorithms/sssp.hpp"
#include "core/property_graph.hpp"
#include "core/query_parser.hpp"
#include "algorithms/gnn_layer.hpp"
#include "algorithms/quantum_simulator.hpp"

namespace py = pybind11;
using namespace graph_engine;

PYBIND11_MODULE(adaptive_graph, m) {
    m.doc() = "Adaptive Graph Engine - High Performance C++ Graph Analytics for Python";

    // Bind CSRGraph struct
    py::class_<core::CSRGraph>(m, "CSRGraph")
        .def(py::init<>())
        .def_readwrite("num_vertices", &core::CSRGraph::num_vertices)
        .def_readwrite("num_edges", &core::CSRGraph::num_edges)
        .def_readwrite("row_offsets", &core::CSRGraph::row_offsets)
        .def_readwrite("column_indices", &core::CSRGraph::column_indices)
        .def_readwrite("weights", &core::CSRGraph::weights)
        .def("__repr__",
             [](const core::CSRGraph &g) {
                 return "<adaptive_graph.CSRGraph with " + std::to_string(g.num_vertices) + " vertices and " + std::to_string(g.num_edges) + " edges>";
             });

    // Bind PropertyGraph
    py::class_<core::PropertyGraph, core::CSRGraph>(m, "PropertyGraph")
        .def(py::init<>())
        .def(py::init<const core::CSRGraph&>())
        .def("set_vertex_property", &core::PropertyGraph::set_vertex_property)
        .def("get_vertex_property", &core::PropertyGraph::get_vertex_property)
        .def("__repr__",
             [](const core::PropertyGraph &g) {
                 return "<adaptive_graph.PropertyGraph with " + std::to_string(g.num_vertices) + " vertices>";
             });

    // Bind QueryResult
    py::class_<core::QueryResult>(m, "QueryResult")
        .def_readwrite("matched_vertices", &core::QueryResult::matched_vertices)
        .def_readwrite("execution_time_ms", &core::QueryResult::execution_time_ms);

    // Bind CypherParser
    m.def("execute_cypher", &core::CypherParser::execute_query, "Execute a basic Cypher query on a PropertyGraph",
          py::arg("graph"), py::arg("query"));

    // Bind IO functions
    m.def("load_graph", &core::load_graph, "Load a graph from an edge list file", py::arg("filepath"));
    m.def("create_tiny_test_graph", &core::create_tiny_test_graph, "Create a tiny test graph for testing");

    // Bind Algorithms
    m.def("pagerank", &algorithms::pagerank_openmp, "Run PageRank (OpenMP)", 
          py::arg("graph"), py::arg("iterations") = 20, py::arg("damping") = 0.85f);
          
    m.def("bfs", &algorithms::bfs_direction_optimizing, "Run Direction-Optimizing BFS (Graph500 Standard)",
          py::arg("graph"), py::arg("source_vertex"));
          
    m.def("connected_components", &algorithms::connected_components_openmp, "Find Connected Components (Label Propagation)",
          py::arg("graph"));
          
    m.def("sssp", &algorithms::sssp_dijkstra, "Run Single-Source Shortest Path (Dijkstra)",
          py::arg("graph"), py::arg("source_vertex"));
          
    m.def("gcn_forward_pass", &algorithms::GCNLayer::forward_pass, "Run a Graph Convolutional Network (GCN) Forward Pass",
          py::arg("graph"), py::arg("node_features"), py::arg("layer_weight"));
          
    m.def("quantum_grover_search", &algorithms::QuantumSimulator::grover_search, "Simulate Quantum Grover Search for a vertex",
          py::arg("graph"), py::arg("target_vertex"), py::arg("iterations") = 1);
}
