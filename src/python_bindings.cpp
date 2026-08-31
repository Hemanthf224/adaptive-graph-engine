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
#include "algorithms/fhe_simulator.hpp"
#include "algorithms/zkp_verifier.hpp"
#include "algorithms/differential_privacy.hpp"
#include "algorithms/federated_learning.hpp"

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

    m.def("encrypted_pagerank", &algorithms::FHESimulator::encrypted_pagerank,
          "Run PageRank on fully homomorphically encrypted graph data (Privacy-Preserving Analytics)",
          py::arg("graph"), py::arg("iterations") = 10, py::arg("damping") = 0.85f);

    // ZKP Verifier Bindings
    py::class_<algorithms::ZKPCommitment>(m, "ZKPCommitment")
        .def_readwrite("commitment_hash", &algorithms::ZKPCommitment::commitment_hash)
        .def_readwrite("randomness", &algorithms::ZKPCommitment::randomness)
        .def_readwrite("num_vertices", &algorithms::ZKPCommitment::num_vertices);

    py::class_<algorithms::ZKPProof>(m, "ZKPProof")
        .def_readwrite("proof_hash", &algorithms::ZKPProof::proof_hash)
        .def_readwrite("is_valid", &algorithms::ZKPProof::is_valid);

    m.def("zkp_commit", &algorithms::ZKPVerifier::commit,
          "Commit to PageRank scores without revealing them (Pedersen commitment)",
          py::arg("scores"), py::arg("randomness") = 0xDEADBEEFCAFEBABEULL);

    m.def("zkp_prove", &algorithms::ZKPVerifier::prove,
          "Generate a Zero-Knowledge Proof of correct PageRank execution",
          py::arg("graph"), py::arg("commitment"), py::arg("scores"));

    m.def("zkp_verify", &algorithms::ZKPVerifier::verify,
          "Verify a ZKP without seeing the underlying scores",
          py::arg("graph"), py::arg("commitment"), py::arg("proof"));

    // Differential Privacy Bindings
    py::class_<algorithms::DPConfig>(m, "DPConfig")
        .def(py::init<double, double, double>())
        .def_readwrite("epsilon", &algorithms::DPConfig::epsilon)
        .def_readwrite("delta", &algorithms::DPConfig::delta)
        .def_readwrite("sensitivity", &algorithms::DPConfig::sensitivity);

    py::class_<algorithms::DPResult>(m, "DPResult")
        .def_readwrite("noisy_scores", &algorithms::DPResult::noisy_scores)
        .def_readwrite("epsilon_used", &algorithms::DPResult::epsilon_used)
        .def_readwrite("noise_scale", &algorithms::DPResult::noise_scale);

    m.def("dp_pagerank", &algorithms::DifferentialPrivacy::dp_pagerank,
          "Run (epsilon, 0)-Differentially Private PageRank via Laplace Mechanism",
          py::arg("graph"), py::arg("epsilon"), py::arg("iterations") = 20);

    m.def("pagerank_sensitivity", &algorithms::DifferentialPrivacy::pagerank_sensitivity,
          "Compute the L1 global sensitivity of PageRank (Blocki-Blum-Datta-Sheffet bound)",
          py::arg("graph"), py::arg("damping") = 0.85);

    // Federated Learning Bindings
    py::class_<algorithms::FederatedModel>(m, "FederatedModel")
        .def_readwrite("global_weights", &algorithms::FederatedModel::global_weights)
        .def_readwrite("num_rounds", &algorithms::FederatedModel::num_rounds)
        .def_readwrite("global_loss", &algorithms::FederatedModel::global_loss);

    m.def("federated_train", &algorithms::FederatedLearning::train,
          "Train a GCN using Federated Learning (FedAvg) across distributed graph shards",
          py::arg("graph"), py::arg("num_nodes") = 4, py::arg("num_rounds") = 10);
}
