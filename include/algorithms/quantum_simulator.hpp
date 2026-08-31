#pragma once
#include "core/graph.hpp"
#include <vector>
#include <complex>

namespace graph_engine {
namespace algorithms {

class QuantumSimulator {
public:
    // Simulates Grover's Algorithm to search for a specific vertex in the graph.
    // Returns the probability distribution of measuring each vertex after `iterations`.
    static std::vector<double> grover_search(const core::CSRGraph& graph, core::vertex_id_t target_vertex, int iterations = 1);
};

} // namespace algorithms
} // namespace graph_engine
