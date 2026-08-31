#pragma once

#include "core/graph.hpp"
#include <vector>
#include <limits>

namespace graph_engine {
namespace algorithms {

// Single-Source Shortest Path using Dijkstra (Sequential CPU)
std::vector<float> sssp_dijkstra(const core::CSRGraph& graph, core::vertex_id_t source);

} // namespace algorithms
} // namespace graph_engine
