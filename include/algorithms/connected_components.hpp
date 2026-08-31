#pragma once

#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// CPU Sequential Connected Components (Label Propagation)
std::vector<int> connected_components_sequential(const core::CSRGraph& graph);

// Multi-Core OpenMP Connected Components (Label Propagation)
std::vector<int> connected_components_openmp(const core::CSRGraph& graph);

} // namespace algorithms
} // namespace graph_engine
