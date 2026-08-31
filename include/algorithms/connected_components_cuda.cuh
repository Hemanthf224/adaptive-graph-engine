#pragma once

#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// GPU CUDA Connected Components (Label Propagation on UVM)
std::vector<int> connected_components_cuda(const core::CSRGraph& graph);

} // namespace algorithms
} // namespace graph_engine
