#pragma once
#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// A basic Graph Convolutional Network (GCN) layer implementation.
// Performs message passing: H^{(l+1)} = ReLU( A * H^{(l)} * W )
// where A is the adjacency matrix, H is the node feature matrix, and W is the weight matrix.

class GCNLayer {
public:
    // Performs a single forward pass of a GCN layer.
    // In this prototype, we assume node features are 1-dimensional for simplicity,
    // which effectively makes H a vector and W a scalar.
    // We compute: next_features[u] = ReLU( sum(features[v]) * weight ) for all v in neighbors(u).
    static std::vector<float> forward_pass(
        const core::CSRGraph& graph, 
        const std::vector<float>& node_features, 
        float layer_weight
    );
};

} // namespace algorithms
} // namespace graph_engine
