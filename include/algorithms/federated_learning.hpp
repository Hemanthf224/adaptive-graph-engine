#pragma once
#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// Represents a single participating node in a federated learning round.
// Each node owns a shard of the full graph and trains locally.
struct FederatedNode {
    int node_id;
    core::CSRGraph local_subgraph;           // This node's private data shard
    std::vector<float> local_features;       // Local node features
    std::vector<float> local_model_weights;  // Locally learned GCN weights
    float local_loss;                        // Loss on local data
};

// Global model aggregated via FedAvg
struct FederatedModel {
    std::vector<float> global_weights;  // Average of all local weights
    int num_rounds;                     // How many rounds of FL have been done
    float global_loss;                  // Weighted average of local losses
};

class FederatedLearning {
public:
    // Simulate partitioning a graph into N shards for N federated nodes
    static std::vector<FederatedNode> partition_graph(const core::CSRGraph& graph, int num_nodes);

    // Simulate one round of FedAvg:
    // 1. Each node trains its local GCN for E local epochs
    // 2. The server aggregates weights via weighted averaging
    static FederatedModel fedavg_round(std::vector<FederatedNode>& nodes, float learning_rate = 0.01f, int local_epochs = 5);

    // Run multiple rounds of FL until convergence
    static FederatedModel train(const core::CSRGraph& graph, int num_nodes = 4, int num_rounds = 10);
};

} // namespace algorithms
} // namespace graph_engine
