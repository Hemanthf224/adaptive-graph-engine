#include "algorithms/federated_learning.hpp"
#include "algorithms/gnn_layer.hpp"
#include <omp.h>
#include <cmath>
#include <numeric>
#include <iostream>
#include <random>

namespace graph_engine {
namespace algorithms {

std::vector<FederatedNode> FederatedLearning::partition_graph(const core::CSRGraph& graph, int num_nodes) {
    std::vector<FederatedNode> nodes(num_nodes);
    size_t V = graph.num_vertices;
    size_t shard_size = std::max<size_t>(1, V / num_nodes);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int n = 0; n < num_nodes; ++n) {
        nodes[n].node_id = n;
        
        // Determine vertex range for this shard
        size_t v_start = n * shard_size;
        size_t v_end = (n == num_nodes - 1) ? V : (n + 1) * shard_size;
        size_t local_V = v_end - v_start;
        
        // Build a local subgraph CSR for this shard's induced subgraph
        nodes[n].local_subgraph.num_vertices = local_V;
        nodes[n].local_subgraph.row_offsets.resize(local_V + 1, 0);
        
        core::edge_id_t edge_count = 0;
        for (size_t u = v_start; u < v_end; ++u) {
            nodes[n].local_subgraph.row_offsets[u - v_start] = edge_count;
            core::edge_id_t start = graph.row_offsets[u];
            core::edge_id_t end_e = graph.row_offsets[u + 1];
            for (core::edge_id_t e = start; e < end_e; ++e) {
                core::vertex_id_t v = graph.column_indices[e];
                // Only include edges within the shard (induced subgraph)
                if (v >= v_start && v < v_end) {
                    nodes[n].local_subgraph.column_indices.push_back(v - v_start);
                    edge_count++;
                }
            }
        }
        nodes[n].local_subgraph.row_offsets[local_V] = edge_count;
        nodes[n].local_subgraph.num_edges = edge_count;

        // Initialize random local features and model weights
        nodes[n].local_features.resize(local_V);
        for (auto& f : nodes[n].local_features) f = dist(rng);
        
        // Initialize GCN weight randomly between 0 and 1
        nodes[n].local_model_weights = {dist(rng)};
        nodes[n].local_loss = 1.0f;
    }

    return nodes;
}

FederatedModel FederatedLearning::fedavg_round(std::vector<FederatedNode>& nodes, float learning_rate, int local_epochs) {
    // Step 1: Local Training (parallel across nodes using OpenMP)
    #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < static_cast<int>(nodes.size()); ++n) {
        auto& node = nodes[n];
        
        for (int epoch = 0; epoch < local_epochs; ++epoch) {
            // Forward pass: run one GCN layer with current local weight
            float weight = node.local_model_weights[0];
            auto output = GCNLayer::forward_pass(node.local_subgraph, node.local_features, weight);
            
            // Compute local MSE loss (self-supervised: predict own features)
            float loss = 0.0f;
            for (size_t i = 0; i < output.size(); ++i) {
                float diff = output[i] - node.local_features[i];
                loss += diff * diff;
            }
            if (!output.empty()) loss /= output.size();
            node.local_loss = loss;
            
            // Gradient descent: dL/dW ≈ 2 * mean(output - target) * mean(features)
            float mean_output = std::accumulate(output.begin(), output.end(), 0.0f) / (output.empty() ? 1 : output.size());
            float mean_feat = std::accumulate(node.local_features.begin(), node.local_features.end(), 0.0f) / (node.local_features.empty() ? 1 : node.local_features.size());
            float gradient = 2.0f * (mean_output - mean_feat) * mean_feat;
            node.local_model_weights[0] -= learning_rate * gradient;
        }
    }

    // Step 2: FedAvg Aggregation on Server
    // Global weight = weighted average of local weights (weighted by local dataset size)
    FederatedModel global_model;
    float total_samples = 0.0f;
    float weighted_weight_sum = 0.0f;
    float weighted_loss_sum = 0.0f;
    
    for (const auto& node : nodes) {
        float n_samples = static_cast<float>(node.local_subgraph.num_vertices);
        total_samples += n_samples;
        weighted_weight_sum += n_samples * node.local_model_weights[0];
        weighted_loss_sum += n_samples * node.local_loss;
    }
    
    global_model.global_weights = {total_samples > 0 ? weighted_weight_sum / total_samples : 0.0f};
    global_model.global_loss = total_samples > 0 ? weighted_loss_sum / total_samples : 0.0f;

    // Step 3: Broadcast global model back to all nodes
    for (auto& node : nodes) {
        node.local_model_weights = global_model.global_weights;
    }

    return global_model;
}

FederatedModel FederatedLearning::train(const core::CSRGraph& graph, int num_nodes, int num_rounds) {
    auto nodes = partition_graph(graph, num_nodes);
    FederatedModel model;
    model.num_rounds = 0;
    
    for (int r = 0; r < num_rounds; ++r) {
        model = fedavg_round(nodes, 0.01f, 5);
        model.num_rounds = r + 1;
    }
    
    return model;
}

} // namespace algorithms
} // namespace graph_engine
