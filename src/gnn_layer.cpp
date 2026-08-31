#include "algorithms/gnn_layer.hpp"
#include <omp.h>
#include <immintrin.h> // AVX2
#include <algorithm>

namespace graph_engine {
namespace algorithms {

std::vector<float> GCNLayer::forward_pass(
    const core::CSRGraph& graph, 
    const std::vector<float>& node_features, 
    float layer_weight
) {
    size_t V = graph.num_vertices;
    std::vector<float> next_features(V, 0.0f);

    #pragma omp parallel for schedule(dynamic, 1024)
    for (core::vertex_id_t u = 0; u < V; ++u) {
        core::edge_id_t start_edge = graph.row_offsets[u];
        core::edge_id_t end_edge = graph.row_offsets[u + 1];
        
        float sum = 0.0f;
        core::edge_id_t e = start_edge;
        
        // AVX2 Vectorized Feature Aggregation (Message Passing)
#ifdef __AVX2__
        __m256 v_sum = _mm256_setzero_ps();
        for (; e + 8 <= end_edge; e += 8) {
            // Gather indices
            __m256i v_indices = _mm256_loadu_si256((__m256i*)&graph.column_indices[e]);
            // Gather features using indices (i32gather)
            __m256 v_feats = _mm256_i32gather_ps(&node_features[0], v_indices, 4);
            // Accumulate
            v_sum = _mm256_add_ps(v_sum, v_feats);
        }
        // Horizontal add
        float temp[8];
        _mm256_storeu_ps(temp, v_sum);
        for (int i = 0; i < 8; ++i) {
            sum += temp[i];
        }
#endif
        // Scalar remainder loop
        for (; e < end_edge; ++e) {
            core::vertex_id_t neighbor = graph.column_indices[e];
            sum += node_features[neighbor];
        }
        
        // Add self-loop feature (typical in GCNs)
        sum += node_features[u];
        
        // Multiply by weight
        float output = sum * layer_weight;
        
        // ReLU Activation
        next_features[u] = std::max(0.0f, output);
    }
    
    return next_features;
}

} // namespace algorithms
} // namespace graph_engine
