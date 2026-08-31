#include "algorithms/fhe_simulator.hpp"
#include <iostream>

namespace graph_engine {
namespace algorithms {

std::vector<float> FHESimulator::encrypted_pagerank(const core::CSRGraph& graph, int iterations, float damping) {
    size_t V = graph.num_vertices;
    if (V == 0) return {};

    // 1. Client Encrypts Initial State
    // The server only sees `EncryptedFloat` objects (simulated ciphertexts)
    std::vector<EncryptedFloat> encrypted_scores(V, EncryptedFloat(1.0f / V));
    std::vector<EncryptedFloat> next_encrypted_scores(V, EncryptedFloat(0.0f));
    
    // Server encrypts the damping constants (Public Keys)
    EncryptedFloat enc_damping(damping);
    EncryptedFloat enc_base((1.0f - damping) / V);

    // 2. Server Computes PageRank blindly on Ciphertext
    for (int iter = 0; iter < iterations; ++iter) {
        // Reset next scores to base probability
        for (size_t i = 0; i < V; ++i) {
            next_encrypted_scores[i] = enc_base;
        }

        // Homomorphic Message Passing
        for (core::vertex_id_t u = 0; u < V; ++u) {
            core::edge_id_t start_edge = graph.row_offsets[u];
            core::edge_id_t end_edge = graph.row_offsets[u + 1];
            size_t out_degree = end_edge - start_edge;
            
            if (out_degree > 0) {
                // Server computes: Enc(score / degree)
                EncryptedFloat enc_contribution = encrypted_scores[u] * EncryptedFloat(1.0f / out_degree);
                
                for (core::edge_id_t e = start_edge; e < end_edge; ++e) {
                    core::vertex_id_t v = graph.column_indices[e];
                    // Homomorphic Addition: C_next = C_next + C_contribution
                    next_encrypted_scores[v] = next_encrypted_scores[v] + (enc_contribution * enc_damping);
                }
            }
        }
        
        // Swap ciphertext pointers
        encrypted_scores = next_encrypted_scores;
    }

    // 3. Client Decrypts Results
    // The server sends `encrypted_scores` back to the client.
    // The client uses their Private Key to decrypt it locally.
    std::vector<float> final_decrypted_scores(V);
    for (size_t i = 0; i < V; ++i) {
        final_decrypted_scores[i] = encrypted_scores[i].decrypt();
    }

    return final_decrypted_scores;
}

} // namespace algorithms
} // namespace graph_engine
