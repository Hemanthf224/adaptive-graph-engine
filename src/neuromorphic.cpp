#include "algorithms/neuromorphic.hpp"
#include <omp.h>
#include <iostream>

namespace graph_engine {
namespace algorithms {

SNNResult NeuromorphicEngine::simulate_lif_network(const core::CSRGraph& graph, int time_steps, float input_current, float threshold, float leak_rate, float synaptic_weight) {
    size_t V = graph.num_vertices;
    
    // Membrane potential for each biological neuron
    std::vector<float> membrane_potential(V, 0.0f);
    
    // Spike accumulators
    std::vector<int> spike_counts(V, 0);
    int total_network_spikes = 0;

    // Buffer to hold incoming spikes from the previous time step
    std::vector<float> incoming_spikes(V, 0.0f);

    for (int t = 0; t < time_steps; ++t) {
        std::vector<float> next_incoming_spikes(V, 0.0f);

        #pragma omp parallel for reduction(+:total_network_spikes)
        for (core::vertex_id_t u = 0; u < V; ++u) {
            // 1. Leaky Integrate
            // Decay previous potential, add constant input current, add incoming synaptic spikes
            membrane_potential[u] = (membrane_potential[u] * leak_rate) + input_current + incoming_spikes[u];

            // 2. Fire
            if (membrane_potential[u] >= threshold) {
                // Neuron fires a spike!
                spike_counts[u]++;
                total_network_spikes++;
                
                // Reset membrane potential (refractory)
                membrane_potential[u] = 0.0f;

                // Propagate spike to all outgoing neighbors
                core::edge_id_t start = graph.row_offsets[u];
                core::edge_id_t end = graph.row_offsets[u + 1];
                for (core::edge_id_t e = start; e < end; ++e) {
                    core::vertex_id_t v = graph.column_indices[e];
                    
                    // Note: atomic add is needed here due to parallel execution
                    #pragma omp atomic
                    next_incoming_spikes[v] += synaptic_weight;
                }
            }
        }
        
        // Update incoming spikes for the next time step
        incoming_spikes = std::move(next_incoming_spikes);
    }

    SNNResult result;
    result.total_spikes = spike_counts;
    result.total_network_spikes = total_network_spikes;
    return result;
}

} // namespace algorithms
} // namespace graph_engine
