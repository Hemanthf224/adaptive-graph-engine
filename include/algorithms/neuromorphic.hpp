#pragma once
#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// Result of a Spiking Neural Network simulation
struct SNNResult {
    std::vector<int> total_spikes; // Number of times each neuron (vertex) fired
    int total_network_spikes;      // Total spikes across the entire network
};

class NeuromorphicEngine {
public:
    // Simulates a Spiking Graph Neural Network using Leaky Integrate-and-Fire (LIF) neurons.
    // 
    // time_steps: Number of discrete time steps to simulate.
    // input_current: Constant current injected into every neuron at each time step.
    // threshold: Voltage threshold required to fire a spike.
    // leak_rate: How fast the membrane potential decays towards resting (0.0).
    // synaptic_weight: How much voltage a spike adds to neighboring neurons.
    static SNNResult simulate_lif_network(const core::CSRGraph& graph, 
                                          int time_steps = 100, 
                                          float input_current = 0.5f,
                                          float threshold = 1.0f, 
                                          float leak_rate = 0.9f, 
                                          float synaptic_weight = 0.2f);
};

} // namespace algorithms
} // namespace graph_engine
