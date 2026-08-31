#include "algorithms/quantum_simulator.hpp"
#include <cmath>
#include <iostream>

namespace graph_engine {
namespace algorithms {

std::vector<double> QuantumSimulator::grover_search(const core::CSRGraph& graph, core::vertex_id_t target_vertex, int iterations) {
    size_t N = graph.num_vertices;
    if (N == 0) return {};

    // 1. Initialize State Vector |s> (Equal Superposition of all vertices)
    // The amplitude for each state is 1 / sqrt(N)
    std::vector<std::complex<double>> state_vector(N);
    double initial_amplitude = 1.0 / std::sqrt(N);
    
    for (size_t i = 0; i < N; ++i) {
        state_vector[i] = std::complex<double>(initial_amplitude, 0.0);
    }

    // Optimal number of Grover iterations is roughly (pi/4) * sqrt(N)
    int optimal_iterations = std::max(1, static_cast<int>((M_PI / 4.0) * std::sqrt(N)));
    if (iterations <= 0) {
        iterations = optimal_iterations;
    }

    // 2. Apply Grover Iterations
    for (int iter = 0; iter < iterations; ++iter) {
        // Step A: The Oracle (Phase Inversion)
        // Flips the sign of the amplitude for the target vertex.
        if (target_vertex < N) {
            state_vector[target_vertex] *= -1.0;
        }

        // Step B: The Grover Diffusion Operator (Inversion about the Mean)
        // Calculate the mean amplitude
        std::complex<double> sum(0.0, 0.0);
        for (size_t i = 0; i < N; ++i) {
            sum += state_vector[i];
        }
        std::complex<double> mean = sum / static_cast<double>(N);

        // Reflect amplitudes about the mean: new_amp = 2 * mean - old_amp
        for (size_t i = 0; i < N; ++i) {
            state_vector[i] = 2.0 * mean - state_vector[i];
        }
    }

    // 3. Calculate Measurement Probabilities
    // Probability is the magnitude squared of the complex amplitude (|alpha|^2)
    std::vector<double> probabilities(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        probabilities[i] = std::norm(state_vector[i]);
    }

    return probabilities;
}

} // namespace algorithms
} // namespace graph_engine
