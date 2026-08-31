#pragma once
#include "core/graph.hpp"
#include <vector>
#include <random>

namespace graph_engine {
namespace algorithms {

// Differential Privacy parameters
struct DPConfig {
    double epsilon;   // Privacy budget (smaller = more private, typically 0.1 - 10.0)
    double delta;     // Failure probability (typically 1e-5 to 1e-8, 0 = pure DP)
    double sensitivity; // L1-sensitivity of the query function
};

// Result from a differentially private query
struct DPResult {
    std::vector<float> noisy_scores;   // Scores with calibrated noise injected
    double epsilon_used;               // Actual epsilon budget consumed
    double noise_scale;                // Scale of Laplace/Gaussian noise added
};

class DifferentialPrivacy {
public:
    // Laplace Mechanism: Achieves (epsilon, 0)-DP
    // Used by Google RAPPOR, US Census Bureau
    // Adds Laplace(sensitivity/epsilon) noise to each score
    static DPResult laplace_mechanism(const std::vector<float>& scores, const DPConfig& config);
    
    // Gaussian Mechanism: Achieves (epsilon, delta)-DP
    // Used by Apple iOS telemetry, Microsoft SEAL
    // Adds Gaussian(sensitivity * sqrt(2*ln(1.25/delta)) / epsilon) noise
    static DPResult gaussian_mechanism(const std::vector<float>& scores, const DPConfig& config);
    
    // Full DP PageRank: Runs PageRank then applies Laplace mechanism
    static DPResult dp_pagerank(const core::CSRGraph& graph, double epsilon, int iterations = 20);
    
    // Compute L1 global sensitivity of PageRank analytically
    static double pagerank_sensitivity(const core::CSRGraph& graph, double damping = 0.85);
};

} // namespace algorithms
} // namespace graph_engine
