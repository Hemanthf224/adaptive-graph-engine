#include "algorithms/differential_privacy.hpp"
#include "algorithms/page_rank.hpp"
#include <cmath>
#include <stdexcept>

namespace graph_engine {
namespace algorithms {

DPResult DifferentialPrivacy::laplace_mechanism(const std::vector<float>& scores, const DPConfig& config) {
    if (config.epsilon <= 0) throw std::invalid_argument("Epsilon must be > 0");
    
    double noise_scale = config.sensitivity / config.epsilon;
    
    // Laplace distribution sampler via inverse CDF method
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> uniform(-0.5, 0.5);
    
    DPResult result;
    result.epsilon_used = config.epsilon;
    result.noise_scale = noise_scale;
    result.noisy_scores.resize(scores.size());
    
    for (size_t i = 0; i < scores.size(); ++i) {
        // Sample from Laplace(0, noise_scale) via inverse CDF: -b * sgn(u) * ln(1 - 2|u|)
        double u = uniform(rng);
        double sign = (u >= 0) ? 1.0 : -1.0;
        double laplace_noise = -noise_scale * sign * std::log(1.0 - 2.0 * std::abs(u));
        result.noisy_scores[i] = static_cast<float>(scores[i] + laplace_noise);
    }
    
    return result;
}

DPResult DifferentialPrivacy::gaussian_mechanism(const std::vector<float>& scores, const DPConfig& config) {
    if (config.epsilon <= 0 || config.delta <= 0)
        throw std::invalid_argument("Epsilon and delta must be > 0 for Gaussian mechanism");
    
    // Optimal Gaussian noise scale: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
    double sigma = config.sensitivity * std::sqrt(2.0 * std::log(1.25 / config.delta)) / config.epsilon;
    
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::normal_distribution<double> gaussian(0.0, sigma);
    
    DPResult result;
    result.epsilon_used = config.epsilon;
    result.noise_scale = sigma;
    result.noisy_scores.resize(scores.size());
    
    for (size_t i = 0; i < scores.size(); ++i) {
        result.noisy_scores[i] = static_cast<float>(scores[i] + gaussian(rng));
    }
    
    return result;
}

double DifferentialPrivacy::pagerank_sensitivity(const core::CSRGraph& graph, double damping) {
    // L1 global sensitivity of PageRank is bounded by 2*damping / (N * (1 - damping))
    // This is the Blocki-Blum-Datta-Sheffet (2013) sensitivity bound
    if (graph.num_vertices == 0) return 0.0;
    return (2.0 * damping) / (graph.num_vertices * (1.0 - damping));
}

DPResult DifferentialPrivacy::dp_pagerank(const core::CSRGraph& graph, double epsilon, int iterations) {
    // 1. Run true PageRank (on the server, in plaintext)
    auto true_scores = algorithms::pagerank_sequential(graph, iterations, 0.85f);
    
    // 2. Compute global sensitivity analytically
    double sensitivity = pagerank_sensitivity(graph, 0.85);
    
    // 3. Apply Laplace mechanism to achieve (epsilon, 0)-DP
    DPConfig config{epsilon, 0.0, sensitivity};
    DPResult result = laplace_mechanism(true_scores, config);
    
    // 4. Post-process: clip negative scores (post-processing preserves DP)
    for (auto& s : result.noisy_scores) {
        if (s < 0.0f) s = 0.0f;
    }
    
    return result;
}

} // namespace algorithms
} // namespace graph_engine
