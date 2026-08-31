#include "algorithms/aerospace_tmr.hpp"
#include "algorithms/page_rank.hpp"
#include <omp.h>
#include <cstring>
#include <iostream>

namespace graph_engine {
namespace algorithms {

TMRResult AerospaceTMR::tmr_pagerank(const core::CSRGraph& graph, int iterations, float damping, bool simulate_cosmic_ray) {
    size_t V = graph.num_vertices;
    std::vector<std::vector<float>> runs(3);

    // 1. Triple Execution (Hardware redundancy simulated via software parallelization)
    #pragma omp parallel for num_threads(3)
    for (int i = 0; i < 3; ++i) {
        runs[i] = algorithms::pagerank_sequential(graph, iterations, damping);
    }

    // 2. Cosmic Ray Injection (Simulate a Single Event Upset in Run 0)
    if (simulate_cosmic_ray && V > 0) {
        uint32_t bits;
        std::memcpy(&bits, &runs[0][0], sizeof(bits));
        // Flip the 16th bit
        bits ^= (1 << 16);
        std::memcpy(&runs[0][0], &bits, sizeof(bits));
    }

    // 3. Majority Voting (TMR)
    TMRResult result;
    result.robust_scores.resize(V);
    result.faults_detected = 0;

    for (size_t i = 0; i < V; ++i) {
        uint32_t v0, v1, v2;
        std::memcpy(&v0, &runs[0][i], sizeof(v0));
        std::memcpy(&v1, &runs[1][i], sizeof(v1));
        std::memcpy(&v2, &runs[2][i], sizeof(v2));

        // Bitwise majority voting (A&B | B&C | C&A)
        uint32_t majority = (v0 & v1) | (v1 & v2) | (v2 & v0);

        std::memcpy(&result.robust_scores[i], &majority, sizeof(majority));

        // Fault detection telemetry
        if (v0 != majority || v1 != majority || v2 != majority) {
            result.faults_detected++;
        }
    }

    return result;
}

} // namespace algorithms
} // namespace graph_engine
