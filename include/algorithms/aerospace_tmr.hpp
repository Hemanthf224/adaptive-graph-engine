#pragma once
#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

// Triple Modular Redundancy (TMR) Result
struct TMRResult {
    std::vector<float> robust_scores; // The final majority-voted result
    int faults_detected;              // Number of cosmic ray SEUs (Single Event Upsets) detected and corrected
};

class AerospaceTMR {
public:
    // Runs PageRank using Triple Modular Redundancy (TMR) for radiation-hardened space environments.
    // The algorithm is executed three independent times.
    // A bitwise majority vote is performed to eliminate any data corruption caused by cosmic rays.
    // 
    // `simulate_cosmic_ray` allows artificially injecting a bit-flip into one of the runs for testing.
    static TMRResult tmr_pagerank(const core::CSRGraph& graph, int iterations = 20, float damping = 0.85f, bool simulate_cosmic_ray = false);
};

} // namespace algorithms
} // namespace graph_engine
