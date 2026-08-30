#pragma once

#include "core/graph.hpp"
#include <vector>
#include <cstdint>
#include <string>

namespace graph_engine {
namespace core {

// Enum to define execution modes
enum class ExecutionMode {
    SEQUENTIAL,
    OPENMP,
    CUDA
};

// The Adaptive Scheduler analyzes the graph properties at runtime
// and dispatches the execution to the most optimal hardware layer.
class AdaptiveScheduler {
public:
    // Analyzes graph and returns the determined execution mode
    static ExecutionMode determine_optimal_mode(const CSRGraph& graph);

    // Executes Breadth-First Search using the dynamically determined mode
    static std::vector<int32_t> execute_bfs(const CSRGraph& graph, vertex_id_t source);
    
    // Convert enum to string for logging
    static std::string mode_to_string(ExecutionMode mode);
};

} // namespace core
} // namespace graph_engine
