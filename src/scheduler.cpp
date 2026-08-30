#include "core/scheduler.hpp"
#include "algorithms/bfs.hpp"
#include "algorithms/bfs_cuda.cuh"
#include <iostream>
#include <chrono>

namespace graph_engine {
namespace core {

std::string AdaptiveScheduler::mode_to_string(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::SEQUENTIAL: return "CPU (Sequential)";
        case ExecutionMode::OPENMP:     return "CPU (OpenMP)";
        case ExecutionMode::CUDA:       return "GPU (CUDA)";
        default:                        return "UNKNOWN";
    }
}

ExecutionMode AdaptiveScheduler::determine_optimal_mode(const CSRGraph& graph) {
    // Basic Heuristics
    // 1. If the graph is microscopically small, PCI-E transfer and thread startup
    //    completely dominate runtime. Run sequentially.
    if (graph.num_vertices < 1000 && graph.num_edges < 10000) {
        return ExecutionMode::SEQUENTIAL;
    }
    
    // 2. If the graph is moderately sized, OpenMP is fastest because it avoids
    //    the PCI-E bottleneck but leverages multi-core CPU.
    if (graph.num_edges < 1000000) {
        return ExecutionMode::OPENMP;
    }
    
    // 3. If the graph is massive, the massive parallelism of the GPU easily
    //    hides the PCI-E transfer latency.
    return ExecutionMode::CUDA;
}

std::vector<int32_t> AdaptiveScheduler::execute_bfs(const CSRGraph& graph, vertex_id_t source) {
    ExecutionMode mode = determine_optimal_mode(graph);
    
    std::cout << "[Adaptive Engine] Graph Size : " << graph.num_vertices << " V, " << graph.num_edges << " E\n";
    std::cout << "[Adaptive Engine] Selected   : " << mode_to_string(mode) << "\n";
    
    std::vector<int32_t> result;
    auto start = std::chrono::high_resolution_clock::now();
    
    switch (mode) {
        case ExecutionMode::SEQUENTIAL:
            result = algorithms::bfs_sequential(graph, source);
            break;
        case ExecutionMode::OPENMP:
            result = algorithms::bfs_openmp(graph, source);
            break;
        case ExecutionMode::CUDA:
            result = algorithms::bfs_cuda(graph, source);
            break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "[Adaptive Engine] BFS finished in " << duration.count() << " ms.\n";
    return result;
}

} // namespace core
} // namespace graph_engine
