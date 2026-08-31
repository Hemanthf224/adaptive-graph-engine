#include "core/scheduler.hpp"
#include "algorithms/bfs.hpp"
#include "algorithms/bfs_cuda.cuh"
#include "algorithms/page_rank_cuda.cuh"
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

SchedulerDecision AdaptiveScheduler::determine_optimal_mode(const CSRGraph& graph) {
    SchedulerDecision decision;
    
    double V = static_cast<double>(graph.num_vertices);
    double E = static_cast<double>(graph.num_edges);
    
    // Graph Features
    double avg_degree = (V > 0) ? (E / V) : 0;
    double bytes_footprint = (V + 1) * 4.0 + (E * 4.0);
    double vram_capacity = 8.0 * 1024 * 1024 * 1024; // 8GB
    
    // 1. Base Score initialization
    decision.cpu_score = 0.1;
    decision.omp_score = 0.5;
    decision.cuda_score = 0.8;
    
    // 2. Small Graph Penalty (PCI-E transfer dominates)
    if (E < 1000000) {
        decision.cuda_score -= 0.6;
        decision.omp_score += 0.3;
        decision.cpu_score += 0.8; // Sequential is best for L3 cache hits
    }
    
    // 3. Power-Law Contention Penalty (High Average Degree)
    if (avg_degree > 10.0) {
        // High degree means massive atomic contention on the GPU
        decision.cuda_score -= 0.4;
        decision.omp_score += 0.4; // CPU cache handles contention better
        decision.reasoning = "High atomic contention risk detected in power-law graph; OpenMP selected over CUDA.";
    } else {
        decision.reasoning = "Low contention, high parallelism. CUDA selected for maximum memory bandwidth.";
    }
    
    // 4. Memory Bounds Check
    if (bytes_footprint > vram_capacity * 0.9) {
        decision.cuda_score -= 10.0; // Out of memory
        decision.reasoning = "Graph exceeds VRAM capacity. Falling back to OpenMP CPU.";
    }
    
    // Determine winner
    if (decision.cpu_score > decision.omp_score && decision.cpu_score > decision.cuda_score) {
        decision.selected_mode = ExecutionMode::SEQUENTIAL;
        if (E < 1000000) decision.reasoning = "Graph easily fits in CPU L3 Cache. Sequential execution prevents thread-spawning overhead.";
    } else if (decision.omp_score > decision.cuda_score) {
        decision.selected_mode = ExecutionMode::OPENMP;
    } else {
        decision.selected_mode = ExecutionMode::CUDA;
    }
    
    return decision;
}

std::vector<int32_t> AdaptiveScheduler::execute_bfs(const CSRGraph& graph, vertex_id_t source) {
    SchedulerDecision decision = determine_optimal_mode(graph);
    ExecutionMode mode = decision.selected_mode;
    
    std::cout << "[Adaptive Engine] Graph Size : " << graph.num_vertices << " V, " << graph.num_edges << " E\n";
    std::cout << "[Adaptive Engine] CPU Score  : " << decision.cpu_score << "\n";
    std::cout << "[Adaptive Engine] OMP Score  : " << decision.omp_score << "\n";
    std::cout << "[Adaptive Engine] CUDA Score : " << decision.cuda_score << "\n";
    std::cout << "[Adaptive Engine] Reasoning  : " << decision.reasoning << "\n";
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
            // WARM START: Prefetch UVM Memory to the GPU before execution
            algorithms::prefetch_graph_to_gpu(graph, 0);
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
