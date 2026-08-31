#include "algorithms/bfs.hpp"
#include "core/profiler.hpp"
#include <queue>
#include <omp.h>

namespace graph_engine {
namespace algorithms {

std::vector<int32_t> bfs_sequential(const core::CSRGraph& graph, core::vertex_id_t source) {
    PROFILE_FUNCTION();
    std::vector<int32_t> distances(graph.num_vertices, -1);
    
    if (source >= graph.num_vertices) return distances;
    
    std::queue<core::vertex_id_t> frontier;
    frontier.push(source);
    distances[source] = 0;

    while (!frontier.empty()) {
        core::vertex_id_t u = frontier.front();
        frontier.pop();

        core::edge_id_t start = graph.row_offsets[u];
        core::edge_id_t end = graph.row_offsets[u + 1];

        for (core::edge_id_t e = start; e < end; ++e) {
            core::vertex_id_t v = graph.column_indices[e];
            if (distances[v] == -1) {
                distances[v] = distances[u] + 1;
                frontier.push(v);
            }
        }
    }

    return distances;
}

std::vector<int32_t> bfs_openmp(const core::CSRGraph& graph, core::vertex_id_t source) {
    PROFILE_FUNCTION();
    std::vector<int32_t> distances(graph.num_vertices, -1);
    
    if (source >= graph.num_vertices) return distances;
    
    // For parallel BFS, a standard queue causes massive thread contention.
    // We use two vectors to represent the current and next frontier.
    std::vector<core::vertex_id_t> current_frontier;
    std::vector<core::vertex_id_t> next_frontier;
    
    current_frontier.push_back(source);
    distances[source] = 0;
    
    int32_t current_level = 0;

    while (!current_frontier.empty()) {
        next_frontier.clear();

        // To avoid thread contention on push_back, we can allocate thread-local vectors
        // and combine them at the end of the level, or just use a critical section for now.
        // For optimal performance, thread-local vectors are better.
        
        #pragma omp parallel
        {
            std::vector<core::vertex_id_t> local_next_frontier;
            
            #pragma omp for schedule(dynamic, 64)
            for (size_t i = 0; i < current_frontier.size(); ++i) {
                core::vertex_id_t u = current_frontier[i];
                core::edge_id_t start = graph.row_offsets[u];
                core::edge_id_t end = graph.row_offsets[u + 1];

                for (core::edge_id_t e = start; e < end; ++e) {
                    core::vertex_id_t v = graph.column_indices[e];
                    
                    // We use __atomic_compare_exchange to ensure only one thread claims the unvisited neighbor
                    int32_t expected = -1;
                    int32_t next_level = current_level + 1;
                    if (__atomic_compare_exchange_n(&distances[v], &expected, next_level, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
                        local_next_frontier.push_back(v);
                    }
                }
            }

            // Combine thread-local frontiers into the global next_frontier
            #pragma omp critical
            {
                next_frontier.insert(next_frontier.end(), local_next_frontier.begin(), local_next_frontier.end());
            }
        }

        std::swap(current_frontier, next_frontier);
        current_level++;
    }

    return distances;
}

std::vector<int32_t> bfs_direction_optimizing(const core::CSRGraph& graph, core::vertex_id_t source) {
    PROFILE_FUNCTION();
    size_t V = graph.num_vertices;
    std::vector<int32_t> distances(V, -1);
    
    if (source >= V) return distances;
    
    distances[source] = 0;
    
    std::vector<core::vertex_id_t> frontier;
    frontier.push_back(source);
    
    std::vector<core::vertex_id_t> next_frontier;
    int32_t current_distance = 0;
    
    // Alpha and Beta thresholds for Direction-Optimizing BFS (Graph500 standard)
    const size_t alpha = 14; 
    const size_t beta = 24;
    
    size_t mf = 0; // Edges in frontier
    size_t mu = graph.num_edges; // Edges unvisited
    
    while (!frontier.empty()) {
        mf = 0;
        #pragma omp parallel for reduction(+:mf)
        for (size_t i = 0; i < frontier.size(); ++i) {
            core::vertex_id_t u = frontier[i];
            mf += (graph.row_offsets[u + 1] - graph.row_offsets[u]);
        }
        
        bool use_bottom_up = (mf > mu / alpha);
        
        current_distance++;
        next_frontier.clear();
        
        if (!use_bottom_up) {
            // TOP-DOWN STEP (Standard BFS)
            #pragma omp parallel
            {
                std::vector<core::vertex_id_t> local_next;
                #pragma omp for schedule(dynamic, 64)
                for (size_t i = 0; i < frontier.size(); ++i) {
                    core::vertex_id_t u = frontier[i];
                    core::edge_id_t start = graph.row_offsets[u];
                    core::edge_id_t end = graph.row_offsets[u + 1];
                    
                    for (core::edge_id_t e = start; e < end; ++e) {
                        core::vertex_id_t v = graph.column_indices[e];
                        int32_t expected = -1;
                        if (__atomic_compare_exchange_n(&distances[v], &expected, current_distance, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            local_next.push_back(v);
                        }
                    }
                }
                
                #pragma omp critical
                {
                    next_frontier.insert(next_frontier.end(), local_next.begin(), local_next.end());
                }
            }
        } else {
            // BOTTOM-UP STEP (Look backwards)
            #pragma omp parallel
            {
                std::vector<core::vertex_id_t> local_next;
                #pragma omp for schedule(dynamic, 1024)
                for (core::vertex_id_t u = 0; u < V; ++u) {
                    if (distances[u] == -1) { // Unvisited
                        core::edge_id_t start = graph.row_offsets[u];
                        core::edge_id_t end = graph.row_offsets[u + 1];
                        
                        for (core::edge_id_t e = start; e < end; ++e) {
                            core::vertex_id_t v = graph.column_indices[e];
                            if (distances[v] == current_distance - 1) { // Parent is in frontier
                                distances[u] = current_distance;
                                local_next.push_back(u);
                                break; // Stop checking edges for this vertex!
                            }
                        }
                    }
                }
                
                #pragma omp critical
                {
                    next_frontier.insert(next_frontier.end(), local_next.begin(), local_next.end());
                }
            }
        }
        
        mu -= mf; // Approximate remaining unvisited edges
        frontier = std::move(next_frontier);
    }
    
    return distances;
}

} // namespace algorithms
} // namespace graph_engine
