#include "algorithms/bfs_cuda.cuh"
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

namespace graph_engine {
namespace algorithms {

// CUDA Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while (0)

// Top-Down BFS Kernel
// One thread per vertex. If vertex is in the current frontier (distance == current_level),
// it pushes the distance to its unvisited neighbors.
__global__ void bfs_kernel(
    core::vertex_id_t num_vertices,
    const core::edge_id_t* row_offsets,
    const core::vertex_id_t* column_indices,
    int32_t* distances,
    int32_t current_level,
    bool* active_flag) 
{
    core::vertex_id_t u = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (u >= num_vertices) return;

    // Is this vertex in the active frontier?
    if (distances[u] == current_level) {
        core::edge_id_t start = row_offsets[u];
        core::edge_id_t end = row_offsets[u + 1];

        // Traverse neighbors
        for (core::edge_id_t e = start; e < end; ++e) {
            core::vertex_id_t v = column_indices[e];
            
            // Atomic check and set. If distance is -1, set it to current_level + 1.
            // atomicCAS returns the old value.
            if (distances[v] == -1) {
                int32_t expected = -1;
                int32_t new_dist = current_level + 1;
                int32_t old_val = atomicCAS(&distances[v], expected, new_dist);
                
                // If we successfully claimed this neighbor, we mark the flag so the CPU knows 
                // there is work to do in the next level.
                if (old_val == expected) {
                    *active_flag = true;
                }
            }
        }
    }
}

std::vector<int32_t> bfs_cuda(const core::CSRGraph& host_graph, core::vertex_id_t source) {
    std::vector<int32_t> host_distances(host_graph.num_vertices, -1);
    
    if (source >= host_graph.num_vertices) return host_distances;
    
    // Setup initial state
    host_distances[source] = 0;

    // Device Pointers
    int32_t* d_distances = nullptr;
    bool* d_active_flag = nullptr;

    // 1. Allocate Device Memory for output distances and flags
    size_t distances_size = host_graph.num_vertices * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_distances, distances_size));
    CUDA_CHECK(cudaMalloc(&d_active_flag, sizeof(bool)));

    // 2. Copy Host -> Device (Only for distances, graph is in UVM!)
    CUDA_CHECK(cudaMemcpy(d_distances, host_distances.data(), distances_size, cudaMemcpyHostToDevice));

    // Zero-Copy pointers for the kernel
    const core::edge_id_t* d_row_offsets = host_graph.row_offsets.data();
    const core::vertex_id_t* d_column_indices = host_graph.column_indices.data();

    // 3. Launch configuration
    int threads_per_block = 256;
    int blocks = (host_graph.num_vertices + threads_per_block - 1) / threads_per_block;
    
    int32_t current_level = 0;
    bool host_active_flag = true;

    // 4. Kernel Execution Loop
    while (host_active_flag) {
        // Reset flag for the next level
        host_active_flag = false;
        CUDA_CHECK(cudaMemcpy(d_active_flag, &host_active_flag, sizeof(bool), cudaMemcpyHostToDevice));

        // Launch kernel
        bfs_kernel<<<blocks, threads_per_block>>>(
            host_graph.num_vertices, 
            d_row_offsets, 
            d_column_indices, 
            d_distances, 
            current_level, 
            d_active_flag
        );
        
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check if any thread found an unvisited neighbor
        CUDA_CHECK(cudaMemcpy(&host_active_flag, d_active_flag, sizeof(bool), cudaMemcpyDeviceToHost));
        
        current_level++;
    }

    // 5. Copy Device -> Host (Fetch results)
    CUDA_CHECK(cudaMemcpy(host_distances.data(), d_distances, distances_size, cudaMemcpyDeviceToHost));

    // 6. Cleanup VRAM
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_active_flag));

    return host_distances;
}

} // namespace algorithms
} // namespace graph_engine
