#include "algorithms/page_rank_cuda.cuh"
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

__global__ void pagerank_reset_kernel(
    core::vertex_id_t num_vertices,
    float* next_scores,
    float base_score) 
{
    core::vertex_id_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < num_vertices) {
        next_scores[u] = base_score;
    }
}

__global__ void pagerank_push_kernel(
    core::vertex_id_t num_vertices,
    const core::edge_id_t* row_offsets,
    const core::vertex_id_t* column_indices,
    const float* scores,
    float* next_scores,
    float damping) 
{
    core::vertex_id_t u = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (u < num_vertices) {
        core::edge_id_t start = row_offsets[u];
        core::edge_id_t end = row_offsets[u + 1];
        core::edge_id_t degree = end - start;

        if (degree > 0) {
            float push_val = damping * (scores[u] / (float)degree);
            for (core::edge_id_t e = start; e < end; ++e) {
                core::vertex_id_t v = column_indices[e];
                atomicAdd(&next_scores[v], push_val);
            }
        }
    }
}

std::vector<float> pagerank_cuda(const core::CSRGraph& host_graph, int iterations, float damping) {
    size_t V = host_graph.num_vertices;
    std::vector<float> host_scores(V, 1.0f / V);
    float base_score = (1.0f - damping) / V;

    // Device Pointers
    float* d_scores = nullptr;
    float* d_next_scores = nullptr;

    // 1. Allocate Device Memory (Only for score arrays)
    size_t scores_size = V * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
    CUDA_CHECK(cudaMalloc(&d_next_scores, scores_size));

    // 2. Copy Host -> Device (Scores only, graph is UVM)
    CUDA_CHECK(cudaMemcpy(d_scores, host_scores.data(), scores_size, cudaMemcpyHostToDevice));

    // Zero-Copy pointers for the kernel
    const core::edge_id_t* d_row_offsets = host_graph.row_offsets.data();
    const core::vertex_id_t* d_column_indices = host_graph.column_indices.data();

    // 3. Launch configuration
    int threads_per_block = 256;
    int blocks = (V + threads_per_block - 1) / threads_per_block;

    // 4. Execution Loop completely in VRAM/UVM
    for (int iter = 0; iter < iterations; ++iter) {
        // Reset next_scores
        pagerank_reset_kernel<<<blocks, threads_per_block>>>(V, d_next_scores, base_score);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Push scores
        pagerank_push_kernel<<<blocks, threads_per_block>>>(
            V, 
            d_row_offsets, 
            d_column_indices, 
            d_scores, 
            d_next_scores, 
            damping
        );
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers for the next iteration
        float* temp = d_scores;
        d_scores = d_next_scores;
        d_next_scores = temp;
    }

    // 5. Copy Device -> Host (Fetch results)
    CUDA_CHECK(cudaMemcpy(host_scores.data(), d_scores, scores_size, cudaMemcpyDeviceToHost));

    // 6. Cleanup VRAM
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_next_scores));

    return host_scores;
}

} // namespace algorithms
} // namespace graph_engine

namespace graph_engine {
namespace algorithms {

std::vector<float> pagerank_cuda_explicit(const core::CSRGraph& host_graph, int iterations, float damping) {
    size_t V = host_graph.num_vertices;
    size_t E = host_graph.num_edges;
    std::vector<float> host_scores(V, 1.0f / V);
    float base_score = (1.0f - damping) / V;

    // Device Pointers
    float* d_scores = nullptr;
    float* d_next_scores = nullptr;
    core::edge_id_t* d_row_offsets = nullptr;
    core::vertex_id_t* d_column_indices = nullptr;

    // 1. Allocate Device Memory (For scores AND the entire graph)
    size_t scores_size = V * sizeof(float);
    size_t row_offsets_size = (V + 1) * sizeof(core::edge_id_t);
    size_t column_indices_size = E * sizeof(core::vertex_id_t);

    CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
    CUDA_CHECK(cudaMalloc(&d_next_scores, scores_size));
    CUDA_CHECK(cudaMalloc(&d_row_offsets, row_offsets_size));
    CUDA_CHECK(cudaMalloc(&d_column_indices, column_indices_size));

    // 2. Explicitly Copy Host -> Device (Scores and full Graph Topology)
    CUDA_CHECK(cudaMemcpy(d_scores, host_scores.data(), scores_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_offsets, host_graph.row_offsets.data(), row_offsets_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_indices, host_graph.column_indices.data(), column_indices_size, cudaMemcpyHostToDevice));

    // 3. Launch configuration
    int threads_per_block = 256;
    int blocks = (V + threads_per_block - 1) / threads_per_block;

    // 4. Execution Loop
    for (int iter = 0; iter < iterations; ++iter) {
        pagerank_reset_kernel<<<blocks, threads_per_block>>>(V, d_next_scores, base_score);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        pagerank_push_kernel<<<blocks, threads_per_block>>>(
            V, 
            d_row_offsets, 
            d_column_indices, 
            d_scores, 
            d_next_scores, 
            damping
        );
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float* temp = d_scores;
        d_scores = d_next_scores;
        d_next_scores = temp;
    }

    // 5. Copy Device -> Host (Fetch results)
    CUDA_CHECK(cudaMemcpy(host_scores.data(), d_scores, scores_size, cudaMemcpyDeviceToHost));

    // 6. Cleanup VRAM (Including the graph topology)
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_next_scores));
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_column_indices));

    return host_scores;
}

} // namespace algorithms
} // namespace graph_engine

namespace graph_engine {
namespace algorithms {

void prefetch_graph_to_gpu(const core::CSRGraph& graph, int device_id) {
    // Explicitly prefetch UVM data to the GPU via a warm-up execution
    // This avoids PCIe page faults during the actual timed benchmark
    pagerank_cuda(graph, 1);
}

} // namespace algorithms
} // namespace graph_engine
