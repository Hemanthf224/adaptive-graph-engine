#include "algorithms/connected_components_cuda.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace graph_engine {
namespace algorithms {

// CUDA Kernel: Initialize Labels
__global__ void init_labels_kernel(int* labels, size_t V) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        labels[tid] = tid;
    }
}

// CUDA Kernel: Propagate Labels
__global__ void propagate_labels_kernel(
    const core::edge_id_t* row_offsets,
    const core::vertex_id_t* column_indices,
    int* labels,
    size_t V,
    bool* changed) 
{
    size_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < V) {
        core::edge_id_t start = row_offsets[u];
        core::edge_id_t end = row_offsets[u + 1];
        
        int label_u = labels[u];

        for (core::edge_id_t e = start; e < end; ++e) {
            core::vertex_id_t v = column_indices[e];
            int label_v = labels[v];

            if (label_u < label_v) {
                atomicMin(&labels[v], label_u);
                *changed = true;
            } else if (label_v < label_u) {
                atomicMin(&labels[u], label_v);
                *changed = true;
                label_u = labels[u]; // refresh local after min
            }
        }
    }
}

std::vector<int> connected_components_cuda(const core::CSRGraph& graph) {
    size_t V = graph.num_vertices;
    
    int* d_labels;
    bool* d_changed;
    cudaMalloc(&d_labels, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));

    int threads = 256;
    int blocks = (V + threads - 1) / threads;

    // Initialize
    init_labels_kernel<<<blocks, threads>>>(d_labels, V);
    cudaDeviceSynchronize();

    bool h_changed = true;
    while (h_changed) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        propagate_labels_kernel<<<blocks, threads>>>(
            graph.row_offsets.data(),
            graph.column_indices.data(),
            d_labels,
            V,
            d_changed
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    std::vector<int> h_labels(V);
    cudaMemcpy(h_labels.data(), d_labels, V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_labels);
    cudaFree(d_changed);

    return h_labels;
}

} // namespace algorithms
} // namespace graph_engine
