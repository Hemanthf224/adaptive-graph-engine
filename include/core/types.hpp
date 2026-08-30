#pragma once

#include <cstdint>
#include <vector>
#include "uvm_allocator.hpp"

namespace graph_engine {
namespace core {

// Node IDs up to 4.2 billion, suitable for most consumer graph datasets
// Keeps memory footprint low for GPU execution
using vertex_id_t = uint32_t;

// Edge IDs can exceed 4 billion easily in large graphs
using edge_id_t = uint64_t;

// Edge weights (if the graph is weighted)
using weight_t = float;

// A vector backed by CUDA Unified Memory
template <typename T>
using uvm_vector = std::vector<T, UVMAllocator<T>>;

} // namespace core
} // namespace graph_engine
