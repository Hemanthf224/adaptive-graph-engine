#pragma once
#include "core/graph.hpp"
#include <vector>

namespace graph_engine {
namespace algorithms {

struct PartitionResult {
    std::vector<int> partition_assignments; // 0 or 1 for each vertex
    size_t edge_cut; // Number of edges crossing the partition
};

class GraphPartitioner {
public:
    // Performs a basic Kernighan-Lin style bisection heuristic
    static PartitionResult kernighan_lin_bisection(const core::CSRGraph& graph, int iterations = 10);
};

} // namespace algorithms
} // namespace graph_engine
