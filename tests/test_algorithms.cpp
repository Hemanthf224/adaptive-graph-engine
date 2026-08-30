#include <gtest/gtest.h>
#include "core/graph.hpp"
#include "algorithms/bfs.hpp"
#include "algorithms/bfs_cuda.cuh"
#include "algorithms/page_rank.hpp"
#include "algorithms/page_rank_cuda.cuh"

using namespace graph_engine;

class AlgorithmTest : public ::testing::Test {
protected:
    core::CSRGraph graph;

    void SetUp() override {
        // Create a simple deterministic graph for testing
        // 0 -> 1, 2
        // 1 -> 0, 3
        // 2 -> 3, 4
        // 3 -> 1
        // 4 -> (none)
        graph.num_vertices = 5;
        graph.num_edges = 8;
        
        graph.row_offsets.push_back(0);
        graph.row_offsets.push_back(2);
        graph.row_offsets.push_back(4);
        graph.row_offsets.push_back(6);
        graph.row_offsets.push_back(7);
        graph.row_offsets.push_back(8); // actually, vertex 4 has no outgoing edges, so row_offsets should be 8 at both 4 and 5 if 4 is dangling.
        
        // Fix the row offsets for the topology:
        graph.row_offsets = {0, 2, 4, 6, 7, 7};
        graph.column_indices = {1, 2, 0, 3, 3, 4, 1};
        // wait, that's 7 edges. Let's just use the tiny test graph function!
        graph = core::create_tiny_test_graph();
    }
};

TEST_F(AlgorithmTest, BFS_Correctness) {
    auto dist_seq = algorithms::bfs_sequential(graph, 0);
    auto dist_omp = algorithms::bfs_openmp(graph, 0);
    auto dist_cuda = algorithms::bfs_cuda(graph, 0);

    // Verify all hardware paths yield identical exact distances
    for (size_t i = 0; i < graph.num_vertices; ++i) {
        EXPECT_EQ(dist_seq[i], dist_omp[i]) << "Mismatch at vertex " << i << " (OpenMP)";
        EXPECT_EQ(dist_seq[i], dist_cuda[i]) << "Mismatch at vertex " << i << " (CUDA)";
    }
}

TEST_F(AlgorithmTest, PageRank_Correctness) {
    auto pr_seq = algorithms::pagerank_sequential(graph, 20, 0.85f);
    auto pr_omp = algorithms::pagerank_openmp(graph, 20, 0.85f);
    auto pr_cuda = algorithms::pagerank_cuda(graph, 20, 0.85f);

    // Verify all hardware paths yield identical floating point values (within epsilon)
    for (size_t i = 0; i < graph.num_vertices; ++i) {
        EXPECT_NEAR(pr_seq[i], pr_omp[i], 1e-4) << "Mismatch at vertex " << i << " (OpenMP)";
        EXPECT_NEAR(pr_seq[i], pr_cuda[i], 1e-4) << "Mismatch at vertex " << i << " (CUDA)";
    }
}
