#pragma once
#include "core/graph.hpp"
#include <vector>
#include <cstdint>

namespace graph_engine {
namespace algorithms {

// ----------------------------------------------------------------
// HyperLogLog: Estimates distinct neighbor count using O(log log N) space
// Used by Redis, Google BigQuery, Facebook for cardinality estimation
// ----------------------------------------------------------------
class HyperLogLog {
public:
    explicit HyperLogLog(int precision = 10); // precision p => 2^p registers
    
    void add(uint64_t item);
    double estimate() const;
    void merge(const HyperLogLog& other);

private:
    int p_; // precision bits
    std::vector<uint8_t> registers_; // M = 2^p registers
    static uint64_t fnv1a_hash(uint64_t x);
    static int leading_zeros(uint64_t x);
};

// ----------------------------------------------------------------
// Count-Min Sketch: Estimates edge/event frequencies in a data stream
// Uses O(epsilon^-1 * log(1/delta)) space
// Used by Facebook trending topics, network traffic analysis
// ----------------------------------------------------------------
class CountMinSketch {
public:
    CountMinSketch(int width = 2048, int depth = 5);
    
    void update(uint64_t item, int64_t count = 1);
    int64_t query(uint64_t item) const;

private:
    int width_, depth_;
    std::vector<std::vector<int64_t>> table_;
    std::vector<uint64_t> seeds_; // Hash seeds
    uint64_t hash(uint64_t item, int row) const;
};

// ----------------------------------------------------------------
// Graph Sketching Engine: Applies sketching to graph analytics
// ----------------------------------------------------------------
class GraphSketcher {
public:
    // Use HyperLogLog to estimate distinct neighbors (approximate degree)
    // Returns estimated degree for each vertex
    static std::vector<double> estimate_degrees(const core::CSRGraph& graph, int precision = 10);
    
    // Use Count-Min Sketch to count edge frequencies in a stream of updates
    static CountMinSketch build_edge_frequency_sketch(const core::CSRGraph& graph);
    
    // Query estimated frequency of a specific edge (u, v)
    static int64_t query_edge_frequency(const CountMinSketch& sketch, core::vertex_id_t u, core::vertex_id_t v);
};

} // namespace algorithms
} // namespace graph_engine
