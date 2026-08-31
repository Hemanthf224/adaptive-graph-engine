#include "algorithms/sketching.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace graph_engine {
namespace algorithms {

// ---- HyperLogLog ----

uint64_t HyperLogLog::fnv1a_hash(uint64_t x) {
    // FNV-1a 64-bit hash (fast, good avalanche)
    uint64_t h = 14695981039346656037ULL;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&x);
    for (int i = 0; i < 8; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int HyperLogLog::leading_zeros(uint64_t x) {
    if (x == 0) return 64;
    int count = 0;
    while ((x & (1ULL << 63)) == 0) { ++count; x <<= 1; }
    return count;
}

HyperLogLog::HyperLogLog(int precision) : p_(precision) {
    registers_.assign(1 << p_, 0);
}

void HyperLogLog::add(uint64_t item) {
    uint64_t h = fnv1a_hash(item);
    int idx = static_cast<int>(h >> (64 - p_)); // top p bits = register index
    uint64_t w = h << p_;                        // remaining bits
    uint8_t rho = static_cast<uint8_t>(leading_zeros(w) + 1);
    registers_[idx] = std::max(registers_[idx], rho);
}

double HyperLogLog::estimate() const {
    int M = 1 << p_;
    double alpha;
    if (M == 16)       alpha = 0.673;
    else if (M == 32)  alpha = 0.697;
    else if (M == 64)  alpha = 0.709;
    else               alpha = 0.7213 / (1.0 + 1.079 / M);

    double Z = 0.0;
    int V = 0;
    for (int j = 0; j < M; ++j) {
        Z += std::pow(2.0, -static_cast<double>(registers_[j]));
        if (registers_[j] == 0) ++V;
    }
    double E = alpha * M * M / Z;

    // Small range correction
    if (E <= 2.5 * M && V > 0) {
        E = M * std::log(static_cast<double>(M) / V);
    }
    return E;
}

void HyperLogLog::merge(const HyperLogLog& other) {
    for (size_t i = 0; i < registers_.size(); ++i) {
        registers_[i] = std::max(registers_[i], other.registers_[i]);
    }
}

// ---- Count-Min Sketch ----

CountMinSketch::CountMinSketch(int width, int depth)
    : width_(width), depth_(depth),
      table_(depth, std::vector<int64_t>(width, 0)) {
    // Generate different hash seeds for each row
    seeds_.resize(depth);
    uint64_t seed = 0xCAFEBABEDEADBEEFULL;
    for (int i = 0; i < depth; ++i) {
        seed = HyperLogLog::fnv1a_hash(seed + i);
        seeds_[i] = seed;
    }
}

uint64_t CountMinSketch::hash(uint64_t item, int row) const {
    return HyperLogLog::fnv1a_hash(item ^ seeds_[row]);
}

void CountMinSketch::update(uint64_t item, int64_t count) {
    for (int i = 0; i < depth_; ++i) {
        int col = static_cast<int>(hash(item, i) % width_);
        table_[i][col] += count;
    }
}

int64_t CountMinSketch::query(uint64_t item) const {
    int64_t min_val = INT64_MAX;
    for (int i = 0; i < depth_; ++i) {
        int col = static_cast<int>(hash(item, i) % width_);
        min_val = std::min(min_val, table_[i][col]);
    }
    return min_val;
}

// ---- Graph Sketcher ----

std::vector<double> GraphSketcher::estimate_degrees(const core::CSRGraph& graph, int precision) {
    size_t V = graph.num_vertices;
    std::vector<double> est_degrees(V, 0.0);

    #pragma omp parallel for schedule(dynamic, 512)
    for (core::vertex_id_t u = 0; u < V; ++u) {
        HyperLogLog hll(precision);
        core::edge_id_t start = graph.row_offsets[u];
        core::edge_id_t end   = graph.row_offsets[u + 1];
        for (core::edge_id_t e = start; e < end; ++e) {
            hll.add(static_cast<uint64_t>(graph.column_indices[e]));
        }
        est_degrees[u] = hll.estimate();
    }
    return est_degrees;
}

CountMinSketch GraphSketcher::build_edge_frequency_sketch(const core::CSRGraph& graph) {
    CountMinSketch sketch(4096, 7);
    for (core::vertex_id_t u = 0; u < graph.num_vertices; ++u) {
        core::edge_id_t start = graph.row_offsets[u];
        core::edge_id_t end   = graph.row_offsets[u + 1];
        for (core::edge_id_t e = start; e < end; ++e) {
            // Encode edge (u, v) as a single 64-bit key
            uint64_t key = (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(graph.column_indices[e]);
            sketch.update(key, 1);
        }
    }
    return sketch;
}

int64_t GraphSketcher::query_edge_frequency(const CountMinSketch& sketch, core::vertex_id_t u, core::vertex_id_t v) {
    uint64_t key = (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
    return sketch.query(key);
}

} // namespace algorithms
} // namespace graph_engine
