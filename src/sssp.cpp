#include "algorithms/sssp.hpp"
#include <queue>
#include <stdexcept>
#include <iostream>

namespace graph_engine {
namespace algorithms {

std::vector<float> sssp_dijkstra(const core::CSRGraph& graph, core::vertex_id_t source) {
    if (!graph.is_weighted) {
        throw std::runtime_error("SSSP Dijkstra requires a weighted graph.");
    }
    
    size_t V = graph.num_vertices;
    std::vector<float> dist(V, std::numeric_limits<float>::infinity());
    
    // Min-heap priority queue: stores pairs of (distance, vertex)
    using pdi = std::pair<float, core::vertex_id_t>;
    std::priority_queue<pdi, std::vector<pdi>, std::greater<pdi>> pq;

    dist[source] = 0.0f;
    pq.push({0.0f, source});

    while (!pq.empty()) {
        float d = pq.top().first;
        core::vertex_id_t u = pq.top().second;
        pq.pop();

        // If we found a shorter path earlier, skip processing
        if (d > dist[u]) continue;

        core::edge_id_t start = graph.row_offsets[u];
        core::edge_id_t end = graph.row_offsets[u + 1];

        for (core::edge_id_t e = start; e < end; ++e) {
            core::vertex_id_t v = graph.column_indices[e];
            float weight = graph.edge_weights[e];

            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}

} // namespace algorithms
} // namespace graph_engine
