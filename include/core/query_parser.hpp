#pragma once
#include "core/property_graph.hpp"
#include <vector>
#include <string>

namespace graph_engine {
namespace core {

struct QueryResult {
    std::vector<vertex_id_t> matched_vertices;
    long execution_time_ms;
};

class CypherParser {
public:
    // Parses and executes a basic Cypher query, e.g., "MATCH (n) WHERE n.age > 20 RETURN n"
    static QueryResult execute_query(const PropertyGraph& graph, const std::string& query);
};

} // namespace core
} // namespace graph_engine
