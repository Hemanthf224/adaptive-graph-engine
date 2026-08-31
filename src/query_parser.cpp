#include "core/query_parser.hpp"
#include <sstream>
#include <iostream>
#include <chrono>
#include <omp.h>

namespace graph_engine {
namespace core {

QueryResult CypherParser::execute_query(const PropertyGraph& graph, const std::string& query) {
    auto start = std::chrono::high_resolution_clock::now();
    QueryResult result;

    // Very basic Lexer: Tokenize by space
    std::istringstream iss(query);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }

    // Expected simple syntax: MATCH (n) WHERE n.key = value RETURN n
    if (tokens.size() < 6 || tokens[0] != "MATCH" || tokens[2] != "WHERE") {
        std::cerr << "[Query Error] Unsupported Syntax. Use: MATCH (n) WHERE n.key = value RETURN n\n";
        return result;
    }

    std::string condition = tokens[3]; // e.g., "n.age"
    std::string op = tokens[4];        // e.g., "=", ">"
    std::string val = tokens[5];       // e.g., "20"
    
    // Parse key from "n.key"
    size_t dot_pos = condition.find('.');
    if (dot_pos == std::string::npos) {
        std::cerr << "[Query Error] Invalid property accessor: " << condition << "\n";
        return result;
    }
    std::string key = condition.substr(dot_pos + 1);

    // Prepare thread-local result vectors to avoid atomic contention
    std::vector<std::vector<vertex_id_t>> local_results;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            local_results.resize(omp_get_num_threads());
        }
        
        int tid = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic, 1024)
        for (vertex_id_t v = 0; v < graph.num_vertices; ++v) {
            std::string prop_val = graph.get_vertex_property(v, key);
            if (prop_val.empty()) continue;

            bool match = false;
            if (op == "=" || op == "==") {
                match = (prop_val == val);
            } else if (op == ">") {
                try { match = (std::stod(prop_val) > std::stod(val)); } catch (...) {}
            } else if (op == "<") {
                try { match = (std::stod(prop_val) < std::stod(val)); } catch (...) {}
            }
            
            if (match) {
                local_results[tid].push_back(v);
            }
        }
    }

    // Merge local results
    for (const auto& local : local_results) {
        result.matched_vertices.insert(result.matched_vertices.end(), local.begin(), local.end());
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return result;
}

} // namespace core
} // namespace graph_engine
