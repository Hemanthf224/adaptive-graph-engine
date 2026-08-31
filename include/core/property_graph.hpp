#pragma once
#include "core/graph.hpp"
#include <unordered_map>
#include <string>
#include <variant>

namespace graph_engine {
namespace core {

// We will store properties as strings for simplicity in this mini-database
using PropertyMap = std::unordered_map<std::string, std::string>;

struct PropertyGraph : public CSRGraph {
    // Maps vertex_id to a dictionary of properties
    std::unordered_map<vertex_id_t, PropertyMap> vertex_properties;
    
    // Maps edge_id to a dictionary of properties
    std::unordered_map<edge_id_t, PropertyMap> edge_properties;

    // Convert standard CSRGraph to PropertyGraph
    PropertyGraph() = default;
    
    PropertyGraph(const CSRGraph& base) : CSRGraph(base) {}

    void set_vertex_property(vertex_id_t v, const std::string& key, const std::string& value) {
        vertex_properties[v][key] = value;
    }

    std::string get_vertex_property(vertex_id_t v, const std::string& key) const {
        auto it = vertex_properties.find(v);
        if (it != vertex_properties.end()) {
            auto prop_it = it->second.find(key);
            if (prop_it != it->second.end()) {
                return prop_it->second;
            }
        }
        return "";
    }
};

} // namespace core
} // namespace graph_engine
