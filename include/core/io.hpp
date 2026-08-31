#pragma once

#include <string>
#include "graph.hpp"

namespace graph_engine {
namespace core {

// Loads a graph from an edge list file (e.g. Matrix Market or plain text).
// Uses a 2-pass frequency counting algorithm to construct the CSR directly
// without requiring an expensive O(E log E) sorting step.
CSRGraph load_graph(const std::string& filepath);

// Saves a CSRGraph directly to a raw binary file for instantaneous future loading
void save_graph_binary(const CSRGraph& graph, const std::string& filepath);

// Loads a CSRGraph directly from a raw binary file, bypassing string parsing entirely
CSRGraph load_graph_binary(const std::string& filepath);

} // namespace core
} // namespace graph_engine
