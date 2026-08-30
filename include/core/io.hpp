#pragma once

#include <string>
#include "graph.hpp"

namespace graph_engine {
namespace core {

// Loads a graph from an edge list file (e.g. Matrix Market or plain text).
// Uses a 2-pass frequency counting algorithm to construct the CSR directly
// without requiring an expensive O(E log E) sorting step.
CSRGraph load_graph(const std::string& filepath);

} // namespace core
} // namespace graph_engine
