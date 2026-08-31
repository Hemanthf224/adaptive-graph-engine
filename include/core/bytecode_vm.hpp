#pragma once
#include "core/property_graph.hpp"
#include <vector>
#include <string>

namespace graph_engine {
namespace core {

enum class OpCode {
    OP_LOAD_PROP,   // Load a property value onto the stack
    OP_CONST_STR,   // Load a constant string onto the stack
    OP_CONST_NUM,   // Load a constant number onto the stack
    OP_CMP_EQ,      // Compare ==
    OP_CMP_GT,      // Compare >
    OP_CMP_LT,      // Compare <
    OP_YIELD,       // Yield the vertex if top of stack is true
    OP_HALT         // Stop execution
};

struct Instruction {
    OpCode opcode;
    std::string operand; // Simplified operand for MVP
};

class BytecodeVM {
public:
    // Compiles a basic Cypher WHERE clause into bytecode
    static std::vector<Instruction> compile(const std::string& condition, const std::string& op, const std::string& val);
    
    // Executes the bytecode against a specific vertex
    static bool execute(const std::vector<Instruction>& program, const PropertyGraph& graph, vertex_id_t v);
};

} // namespace core
} // namespace graph_engine
