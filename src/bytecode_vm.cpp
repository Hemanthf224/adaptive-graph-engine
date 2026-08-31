#include "core/bytecode_vm.hpp"
#include <stack>
#include <stdexcept>
#include <iostream>

namespace graph_engine {
namespace core {

std::vector<Instruction> BytecodeVM::compile(const std::string& condition, const std::string& op, const std::string& val) {
    std::vector<Instruction> program;
    
    // Parse key from "n.key"
    size_t dot_pos = condition.find('.');
    std::string key = (dot_pos != std::string::npos) ? condition.substr(dot_pos + 1) : condition;

    // 1. Load the property from the graph
    program.push_back({OpCode::OP_LOAD_PROP, key});
    
    // 2. Load the constant value to compare against
    program.push_back({OpCode::OP_CONST_STR, val});
    
    // 3. Compare
    if (op == "=" || op == "==") {
        program.push_back({OpCode::OP_CMP_EQ, ""});
    } else if (op == ">") {
        program.push_back({OpCode::OP_CMP_GT, ""});
    } else if (op == "<") {
        program.push_back({OpCode::OP_CMP_LT, ""});
    } else {
        throw std::runtime_error("Unsupported Operator in Bytecode Compiler");
    }
    
    // 4. Yield and Halt
    program.push_back({OpCode::OP_YIELD, ""});
    program.push_back({OpCode::OP_HALT, ""});
    
    return program;
}

bool BytecodeVM::execute(const std::vector<Instruction>& program, const PropertyGraph& graph, vertex_id_t v) {
    std::stack<std::string> stack;
    
    for (const auto& inst : program) {
        switch (inst.opcode) {
            case OpCode::OP_LOAD_PROP: {
                stack.push(graph.get_vertex_property(v, inst.operand));
                break;
            }
            case OpCode::OP_CONST_STR: {
                stack.push(inst.operand);
                break;
            }
            case OpCode::OP_CMP_EQ: {
                if (stack.size() < 2) return false;
                std::string right = stack.top(); stack.pop();
                std::string left = stack.top(); stack.pop();
                stack.push((left == right) ? "1" : "0");
                break;
            }
            case OpCode::OP_CMP_GT: {
                if (stack.size() < 2) return false;
                std::string right = stack.top(); stack.pop();
                std::string left = stack.top(); stack.pop();
                try {
                    bool res = std::stod(left) > std::stod(right);
                    stack.push(res ? "1" : "0");
                } catch (...) { stack.push("0"); }
                break;
            }
            case OpCode::OP_CMP_LT: {
                if (stack.size() < 2) return false;
                std::string right = stack.top(); stack.pop();
                std::string left = stack.top(); stack.pop();
                try {
                    bool res = std::stod(left) < std::stod(right);
                    stack.push(res ? "1" : "0");
                } catch (...) { stack.push("0"); }
                break;
            }
            case OpCode::OP_YIELD: {
                if (stack.empty()) return false;
                return stack.top() == "1";
            }
            case OpCode::OP_HALT: {
                return false;
            }
        }
    }
    return false;
}

} // namespace core
} // namespace graph_engine
