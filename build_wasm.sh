#!/bin/bash
# ---------------------------------------------------------
# WebAssembly Compilation Script for Adaptive Graph Engine
# ---------------------------------------------------------

# Requires Emscripten SDK (emsdk)
# Run `emsdk activate latest` before running this script.

echo "Compiling Adaptive Graph Engine to WebAssembly (Wasm)..."

mkdir -p build_wasm

emcc -O3 -std=c++17 \
    -I./include \
    src/wasm_bindings.cpp \
    src/page_rank.cpp \
    src/bfs.cpp \
    src/graph.cpp \
    -lembind \
    -o build_wasm/adaptive_graph.js \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="createAdaptiveGraph" \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=268435456 # 256MB

echo "Compilation Complete! adaptive_graph.js and adaptive_graph.wasm generated in build_wasm/"
