import os
import sys
import time

# Add the build directory to the Python path so it can find the .so module
sys.path.append(os.path.join(os.path.dirname(__file__), 'build', 'src'))

import adaptive_graph

print("========================================")
print("  Adaptive Graph Engine Python API Test")
print("========================================")

# Load the graph
print("\n[Python] Loading graph from C++ backend...")
start_time = time.time()
graph = adaptive_graph.load("data/amazon0302.txt")
load_time = time.time() - start_time
print(f"[Python] Graph loaded in {load_time:.4f} seconds.")

# Print stats using the C++ method
graph.print()

print("\n[Python] Executing 20 iterations of CUDA PageRank (Zero-Copy UVM)...")
start_time = time.time()
scores = adaptive_graph.pagerank_cuda(graph, 20, 0.85)
pr_time = time.time() - start_time
print(f"[Python] CUDA PageRank completed in {pr_time * 1000:.2f} ms.")

print("\n[Python] Sample Top Scores (First 10 nodes):")
for i in range(min(10, len(scores))):
    print(f"Node {i}: {scores[i]:.6f}")

print("\n[Python] API Test Complete! C++ engine successfully driven from Python.")
