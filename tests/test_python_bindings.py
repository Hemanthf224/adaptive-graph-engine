import sys
import os

# Assume the module is built in build/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))

try:
    import adaptive_graph as age
except ImportError as e:
    print(f"Error importing adaptive_graph. Did you build the PyBind11 module? {e}")
    sys.exit(1)

def main():
    print("========================================")
    print("Testing Adaptive Graph Python Bindings")
    print("========================================")

    # 1. Create a tiny test graph natively in C++
    print("\n[1] Creating Graph...")
    graph = age.create_tiny_test_graph()
    print(f"Graph loaded: {graph}")
    print(f"Vertices: {graph.num_vertices}, Edges: {graph.num_edges}")

    # 2. Run PageRank
    print("\n[2] Running OpenMP PageRank...")
    pr_scores = age.pagerank(graph, iterations=20, damping=0.85)
    print(f"PageRank Scores (Top 5): {pr_scores[:5]}")

    # 3. Run Direction-Optimizing BFS
    print("\n[3] Running Direction-Optimizing BFS...")
    distances = age.bfs(graph, source_vertex=0)
    print(f"BFS Distances from Vertex 0: {distances}")

    # 4. Connected Components
    print("\n[4] Finding Connected Components...")
    components = age.connected_components(graph)
    print(f"Component Labels (Top 5): {components[:5]}")

    print("\n[SUCCESS] Python bindings are fully operational!")

if __name__ == "__main__":
    main()
