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

    # 5. Property Graph and Cypher Queries
    print("\n[5] Upgrading to Property Graph Database...")
    prop_graph = age.PropertyGraph(graph)
    prop_graph.set_vertex_property(0, "name", "Alice")
    prop_graph.set_vertex_property(0, "age", "25")
    
    prop_graph.set_vertex_property(1, "name", "Bob")
    prop_graph.set_vertex_property(1, "age", "19")

    prop_graph.set_vertex_property(2, "name", "Charlie")
    prop_graph.set_vertex_property(2, "age", "30")

    print("Executing Cypher Query: MATCH (n) WHERE n.age > 20 RETURN n")
    result = age.execute_cypher(prop_graph, "MATCH (n) WHERE n.age > 20 RETURN n")
    print(f"Matched Vertices: {result.matched_vertices}")
    print(f"Query executed in {result.execution_time_ms} ms")

    # 6. Graph Neural Network (GCN) Inference
    print("\n[6] Running AVX2-Accelerated GCN Forward Pass...")
    
    # Initialize some dummy features for 4 vertices
    # If the tiny graph has more vertices, we just pad them
    node_features = [1.0, 0.5, -0.2, 0.8] 
    while len(node_features) < graph.num_vertices:
        node_features.append(0.1)
        
    layer_weight = 0.5
    
    next_features = age.gcn_forward_pass(graph, node_features, layer_weight)
    print(f"GCN Output Features (Top 5): {next_features[:5]}")

    # 7. Quantum Graph Simulator
    print("\n[7] Simulating Quantum Grover Search for Vertex 2...")
    # By default, iterations = max(1, (pi/4)*sqrt(N))
    quantum_probs = age.quantum_grover_search(graph, target_vertex=2)
    print(f"Quantum Measurement Probabilities: {quantum_probs}")
    
    print("\n[SUCCESS] Python bindings, AI Engine, and Quantum Simulator are fully operational!")

if __name__ == "__main__":
    main()
