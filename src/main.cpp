#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>
#include "core/types.hpp"
#include "core/utils.hpp"
#include "core/graph.hpp"
#include "core/io.hpp"
#include "core/scheduler.hpp"
#include "algorithms/bfs.hpp"
#include "algorithms/bfs_cuda.cuh"
#include "algorithms/page_rank.hpp"
#include "algorithms/page_rank_cuda.cuh"
#include "algorithms/connected_components.hpp"
#include "algorithms/connected_components_cuda.cuh"
#include "algorithms/sssp.hpp"
#include "algorithms/triangle_count.hpp"
#include "core/profiler.hpp"
#include "core/arena.hpp"
#include "core/cxxopts.hpp"
#include <iomanip>

int main(int argc, char** argv) {
    graph_engine::core::Profiler::Get().BeginSession("AdaptiveGraph_Trace", "trace.json");

    // Initialize Global Memory Arena (1GB Pre-allocation for temporal data)
    graph_engine::core::LinearArenaAllocator global_arena(1024ULL * 1024ULL * 1024ULL);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Print environment details
    graph_engine::core::print_environment_info(mpi_rank, mpi_size);

    if (mpi_size > 1) {
        // ==========================================
        // MPI CLUSTER MODE
        // ==========================================
        if (argc > 1) {
            std::string filepath = argv[1];
            if (mpi_rank == 0) {
                std::cout << "\n======================================================\n";
                std::cout << "           MPI CLUSTER MODE INITIALIZED               \n";
                std::cout << "======================================================\n";
            }
            
            try {
                // Every node loads the graph (in a real system, we'd use parallel I/O, but this works for demo)
                graph_engine::core::CSRGraph graph = graph_engine::core::load_graph(filepath);
                
                // Synchronize before starting the timer
                MPI_Barrier(MPI_COMM_WORLD);
                auto start = std::chrono::high_resolution_clock::now();
                
                std::vector<float> pr_mpi = graph_engine::algorithms::pagerank_mpi(graph, 20);
                
                auto end = std::chrono::high_resolution_clock::now();
                double time_mpi = std::chrono::duration<double, std::milli>(end - start).count();

                if (mpi_rank == 0) {
                    std::cout << "[SUCCESS] MPI Distributed PageRank completed in " << time_mpi << " ms\n";
                }
            } catch (const std::exception& e) {
                if (mpi_rank == 0) std::cerr << "Error: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            if (mpi_rank == 0) std::cout << "Please provide a dataset path." << std::endl;
        }

    } else if (mpi_rank == 0) {
        // ==========================================
        // SINGLE-NODE MODE
        // ==========================================
        std::cout << "Starting Adaptive Graph Engine (Single Node)...\n" << std::endl;
        
        cxxopts::Options options("graph_engine", "Adaptive Graph Engine");
        options.allow_unrecognised_options();
        options.add_options()
            ("benchmark", "Run full benchmarking suite")
            ("scaling", "Run Amdahl's Law thread scaling analysis")
            ("cc", "Run Connected Components")
            ("sssp", "Run Single Source Shortest Path")
            ("triangles", "Run Triangle Counting")
            ("runs", "Number of benchmark iterations", cxxopts::value<int>()->default_value("4"));

        auto result = options.parse(argc, argv);
        
        bool run_benchmark = result.count("benchmark");
        bool run_scaling = result.count("scaling");
        bool run_cc = result.count("cc");
        bool run_sssp = result.count("sssp");
        bool run_triangles = result.count("triangles");
        int num_runs = result["runs"].as<int>();
        std::string filepath = "";

        if (argc > 1 && argv[1][0] != '-') {
            filepath = argv[1];
        }

        if (!filepath.empty()) {
            std::cout << "Loading graph from: " << filepath << std::endl;
            try {
                graph_engine::core::CSRGraph graph = graph_engine::core::load_graph(filepath);
                
                if (run_benchmark) {
                    std::cout << "\n======================================================\n";
                    std::cout << "           PAGERANK BENCHMARK SUITE STARTED           \n";
                    std::cout << "======================================================\n";
                    std::cout << "Graph: " << graph.num_vertices << " Vertices, " << graph.num_edges << " Edges\n";
                    std::cout << "Compute-Bound Workload: 20 Iterations of PageRank\n";
                    std::cout << "Statistical Mode: " << num_runs << " runs (first run discarded as cold-start)\n\n";

                    double avg_time_seq = 0.0;
                    double avg_time_omp = 0.0;
                    double avg_time_cuda = 0.0;
                    int valid_runs = (num_runs > 1) ? (num_runs - 1) : 1;

                    // 1. Sequential
                    std::cout << "[Running] Sequential PageRank (" << num_runs << " runs)...\n";
                    for (int r = 0; r < num_runs; ++r) {
                        auto start_seq = std::chrono::high_resolution_clock::now();
                        std::vector<float> pr_seq = graph_engine::algorithms::pagerank_sequential(graph, 20);
                        auto end_seq = std::chrono::high_resolution_clock::now();
                        double time_seq = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();
                        if (num_runs == 1 || r > 0) avg_time_seq += time_seq;
                    }
                    avg_time_seq /= valid_runs;

                    // 2. OpenMP
                    std::cout << "[Running] OpenMP PageRank (24 Threads, " << num_runs << " runs)...\n";
                    for (int r = 0; r < num_runs; ++r) {
                        auto start_omp = std::chrono::high_resolution_clock::now();
                        std::vector<float> pr_omp = graph_engine::algorithms::pagerank_openmp(graph, 20);
                        auto end_omp = std::chrono::high_resolution_clock::now();
                        double time_omp = std::chrono::duration<double, std::milli>(end_omp - start_omp).count();
                        if (num_runs == 1 || r > 0) avg_time_omp += time_omp;
                    }
                    avg_time_omp /= valid_runs;

                    // 3. CUDA (Unified Memory)
                    std::cout << "[Running] CUDA PageRank (UVM, 20 Iterations, " << num_runs << " runs)...\n";
                    
                    // WARM START: Prefetch UVM Memory to the GPU before measuring performance
                    int deviceId = 0;
                    graph_engine::algorithms::prefetch_graph_to_gpu(graph, deviceId);

                    for (int r = 0; r < num_runs; ++r) {
                        auto start_cuda = std::chrono::high_resolution_clock::now();
                        std::vector<float> pr_cuda = graph_engine::algorithms::pagerank_cuda(graph, 20);
                        auto end_cuda = std::chrono::high_resolution_clock::now();
                        double time_cuda = std::chrono::duration<double, std::milli>(end_cuda - start_cuda).count();
                        if (num_runs == 1 || r > 0) avg_time_cuda += time_cuda;
                    }
                    avg_time_cuda /= valid_runs;

                    // 4. CUDA (Explicit Copy)
                    double avg_time_cuda_exp = 0.0;
                    std::cout << "[Running] CUDA PageRank (Explicit, 20 Iterations, " << num_runs << " runs)...\n";
                    
                    for (int r = 0; r < num_runs; ++r) {
                        auto start_cuda_exp = std::chrono::high_resolution_clock::now();
                        std::vector<float> pr_cuda_exp = graph_engine::algorithms::pagerank_cuda_explicit(graph, 20);
                        auto end_cuda_exp = std::chrono::high_resolution_clock::now();
                        double time_cuda_exp = std::chrono::duration<double, std::milli>(end_cuda_exp - start_cuda_exp).count();
                        if (num_runs == 1 || r > 0) avg_time_cuda_exp += time_cuda_exp;
                    }
                    avg_time_cuda_exp /= valid_runs;

                    std::cout << "\n================ PERFORMANCE REPORT ================\n";
                    std::cout << std::left << std::setw(20) << "Hardware" << std::setw(20) << "Avg Time (ms)" << "Speedup vs Seq\n";
                    std::cout << "----------------------------------------------------\n";
                    std::cout << std::left << std::setw(20) << "CPU Sequential" << std::setw(20) << avg_time_seq << "1.0x\n";
                    std::cout << std::left << std::setw(20) << "CPU OpenMP"     << std::setw(20) << avg_time_omp << (avg_time_seq/avg_time_omp) << "x\n";
                    std::cout << std::left << std::setw(20) << "GPU CUDA (UVM)" << std::setw(20) << avg_time_cuda << (avg_time_seq/avg_time_cuda) << "x\n";
                    std::cout << std::left << std::setw(20) << "GPU CUDA (Expl)" << std::setw(20) << avg_time_cuda_exp << (avg_time_seq/avg_time_cuda_exp) << "x\n";
                    std::cout << "====================================================\n";

                } else if (run_scaling) {
                    std::cout << "\n======================================================\n";
                    std::cout << "           OPENMP STRONG SCALING ANALYSIS             \n";
                    std::cout << "======================================================\n";
                    std::cout << "Graph: " << graph.num_vertices << " Vertices, " << graph.num_edges << " Edges\n";
                    std::cout << "Compute-Bound Workload: 20 Iterations of PageRank\n\n";

                    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 24};
                    for (int threads : thread_counts) {
                        omp_set_num_threads(threads);
                        std::cout << "[Running] OpenMP PageRank (" << threads << " Threads)...\n";
                        
                        auto start_omp = std::chrono::high_resolution_clock::now();
                        std::vector<float> pr_omp = graph_engine::algorithms::pagerank_openmp(graph, 20);
                        auto end_omp = std::chrono::high_resolution_clock::now();
                        
                        double time_omp = std::chrono::duration<double, std::milli>(end_omp - start_omp).count();
                        std::cout << "[SCALING_RESULT] Threads: " << threads << " | Time: " << time_omp << " ms\n";
                    }

                } else if (run_cc) {
                    std::cout << "\n======================================================\n";
                    std::cout << "        CONNECTED COMPONENTS BENCHMARK SUITE          \n";
                    std::cout << "======================================================\n";
                    std::cout << "Graph: " << graph.num_vertices << " Vertices, " << graph.num_edges << " Edges\n";
                    std::cout << "Algorithm: Push-Based Label Propagation\n\n";

                    // 1. Sequential
                    std::cout << "[Running] Sequential Connected Components...\n";
                    auto start_seq = std::chrono::high_resolution_clock::now();
                    std::vector<int> cc_seq = graph_engine::algorithms::connected_components_sequential(graph);
                    auto end_seq = std::chrono::high_resolution_clock::now();
                    double time_seq = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

                    // 2. OpenMP
                    std::cout << "[Running] OpenMP Connected Components...\n";
                    auto start_omp = std::chrono::high_resolution_clock::now();
                    std::vector<int> cc_omp = graph_engine::algorithms::connected_components_openmp(graph);
                    auto end_omp = std::chrono::high_resolution_clock::now();
                    double time_omp = std::chrono::duration<double, std::milli>(end_omp - start_omp).count();

                    // 3. CUDA
                    std::cout << "[Running] CUDA Connected Components (UVM)...\n";
                    auto start_cuda = std::chrono::high_resolution_clock::now();
                    std::vector<int> cc_cuda = graph_engine::algorithms::connected_components_cuda(graph);
                    auto end_cuda = std::chrono::high_resolution_clock::now();
                    double time_cuda = std::chrono::duration<double, std::milli>(end_cuda - start_cuda).count();

                    std::cout << "\n================ PERFORMANCE REPORT ================\n";
                    std::cout << std::left << std::setw(20) << "Hardware" << std::setw(20) << "Avg Time (ms)" << "Speedup vs Seq\n";
                    std::cout << "----------------------------------------------------\n";
                    std::cout << std::left << std::setw(20) << "CPU Sequential" << std::setw(20) << time_seq << "1.0x\n";
                    std::cout << std::left << std::setw(20) << "CPU OpenMP"     << std::setw(20) << time_omp << (time_seq/time_omp) << "x\n";
                    std::cout << std::left << std::setw(20) << "GPU CUDA (UVM)" << std::setw(20) << time_cuda << (time_seq/time_cuda) << "x\n";
                    std::cout << "====================================================\n";

                } else if (run_sssp) {
                    std::cout << "\n======================================================\n";
                    std::cout << "        SINGLE-SOURCE SHORTEST PATH BENCHMARK         \n";
                    std::cout << "======================================================\n";
                    std::cout << "Graph: " << graph.num_vertices << " Vertices, " << graph.num_edges << " Edges\n";
                    std::cout << "Algorithm: Dijkstra's SSSP (Sequential Priority Queue)\n\n";

                    std::cout << "[Running] Sequential Dijkstra from Source Node 0...\n";
                    auto start_seq = std::chrono::high_resolution_clock::now();
                    std::vector<float> dist = graph_engine::algorithms::sssp_dijkstra(graph, 0);
                    auto end_seq = std::chrono::high_resolution_clock::now();
                    double time_seq = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

                    std::cout << "\n================ PERFORMANCE REPORT ================\n";
                    std::cout << std::left << std::setw(20) << "Hardware" << std::setw(20) << "Avg Time (ms)" << "Speedup vs Seq\n";
                    std::cout << "----------------------------------------------------\n";
                    std::cout << std::left << std::setw(20) << "CPU Sequential" << std::setw(20) << time_seq << "1.0x\n";
                    std::cout << "====================================================\n";

                } else if (run_triangles) {
                    std::cout << "\n======================================================\n";
                    std::cout << "        TRIANGLE COUNTING (COMMUNITY DETECTION)       \n";
                    std::cout << "======================================================\n";
                    std::cout << "Graph: " << graph.num_vertices << " Vertices, " << graph.num_edges << " Edges\n";
                    
                    auto start_omp = std::chrono::high_resolution_clock::now();
                    uint64_t triangles = graph_engine::algorithms::triangle_counting_openmp(graph);
                    auto end_omp = std::chrono::high_resolution_clock::now();
                    double time_omp = std::chrono::duration<double, std::milli>(end_omp - start_omp).count();
                    
                    std::cout << "[SUCCESS] Found " << triangles << " triangles.\n";
                    std::cout << "[INFO] OpenMP Time : " << time_omp << " ms\n";

                } else {
                    std::cout << "\n======================================================\n";
                    std::cout << "        DIRECTION-OPTIMIZING BFS (GRAPH500)           \n";
                    std::cout << "======================================================\n";
                    std::cout << "Graph: " << graph.num_vertices << " Vertices, " << graph.num_edges << " Edges\n";

                    auto start_do = std::chrono::high_resolution_clock::now();
                    std::vector<int32_t> dist = graph_engine::algorithms::bfs_direction_optimizing(graph, 0);
                    auto end_do = std::chrono::high_resolution_clock::now();
                    double time_do = std::chrono::duration<double, std::milli>(end_do - start_do).count();

                    std::cout << "[SUCCESS] Direction-Optimizing BFS completed in " << time_do << " ms.\n";
                }

            } catch (const std::exception& e) {
                std::cerr << "Error loading graph: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            std::cout << "No file provided. Initializing tiny test graph (CSR Format)..." << std::endl;
            graph_engine::core::CSRGraph graph = graph_engine::core::create_tiny_test_graph();
            std::cout << "\n--- Adaptive Engine Execution ---\n";
            std::vector<int32_t> dist = graph_engine::core::AdaptiveScheduler::execute_bfs(graph, 0);
        }

        std::cout << "\nEngine ready for operations." << std::endl;
    }

    // Finalize MPI
    graph_engine::core::Profiler::Get().EndSession();
    MPI_Finalize();
    return 0;
}
