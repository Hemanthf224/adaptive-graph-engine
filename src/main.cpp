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
#include "core/profiler.hpp"
#include <iomanip>

int main(int argc, char** argv) {
    graph_engine::core::Profiler::Get().BeginSession("AdaptiveGraph_Trace", "trace.json");

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
        
        bool run_benchmark = false;
        bool run_scaling = false;
        bool run_cc = false;
        bool run_sssp = false;
        int num_runs = 4; // 1 cold start + 3 hot runs
        std::string filepath = "";

        if (argc > 1) {
            filepath = argv[1];
            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--benchmark") run_benchmark = true;
                if (arg == "--scaling") run_scaling = true;
                if (arg == "--cc") run_cc = true;
                if (arg == "--sssp") run_sssp = true;
                if (arg == "--runs" && i + 1 < argc) {
                    num_runs = std::stoi(argv[++i]);
                }
            }
            
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

                } else {
                    std::cout << "\n--- Adaptive Engine Execution ---\n";
                    std::vector<int32_t> dist = graph_engine::core::AdaptiveScheduler::execute_bfs(graph, 0);
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
