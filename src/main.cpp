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
#include <iomanip>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Print environment details
    graph_engine::core::print_environment_info(mpi_rank, mpi_size);

    if (mpi_rank == 0) {
        std::cout << "Starting Adaptive Graph Engine...\n" << std::endl;
        
        bool run_benchmark = false;
        bool run_scaling = false;
        std::string filepath = "";

        if (argc > 1) {
            filepath = argv[1];
            if (argc > 2) {
                std::string arg2 = argv[2];
                if (arg2 == "--benchmark") run_benchmark = true;
                if (arg2 == "--scaling") run_scaling = true;
            }
            
            std::cout << "Loading graph from: " << filepath << std::endl;
            try {
                graph_engine::core::CSRGraph graph = graph_engine::core::load_graph(filepath);
                
                if (run_benchmark) {
                    std::cout << "\n======================================================\n";
                    std::cout << "           PAGERANK BENCHMARK SUITE STARTED           \n";
                    std::cout << "======================================================\n";
                    std::cout << "Graph: " << graph.num_vertices << " Vertices, " << graph.num_edges << " Edges\n";
                    std::cout << "Compute-Bound Workload: 20 Iterations of PageRank\n\n";

                    // 1. Sequential
                    std::cout << "[Running] Sequential PageRank...\n";
                    auto start_seq = std::chrono::high_resolution_clock::now();
                    std::vector<float> pr_seq = graph_engine::algorithms::pagerank_sequential(graph, 20);
                    auto end_seq = std::chrono::high_resolution_clock::now();
                    double time_seq = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

                    // 2. OpenMP
                    std::cout << "[Running] OpenMP PageRank (24 Threads)...\n";
                    auto start_omp = std::chrono::high_resolution_clock::now();
                    std::vector<float> pr_omp = graph_engine::algorithms::pagerank_openmp(graph, 20);
                    auto end_omp = std::chrono::high_resolution_clock::now();
                    double time_omp = std::chrono::duration<double, std::milli>(end_omp - start_omp).count();

                    // 3. CUDA
                    std::cout << "[Running] CUDA PageRank (20 Iterations entirely in VRAM)...\n";
                    
                    // WARM START: Prefetch UVM Memory to the GPU before measuring performance
                    int deviceId = 0;
                    graph_engine::algorithms::prefetch_graph_to_gpu(graph, deviceId);

                    auto start_cuda = std::chrono::high_resolution_clock::now();
                    std::vector<float> pr_cuda = graph_engine::algorithms::pagerank_cuda(graph, 20);
                    auto end_cuda = std::chrono::high_resolution_clock::now();
                    double time_cuda = std::chrono::duration<double, std::milli>(end_cuda - start_cuda).count();

                    std::cout << "\n================ PERFORMANCE REPORT ================\n";
                    std::cout << std::left << std::setw(20) << "Hardware" << std::setw(20) << "Time (ms)" << "Speedup vs Seq\n";
                    std::cout << "----------------------------------------------------\n";
                    std::cout << std::left << std::setw(20) << "CPU Sequential" << std::setw(20) << time_seq << "1.0x\n";
                    std::cout << std::left << std::setw(20) << "CPU OpenMP"     << std::setw(20) << time_omp << (time_seq/time_omp) << "x\n";
                    std::cout << std::left << std::setw(20) << "GPU CUDA"       << std::setw(20) << time_cuda << (time_seq/time_cuda) << "x\n";
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
    MPI_Finalize();
    return 0;
}
