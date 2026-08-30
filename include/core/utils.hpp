#pragma once

#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>

namespace graph_engine {
namespace core {

inline void print_environment_info(int mpi_rank, int mpi_size) {
    if (mpi_rank == 0) {
        std::cout << "=== Adaptive Graph Engine Environment ===" << std::endl;
        
        // OpenMP info
        int num_threads = 0;
        #pragma omp parallel
        {
            #pragma omp single
            num_threads = omp_get_num_threads();
        }
        std::cout << "OpenMP Threads : " << num_threads << std::endl;
        
        // MPI info
        std::cout << "MPI Ranks      : " << mpi_size << std::endl;
        
        // CUDA info
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err == cudaSuccess && deviceCount > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "CUDA Device(s) : " << deviceCount << std::endl;
            std::cout << "Primary GPU    : " << prop.name << " (Compute " 
                      << prop.major << "." << prop.minor << ")" << std::endl;
            std::cout << "VRAM           : " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        } else {
            std::cout << "CUDA           : Not Available or No Devices" << std::endl;
        }
        
        std::cout << "=========================================" << std::endl;
    }
}

} // namespace core
} // namespace graph_engine
