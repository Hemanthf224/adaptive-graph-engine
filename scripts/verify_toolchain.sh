#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "=== TOOLCHAIN SUMMARY ==="
echo "g++:     $(g++ --version | head -1)"
echo "cmake:   $(cmake --version | head -1)"
echo "nvcc:    $(nvcc --version | tail -1)"
echo "mpirun:  $(ompi_info --version | head -1)"
GPU_OUT=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "GPU:     $GPU_OUT"
else
    echo "GPU:     not available"
fi
echo '#include <omp.h>' > /tmp/omp_test_check.cpp
g++ -fopenmp -c /tmp/omp_test_check.cpp -o /tmp/omp_test_check.o 2>/dev/null
if [ $? -eq 0 ]; then
    echo "OpenMP:  OK"
else
    echo "OpenMP:  MISSING"
fi
echo "========================="
