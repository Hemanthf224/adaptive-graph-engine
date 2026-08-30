#!/bin/bash
# ==============================================================================
# WSL2 Ubuntu Development Environment Setup
# Project: Adaptive GPU-Accelerated Graph Processing Engine
# ==============================================================================
# 
# This script installs all required development tools for the project.
# Run this inside WSL2 Ubuntu with: bash wsl_setup.sh
#
# It will ask for your sudo password once at the beginning.
# ==============================================================================

set -e  # Exit on any error

echo "=============================================="
echo " Adaptive Graph Engine — WSL2 Setup Script"
echo "=============================================="
echo ""

# ------------------------------------------
# STEP 1: Update package lists
# ------------------------------------------
echo "[1/6] Updating package lists..."
sudo apt update -y
echo "✓ Package lists updated."
echo ""

# ------------------------------------------
# STEP 2: Install essential build tools
# ------------------------------------------
echo "[2/6] Installing build-essential (gcc, g++, make)..."
sudo apt install -y build-essential
echo "✓ build-essential installed."
echo ""

# ------------------------------------------
# STEP 3: Install CMake
# ------------------------------------------
echo "[3/6] Installing CMake..."
sudo apt install -y cmake
echo "✓ CMake installed."
echo ""

# ------------------------------------------
# STEP 4: Install OpenMPI
# ------------------------------------------
echo "[4/6] Installing OpenMPI..."
sudo apt install -y openmpi-bin libopenmpi-dev
echo "✓ OpenMPI installed."
echo ""

# ------------------------------------------
# STEP 5: Install additional useful tools
# ------------------------------------------
echo "[5/6] Installing additional tools (pkg-config, wget, curl)..."
sudo apt install -y pkg-config wget curl
echo "✓ Additional tools installed."
echo ""

# ------------------------------------------
# STEP 6: Verify all installations
# ------------------------------------------
echo "[6/6] Verifying installations..."
echo ""
echo "=== VERIFICATION REPORT ==="
echo ""

echo "--- C++ Compiler ---"
g++ --version | head -1

echo ""
echo "--- GCC ---"
gcc --version | head -1

echo ""
echo "--- Make ---"
make --version | head -1

echo ""
echo "--- CMake ---"
cmake --version

echo ""
echo "--- OpenMPI ---"
mpirun --version | head -1
echo "mpicc: $(which mpicc)"

echo ""
echo "--- OpenMP test ---"
echo '#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    {
        #pragma omp single
        printf("OpenMP threads: %d\n", omp_get_num_threads());
    }
    return 0;
}' > /tmp/omp_test.c
gcc -fopenmp /tmp/omp_test.c -o /tmp/omp_test && /tmp/omp_test
rm -f /tmp/omp_test /tmp/omp_test.c
echo "✓ OpenMP works."

echo ""
echo "--- System ---"
echo "OS: $(lsb_release -ds)"
echo "Kernel: $(uname -r)"
echo "Arch: $(uname -m)"

echo ""
echo "=== GPU Access (from WSL2) ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo "✓ GPU accessible from WSL2."
else
    echo "⚠ nvidia-smi not found in WSL2."
    echo "  This is expected — we will install the CUDA toolkit separately."
    echo "  The GPU is still accessible via the Windows driver paravirtualization."
fi

echo ""
echo "=============================================="
echo " Base setup complete!"
echo ""
echo " NEXT STEP: Install CUDA Toolkit for WSL2."
echo " This requires adding NVIDIA's package repository."
echo " Run the CUDA setup commands separately (see instructions)."
echo "=============================================="
