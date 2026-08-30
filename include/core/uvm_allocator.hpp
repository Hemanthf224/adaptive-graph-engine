#pragma once

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <iostream>

namespace graph_engine {
namespace core {

// A custom C++ Allocator that allocates memory using cudaMallocManaged (UVM).
// This allows std::vector to automatically create Unified Memory arrays.
template <class T>
struct UVMAllocator {
    typedef T value_type;

    UVMAllocator() = default;
    template <class U> constexpr UVMAllocator(const UVMAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        T* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, n * sizeof(T));
        if (err != cudaSuccess) {
            std::cerr << "UVM Allocation failed: " << cudaGetErrorString(err) << std::endl;
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* p, std::size_t n) noexcept {
        (void)n; // unused
        cudaFree(p);
    }
};

template <class T, class U>
bool operator==(const UVMAllocator<T>&, const UVMAllocator<U>&) { return true; }
template <class T, class U>
bool operator!=(const UVMAllocator<T>&, const UVMAllocator<U>&) { return false; }

} // namespace core
} // namespace graph_engine
