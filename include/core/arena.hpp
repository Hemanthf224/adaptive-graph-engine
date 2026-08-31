#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <mutex>

namespace graph_engine {
namespace core {

/**
 * @class LinearArenaAllocator
 * @brief High-performance custom memory allocator for C++.
 * 
 * Bypasses the OS-level `malloc`/`new` fragmentation overhead by pre-allocating
 * a massive contiguous block of RAM upfront. Memory is allocated to the application
 * instantly via simple pointer arithmetic (bumping).
 * 
 * Essential for HPC Graph Engines where millions of small temporal vertex structures
 * are created and destroyed rapidly.
 */
class LinearArenaAllocator {
private:
    uint8_t* m_Buffer;
    size_t m_Capacity;
    size_t m_Offset;
    std::mutex m_Mutex;

public:
    /**
     * @brief Pre-allocates the Arena memory pool.
     * @param capacity_bytes Total size of the arena (e.g. 1024 * 1024 * 1024 for 1GB)
     */
    LinearArenaAllocator(size_t capacity_bytes) : m_Capacity(capacity_bytes), m_Offset(0) {
        m_Buffer = static_cast<uint8_t*>(std::malloc(m_Capacity));
        if (!m_Buffer) {
            throw std::bad_alloc();
        }
        std::cout << "[ARENA] Pre-allocated " << (m_Capacity / (1024 * 1024)) << " MB memory pool.\n";
    }

    ~LinearArenaAllocator() {
        if (m_Buffer) {
            std::free(m_Buffer);
        }
    }

    /**
     * @brief Allocates memory from the Arena. O(1) complexity.
     * @param size Size in bytes to allocate.
     * @param alignment Memory alignment boundary (default 8 or 16).
     * @return Pointer to the allocated memory.
     */
    void* Allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        std::lock_guard<std::mutex> lock(m_Mutex);

        // Calculate alignment adjustment
        size_t adjustment = alignment - (reinterpret_cast<uintptr_t>(m_Buffer + m_Offset) % alignment);
        if (adjustment == alignment) adjustment = 0;

        if (m_Offset + adjustment + size > m_Capacity) {
            throw std::runtime_error("[ARENA] Out of Memory! Arena capacity exceeded.");
        }

        m_Offset += adjustment;
        void* ptr = m_Buffer + m_Offset;
        m_Offset += size;

        return ptr;
    }

    /**
     * @brief Instantly frees all memory in the arena. O(1) complexity.
     * Does NOT call destructors.
     */
    void Reset() {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_Offset = 0;
    }

    size_t GetCapacity() const { return m_Capacity; }
    size_t GetUsed() const { return m_Offset; }
};

} // namespace core
} // namespace graph_engine
