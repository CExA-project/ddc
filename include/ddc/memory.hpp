#pragma once

#include <memory>
#include <type_traits>

template <class T>
struct HostAllocator;

struct MemorySpace
{
};

template <class T>
constexpr inline bool is_memory_space = std::is_base_of_v<MemorySpace, T>;

struct HostMemorySpace : MemorySpace
{
    template <class T>
    using default_allocator_type = HostAllocator<T>;
};

template <class Memory, class T>
using DefaultAllocator_t = typename Memory::template default_allocator_type<T>;

template <class T>
class HostAllocator
{
public:
    using value_type = T;
    using memory_space_type = HostMemorySpace;

    constexpr HostAllocator() = default;

    constexpr HostAllocator(HostAllocator const& x) = default;

    constexpr HostAllocator(HostAllocator&& x) noexcept = default;

    ~HostAllocator() = default;

    constexpr HostAllocator& operator=(HostAllocator const& x) = default;

    constexpr HostAllocator& operator=(HostAllocator&& x) noexcept = default;

    [[nodiscard]] T* allocate(std::size_t n)
    {
        return new (std::align_val_t(64)) value_type[n];
    }

    void deallocate(T* p, std::size_t)
    {
        operator delete[](p, std::align_val_t(64));
    }
};
