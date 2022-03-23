#pragma once

#include <limits>
#include <utility>

namespace reducer
{

template <class T>
struct sum
{
    using result_type = T;

    static constexpr void initialize(T& init) noexcept
    {
        init = 0;
    }

    static constexpr void reduce(T& out, T const& in) noexcept
    {
        out = out + in;
    }
};

template <class T>
struct prod
{
    using result_type = T;

    static constexpr void initialize(T& init) noexcept
    {
        init = 1;
    }

    static constexpr void reduce(T& out, T const& in) noexcept
    {
        out = out * in;
    }
};

struct land
{
    using result_type = bool;

    static constexpr void initialize(bool& init) noexcept
    {
        init = true;
    }

    static constexpr void reduce(bool& out, bool const& in) noexcept
    {
        out = out && in;
    }
};

struct lor
{
    using result_type = bool;

    static constexpr void initialize(bool& init) noexcept
    {
        init = false;
    }

    static constexpr void reduce(bool& out, bool const& in) noexcept
    {
        out = out || in;
    }
};

template <class T>
struct band
{
    using result_type = T;

    static constexpr void initialize(T& init) noexcept
    {
        init = ~T(0);
    }

    static constexpr void reduce(T& out, T const& in) noexcept
    {
        out = out & in;
    }
};

template <class T>
struct bor
{
    using result_type = T;

    static constexpr void initialize(T& init) noexcept
    {
        init = 0;
    }

    static constexpr void reduce(T& out, T const& in) noexcept
    {
        out = out | in;
    }
};

template <class T>
struct bxor
{
    using result_type = T;

    static constexpr void initialize(T& init) noexcept
    {
        init = 0;
    }

    static constexpr void reduce(T& out, T const& in) noexcept
    {
        out = out ^ in;
    }
};

template <class T>
struct min
{
    using result_type = T;

    static constexpr void initialize(T& init) noexcept
    {
        init = std::numeric_limits<T>::max();
    }

    static constexpr void reduce(T& out, T const& in) noexcept
    {
        out = out < in ? out : in;
    }
};

template <class T>
struct max
{
    using result_type = T;

    static constexpr void initialize(T& init) noexcept
    {
        init = std::numeric_limits<T>::lowest();
    }

    static constexpr void reduce(T& out, T const& in) noexcept
    {
        out = out > in ? out : in;
    }
};

template <class T>
struct minmax
{
    using result_type = std::pair<T, T>;

    static constexpr void initialize(std::pair<T, T>& init) noexcept
    {
        min<T>::initialize(init.first);
        max<T>::initialize(init.second);
    }

    static constexpr void reduce(std::pair<T, T>& out, std::pair<T, T> const& in) noexcept
    {
        min<T>::reduce(out.first, in.first);
        max<T>::reduce(out.second, in.second);
    }
};

} // namespace reducer
