// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <utility>

#include "for_each.hpp"

namespace experimental {

namespace mdranges {

template <std::size_t I, class CartRange>
using iterator_t = decltype(std::declval<CartRange&>().template begin<I>());

template <class CartRange>
using range_value_t = typename CartRange::value_type;

template <class CartRange>
inline constexpr std::size_t range_rank_v = CartRange::rank();

} // namespace mdranges

namespace detail {

template <class ValueType, class F, class TupleOfIterators, std::size_t... Is>
class ForEachKokkosLambdaAdapter
{
    template <std::size_t I>
    using index_type = std::size_t;

    F m_f;

    TupleOfIterators m_its_b;

public:
    ForEachKokkosLambdaAdapter(TupleOfIterators const& its_b, F const& f) : m_f(f), m_its_b(its_b)
    {
    }

    DDC_FORCEINLINE_FUNCTION void operator()(index_type<Is>... ids) const
    {
        m_f(ValueType(*(std::get<Is>(m_its_b) + ids)...));
    }
};

// Should work with a cartesian product of random access iterators
template <class ExecSpace, class CartRange, class Functor, std::size_t... Is>
inline void for_each_kokkos(
        CartRange const& mdrange,
        Functor const& f,
        std::index_sequence<Is...>) noexcept
{
    Kokkos::Array<std::size_t, mdranges::range_rank_v<CartRange>> const begin {};
    Kokkos::Array<std::size_t, mdranges::range_rank_v<CartRange>> const end {
            mdrange.template extent<Is>()...};
    Kokkos::parallel_for(
            Kokkos::MDRangePolicy<
                    ExecSpace,
                    Kokkos::Rank<
                            mdranges::range_rank_v<CartRange>,
                            Kokkos::Iterate::Right,
                            Kokkos::Iterate::Right>>(begin, end),
            ForEachKokkosLambdaAdapter<
                    mdranges::range_value_t<CartRange>,
                    Functor,
                    std::tuple<mdranges::iterator_t<Is, CartRange>...>,
                    Is...>(std::make_tuple(mdrange.template begin<Is>()...), f));
}

// Should work with a cartesian product of forward iterators
template <class CartRange, class Functor, class... Its>
inline void for_each_serial(CartRange const& mdrange, Functor const& f, Its const&... its) noexcept
{
    static constexpr std::size_t I = sizeof...(Its);
    if constexpr (I == mdranges::range_rank_v<CartRange>) {
        f(mdranges::range_value_t<CartRange>(*its...));
    } else {
        mdranges::iterator_t<I, CartRange> const begin = mdrange.template begin<I>();
        mdranges::iterator_t<I, CartRange> const end = mdrange.template end<I>();
        for (mdranges::iterator_t<I, CartRange> it = begin; it != end; ++it) {
            for_each_serial(mdrange, f, its..., it);
        }
    }
}

} // namespace detail

/** iterates over a nD domain using the serial execution policy
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class CartRange, class Functor>
inline void for_each(parallel_host_policy, CartRange const& mdrange, Functor&& f) noexcept
{
    detail::for_each_kokkos<Kokkos::DefaultHostExecutionSpace>(
            mdrange,
            std::forward<Functor>(f),
            std::make_index_sequence<mdranges::range_rank_v<CartRange>>());
}

/** iterates over a nD domain using the parallel_device_policy execution policy
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class CartRange, class Functor>
inline void for_each(parallel_device_policy, CartRange const& mdrange, Functor&& f) noexcept
{
    detail::for_each_kokkos<Kokkos::DefaultExecutionSpace>(
            mdrange,
            std::forward<Functor>(f),
            std::make_index_sequence<mdranges::range_rank_v<CartRange>>());
}

/** iterates over a nD domain using the serial execution policy
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class CartRange, class Functor>
inline void for_each(serial_host_policy, CartRange const& mdrange, Functor&& f) noexcept
{
    detail::for_each_serial(mdrange, std::forward<Functor>(f));
}

} // namespace experimental
