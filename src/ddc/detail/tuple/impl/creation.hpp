#pragma once

#include <type_traits>
#include <utility>

#include "tuple.hpp"
#include "traits.hpp"
#include "helper.hpp"

namespace cexa {

// tuple.creation
template <class... TTypes>
KOKKOS_INLINE_FUNCTION constexpr tuple<impl::unwrap_ref_decay_t<TTypes>...>
make_tuple(TTypes&&... t) {
  return {std::forward<TTypes>(t)...};
}

template <class... TTypes>
KOKKOS_INLINE_FUNCTION constexpr tuple<TTypes&&...> forward_as_tuple(
    TTypes&&... t) noexcept {
  return tuple<TTypes&&...>(std::forward<TTypes>(t)...);
}

template <class... TTypes>
KOKKOS_INLINE_FUNCTION constexpr tuple<TTypes&...> tie(TTypes&... t) noexcept {
  return tuple<TTypes&...>{t...};
}

// tuple_cat helper
namespace impl {
template <std::size_t A, class B, class C, class... D>
struct cartesian_product_impl;

template <std::size_t I, std::size_t... Res1, std::size_t... Res2>
struct cartesian_product_impl<I, std::index_sequence<Res1...>,
                              std::index_sequence<Res2...>> {
  using seq1 = std::index_sequence<Res1...>;
  using seq2 = std::index_sequence<Res2...>;
};

template <std::size_t I, std::size_t... Res1, std::size_t... Res2,
          class... RemainingSeq>
struct cartesian_product_impl<I, std::index_sequence<Res1...>,
                              std::index_sequence<Res2...>,
                              std::index_sequence<>, RemainingSeq...>
    : cartesian_product_impl<I + 1, std::index_sequence<Res1...>,
                             std::index_sequence<Res2...>, RemainingSeq...> {};

template <std::size_t I, std::size_t... Res1, std::size_t... Res2,
          std::size_t Head, std::size_t... Tail, class... RemainingSeq>
struct cartesian_product_impl<
    I, std::index_sequence<Res1...>, std::index_sequence<Res2...>,
    std::index_sequence<Head, Tail...>, RemainingSeq...>
    : cartesian_product_impl<
          I + 1, std::index_sequence<Res1..., Head, Tail...>,
          std::index_sequence<Res2..., I, ((void)Tail, I)...>,
          RemainingSeq...> {};

template <class... Tuples>
struct cartesian_product
    : cartesian_product_impl<0, std::index_sequence<>, std::index_sequence<>,
                             std::make_index_sequence<tuple_size<
                                 std::remove_reference_t<Tuples>>::value>...> {
};

template <class... Tuples, std::size_t... Ints1, std::size_t... Ints2>
KOKKOS_FORCEINLINE_FUNCTION constexpr tuple<cexa::tuple_element_t<
    Ints1,
    impl::remove_cvref_t<cexa::tuple_element_t<Ints2, tuple<Tuples...>>>>...>
tuple_cat_impl(tuple<Tuples...>&& tuples, std::index_sequence<Ints1...>,
               std::index_sequence<Ints2...>) {
  return {get<Ints1>(std::move(get<Ints2>(tuples)))...};
}
}  // namespace impl

template <class... Tuples,
          class = std::enable_if_t<(impl::is_tuple_like<Tuples>::value && ...)>>
KOKKOS_INLINE_FUNCTION constexpr auto tuple_cat(Tuples&&... tuples) {
  using cartesian_product_t = impl::cartesian_product<Tuples...>;
  return impl::tuple_cat_impl(cexa::forward_as_tuple(tuples...),
                              typename cartesian_product_t::seq1{},
                              typename cartesian_product_t::seq2{});
}
}  // namespace cexa
