#pragma once

#include <array>
#include <cstddef>
#include <type_traits>  // integral_constant
#if defined(CEXA_HAS_CXX20)
#include <ranges>
#endif

#include "tuple_fwd.hpp"
#include "traits.hpp"

namespace cexa {

// tuple_element
template <std::size_t I, class T>
struct tuple_element;

namespace impl {
template<std::size_t I, class T, class = tuple_element<I, std::remove_cv_t<T>>>
using has_tuple_element = T;
}

template <std::size_t I, class T>
struct tuple_element<I, const impl::has_tuple_element<I, T>> {
  using type = std::conditional_t<
      impl::is_subrange_v<T>,
      typename tuple_element<I, T>::type,
      std::add_const_t<typename tuple_element<I, T>::type>>;
};

template <std::size_t I, class T>
struct tuple_element<I, volatile T> {
  using type =
      std::add_volatile_t<typename tuple_element<I, T>::type>;
};

template <std::size_t I, class T>
struct tuple_element<I, const volatile T> {
  using type = 
     std::add_cv_t<typename tuple_element<I, T>::type>;
};

template <class T, class... Types>
struct tuple_element<0, tuple<T, Types...>> {
  using type = T;
};

template <std::size_t I, class T, class... Types>
struct tuple_element<I, tuple<T, Types...>>
    : tuple_element<I - 1, tuple<Types...>> {};

template <class T, class U>
struct tuple_element<0, std::pair<T, U>> {
  using type = T;
};

template <class T, class U>
struct tuple_element<1, std::pair<T, U>> {
  using type = U;
};

template <std::size_t I, class T, std::size_t N>
struct tuple_element<I, std::array<T, N>> {
  using type = T;
};

#if defined(CEXA_HAS_CXX20)
template <class I, class S, std::ranges::subrange_kind K>
struct tuple_element<0, std::ranges::subrange<I, S, K>> {
  using type = I;
};

template <class I, class S, std::ranges::subrange_kind K>
struct tuple_element<1, std::ranges::subrange<I, S, K>> {
  using type = S;
};
#endif

template <std::size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

// tuple_size
template <class T>
struct tuple_size;

namespace impl {
template <class T, std::size_t = tuple_size<std::remove_cv_t<T>>::value>
using has_tuple_size = T;
}

template <class T>
struct tuple_size<const impl::has_tuple_size<T>> : tuple_size<T> {};

template <class T>
struct tuple_size<volatile impl::has_tuple_size<T>> : tuple_size<T> {};

template <class T>
struct tuple_size<const volatile impl::has_tuple_size<T>> : tuple_size<T> {};

template <class... Types>
struct tuple_size<tuple<Types...>>
    : std::integral_constant<std::size_t, sizeof...(Types)> {};

template <class T, class U>
struct tuple_size<std::pair<T, U>> : std::integral_constant<std::size_t, 2> {};

template <class T, std::size_t N>
struct tuple_size<std::array<T, N>> : std::integral_constant<std::size_t, N> {};

#if defined(CEXA_HAS_CXX20)
template <class I, class S, std::ranges::subrange_kind K>
struct tuple_size<std::ranges::subrange<I, S, K>>
    : std::integral_constant<std::size_t, 2> {};
#endif

template <class T>
inline constexpr std::size_t tuple_size_v = tuple_size<T>::value;
}  // namespace cexa

template <typename... Types>
struct std::tuple_size<cexa::tuple<Types...>>
    : std::integral_constant<std::size_t, sizeof...(Types)> {};

template <std::size_t I, typename... Types>
struct std::tuple_element<I, cexa::tuple<Types...>>
    : cexa::tuple_element<I, cexa::tuple<Types...>> {};
