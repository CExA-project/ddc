#pragma once

#include <type_traits>
#include <utility>

#include "macros.hpp"
#include "traits.hpp"
#include "tuple.hpp"

namespace cexa {

namespace impl {
template <class>
constexpr bool is_reference_wrapper_v = false;
template <class U>
constexpr bool is_reference_wrapper_v<std::reference_wrapper<U>> = true;

template <class C, class Pointed, class Object, class... Args>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) invoke_ptr(Pointed C::* member,
                                                           Object&& object,
                                                           Args&&... args) {
  using object_t            = remove_cvref_t<Object>;
  constexpr bool is_wrapped = is_reference_wrapper_v<object_t>;
  constexpr bool is_derived_object =
      std::is_same_v<C, object_t> || std::is_base_of_v<C, object_t>;

  if constexpr (std::is_function_v<Pointed>) {
    if constexpr (is_derived_object)
      return (std::forward<Object>(object).*
              member)(std::forward<Args>(args)...);
    else if constexpr (is_wrapped)
      return (object.get().*member)(std::forward<Args>(args)...);
    else
      return ((*std::forward<Object>(object)).*
              member)(std::forward<Args>(args)...);
  } else {
    static_assert(std::is_object_v<Pointed> && sizeof...(args) == 0);
    if constexpr (is_derived_object)
      return std::forward<Object>(object).*member;
    else if constexpr (is_wrapped)
      return object.get().*member;
    else
      return (*std::forward<Object>(object)).*member;
  }
}

template <class F, class... Args>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) invoke(F&& f, Args&&... args) {
  if constexpr (std::is_member_pointer_v<remove_cvref_t<F>>) {
    return invoke_ptr(f, std::forward<Args>(args)...);
  } else {
    return (std::forward<F>(f))(std::forward<Args>(args)...);
  }
}

template <class F, class Tuple, std::size_t... I>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply(
    F&& f, Tuple&& t, std::index_sequence<I...>) {
  return invoke(std::forward<F>(f), cexa::get<I>(std::forward<Tuple>(t))...);
}

template <class T, class Tuple, std::size_t... I>
KOKKOS_INLINE_FUNCTION constexpr T make_from_tuple(Tuple&& t,
                                                   std::index_sequence<I...>) {
  return T(cexa::get<I>(std::forward<Tuple>(t))...);
}

template <class U, class T, std::size_t = tuple_size_v<impl::remove_cvref_t<T>>>
struct make_tuple_constraint : std::true_type {};

template <class U, class Tuple>
struct make_tuple_constraint<U, Tuple, 1> {
  static constexpr bool value = !impl::reference_constructs_from_temporary_v<
      U, decltype(get<0>(std::declval<Tuple>()))>;
};

template <class T, class Tuple, class seq>
struct is_constructible_from_tuple;

template <class T, class Tuple, std::size_t... Ints>
struct is_constructible_from_tuple<T, Tuple, std::index_sequence<Ints...>> {
  static constexpr bool value =
      std::is_constructible_v<T, decltype(get<Ints>(std::declval<Tuple>()))...>;
};

template <class T, class Tuple>
inline constexpr bool is_constructible_from_tuple_v =
    is_constructible_from_tuple<T, Tuple,
                                std::make_index_sequence<tuple_size_v<
                                    impl::remove_cvref_t<Tuple>>>>::value;

}  // namespace impl

template <class F, class Tuple>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  static_assert(impl::is_tuple_v<impl::remove_cvref_t<Tuple>>,
                "cexa::apply can only be called with cexa::tuple");
  return impl::apply(
      std::forward<F>(f), std::forward<Tuple>(t),
      std::make_index_sequence<tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

template <
    class T, class Tuple,
    class = std::enable_if_t<impl::is_constructible_from_tuple_v<T, Tuple>>>
KOKKOS_INLINE_FUNCTION constexpr T make_from_tuple(Tuple&& t) {
  static_assert(impl::is_tuple_v<impl::remove_cvref_t<Tuple>>,
                "cexa::make_from_tuple can only be called with cexa::tuple");
  constexpr std::size_t size = tuple_size_v<std::remove_reference_t<Tuple>>;
  static_assert(
      impl::make_tuple_constraint<T, impl::remove_cvref_t<Tuple>>::value);
  return impl::make_from_tuple<T>(std::forward<Tuple>(t),
                                  std::make_index_sequence<size>{});
}
}  // namespace cexa
