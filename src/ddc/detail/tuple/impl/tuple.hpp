#pragma once

#include <Kokkos_Macros.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "macros.hpp"
#include "traits.hpp"
#include "helper.hpp"
#include "tuple_fwd.hpp"

#if defined(CEXA_HAS_CXX20)
#include <compare>
#endif

namespace cexa {

// Pre-declarations of get, as C++17 doesn't allow the function to be used
// before its declaration.
template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)), typename tuple_element<I, tuple<Types...>>::type&>
get(tuple<Types...>& t) noexcept;

template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)), typename tuple_element<I, tuple<Types...>>::type&&>
get(tuple<Types...>&& t) noexcept;

template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)),
    const typename tuple_element<I, tuple<Types...>>::type&>
get(const tuple<Types...>& t) noexcept;

template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)),
    const typename tuple_element<I, tuple<Types...>>::type&&>
get(const tuple<Types...>&& t) noexcept;

template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (std::is_same_v<T, Types> || ...), T&>
get(tuple<Types...>& t) noexcept;

template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (std::is_same_v<T, Types> || ...), T&&>
get(tuple<Types...>&& t) noexcept;

template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (std::is_same_v<T, Types> || ...), const T&>
get(const tuple<Types...>& t) noexcept;

template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr const std::enable_if_t<
    (std::is_same_v<T, Types> || ...), const T&&>
get(const tuple<Types...>&& t) noexcept;

namespace impl {

template <std::size_t I, typename... Ts>
struct nth_type;
template <typename T, typename... Ts>
struct nth_type<0, T, Ts...> {
  using type = T;
};
template <std::size_t I, typename T, typename... Ts>
struct nth_type<I, T, Ts...> : nth_type<I - 1, Ts...> {};

#define FWD(x) std::forward<decltype(x)>(x)
// #define FWD(x) static_cast<decltype(x)>(x)

template <class Tuple, class UTuple>
struct all_types_constructible : std::false_type {};

template <class... Types, class UTuple>
struct all_types_constructible<tuple<Types...>, UTuple> {
  using t = void;
  template <class Seq>
  struct all_types_constructible_helper;
  template <std::size_t... Ints>
  struct all_types_constructible_helper<std::index_sequence<Ints...>> {
    static inline constexpr bool value =
        (std::is_constructible_v<Types,
                                 decltype(get<Ints>(std::declval<UTuple>()))> &&
         ...);
  };

  static inline constexpr bool value = all_types_constructible_helper<
      decltype(std::index_sequence_for<Types...>{})>::value;
};

template <class Tuple, class UTuple>
inline constexpr bool all_types_constructible_v =
    all_types_constructible<Tuple, UTuple>::value;

template <class UTuple, class Tuple>
struct all_types_convertible : std::false_type {};

template <class... Types, class UTuple>
struct all_types_convertible<UTuple, tuple<Types...>> {
  template <class Seq>
  struct all_types_convertible_helper;
  template <std::size_t... Ints>
  struct all_types_convertible_helper<std::index_sequence<Ints...>> {
    static inline constexpr bool value =
        (std::is_convertible_v<decltype(get<Ints>(FWD(std::declval<UTuple>()))),
                               Types> &&
         ...);
  };

  static inline constexpr bool value = all_types_convertible_helper<
      decltype(std::index_sequence_for<Types...>{})>::value;
};

template <class UTuple, class Tuple>
inline constexpr bool all_types_convertible_v =
    all_types_convertible<UTuple, Tuple>::value;

template <class T>
struct is_pair : std::false_type {};

template <class T, class U>
struct is_pair<std::pair<T, U>> : std::true_type {};

template <class Tuple, class UTuple>
struct any_types_reference_constructs_from_temporary : std::false_type {};

template <class... Types, class UTuple>
struct any_types_reference_constructs_from_temporary<tuple<Types...>, UTuple> {
  template <class Seq>
  struct any_types_reference_constructs_from_temporary_helper;
  template <std::size_t... Ints>
  struct any_types_reference_constructs_from_temporary_helper<
      std::index_sequence<Ints...>> {
    static inline constexpr bool value =
        (impl::reference_constructs_from_temporary_v<
             Types, decltype(get<Ints>(FWD(std::declval<UTuple>())))> ||
         ...);
  };

  static inline constexpr bool value =
      any_types_reference_constructs_from_temporary_helper<
          decltype(std::index_sequence_for<Types...>{})>::value;
};

template <class Tuple, class UTuple>
inline constexpr bool any_types_reference_constructs_from_temporary_v =
    any_types_reference_constructs_from_temporary<Tuple, UTuple>::value;

template <bool, bool, bool, bool>
struct Bools {};

template <class B, typename... Types>
struct store;

template <class T>
struct is_store : std::false_type {};

template <bool a, bool b, bool c, bool d, class... Types>
struct is_store<store<Bools<a, b, c, d>, Types...>> : std::true_type {};

template <class T>
inline constexpr bool is_store_v = is_store<impl::remove_cvref_t<T>>::value;

template <class... Types>
struct store_alias;

template <>
struct store_alias<> {
  using type = store<Bools<false, false, false, false>>;
};
template <class T, class... Types>
struct store_alias<T, Types...> {
  using type = store<
      Bools<std::is_copy_assignable_v<T>, std::is_copy_assignable_v<const T>,
            std::is_move_assignable_v<T>, std::is_assignable_v<const T&, T&&>>,
      T, Types...>;
};

template <class... Types>
using store_ = typename store_alias<Types...>::type;

template <>
struct store<Bools<false, false, false, false>> {
  KOKKOS_DEFAULTED_FUNCTION constexpr store()             = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr store(const store&) = default;
  KOKKOS_INLINE_FUNCTION constexpr store& operator=(const store&) {
    return *this;
  };
  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(const store&) const {
    return *this;
  };

  KOKKOS_INLINE_FUNCTION constexpr void swap(store&) noexcept {}
  KOKKOS_INLINE_FUNCTION constexpr void swap(const store&) const noexcept {}
#if defined(CEXA_HAS_CXX20)
  KOKKOS_DEFAULTED_FUNCTION auto operator<=>(const store&) const = default;
#endif
};

#if !defined(CEXA_HAS_CXX20)
KOKKOS_INLINE_FUNCTION constexpr bool operator==(const store_<>&,
                                                 const store_<>&) {
  return true;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator<(const store_<>&,
                                                const store_<>&) {
  return false;
}
#endif

#if defined(CEXA_HAS_CXX20)
#define CEXA_CONSTEXPR_CXX20 constexpr

// TODO: check if the non-three way comparable case should always return
// weak_ordering
#define STORE_SPACESHIP_COMP                                                  \
  template <bool b1, bool b2, bool b3, bool b4, class U, class... UTypes>     \
    requires std::three_way_comparable_with<T, U>                             \
  KOKKOS_INLINE_FUNCTION constexpr auto operator<=>(                          \
      const store<Bools<b1, b2, b3, b4>, U, UTypes...>& rhs)                  \
      const->std::common_comparison_category_t<decltype(value <=> rhs.value), \
                                               decltype(rest <=> rhs.rest)> { \
    auto res = value <=> rhs.value;                                           \
    return res != 0 ? res : rest <=> rhs.rest;                                \
  }                                                                           \
  template <bool b1, bool b2, bool b3, bool b4, class U, class... UTypes>     \
  KOKKOS_INLINE_FUNCTION constexpr std::weak_ordering operator<=>(            \
      const store<Bools<b1, b2, b3, b4>, U, UTypes...>& rhs) const {          \
    if (value < rhs.value) {                                                  \
      return std::weak_ordering::less;                                        \
    } else if (rhs.value < value) {                                           \
      return std::weak_ordering::greater;                                     \
    } else {                                                                  \
      return static_cast<std::weak_ordering>(rest <=> rhs.rest);              \
    }                                                                         \
  }                                                                           \
  template <bool b1, bool b2, bool b3, bool b4, class U, class... UTypes>     \
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(                           \
      const store<Bools<b1, b2, b3, b4>, U, UTypes...>& rhs) const {          \
    return operator<=>(rhs) == 0;                                             \
  }
#else
#define CEXA_CONSTEXPR_CXX20
#define STORE_SPACESHIP_COMP
#endif

#define STORE_COMMON_FUNCS                                                     \
  T value{};                                                                   \
  store_<Types...> rest;                                                       \
                                                                               \
  KOKKOS_DEFAULTED_FUNCTION constexpr store() = default;                       \
                                                                               \
  KOKKOS_DEFAULTED_FUNCTION constexpr store(const store& other) = default;     \
  template <class Dummy = void,                                                \
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&      \
                                           std::is_move_constructible_v<T>>>   \
  KOKKOS_INLINE_FUNCTION constexpr store(store&& other)                        \
      : value(FWD(other).value), rest(FWD(other).rest) {}                      \
                                                                               \
  template <typename U, typename... UTypes,                                    \
            class = std::enable_if_t<!is_store_v<U> &&                         \
                                     (!is_store_v<UTypes> && ...)>>            \
  KOKKOS_INLINE_FUNCTION constexpr store(U&& u, UTypes&&... args)              \
      : value(FWD(u)), rest(FWD(args)...) {}                                   \
                                                                               \
  KOKKOS_DEFAULTED_FUNCTION CEXA_CONSTEXPR_CXX20 ~store() = default;           \
                                                                               \
  template <bool b1, bool b2, bool b3, bool b4, class U, class... UTypes,      \
            class =                                                            \
                std::enable_if_t<sizeof...(UTypes) == sizeof...(Types) &&      \
                                 !(std::is_same_v<U, T> &&                     \
                                   (std::is_same_v<UTypes, Types> && ...)) &&  \
                                 std::is_assignable_v<T&, const U&>>>          \
  KOKKOS_INLINE_FUNCTION constexpr store& operator=(                           \
      const store<Bools<b1, b2, b3, b4>, U, UTypes...>& other) {               \
    value = other.value;                                                       \
    rest  = other.rest;                                                        \
    return *this;                                                              \
  }                                                                            \
  template <bool b1, bool b2, bool b3, bool b4, class U, class... UTypes,      \
            class =                                                            \
                std::enable_if_t<sizeof...(UTypes) == sizeof...(Types) &&      \
                                 !(std::is_same_v<U, T> &&                     \
                                   (std::is_same_v<UTypes, Types> && ...)) &&  \
                                 std::is_assignable_v<const T&, const U&>>>    \
  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(                     \
      const store<Bools<b1, b2, b3, b4>, U, UTypes...>& other) const {         \
    value = other.value;                                                       \
    rest  = other.rest;                                                        \
    return *this;                                                              \
  }                                                                            \
  template <bool b1, bool b2, bool b3, bool b4, class U, class... UTypes,      \
            class =                                                            \
                std::enable_if_t<sizeof...(UTypes) == sizeof...(Types) &&      \
                                 !(std::is_same_v<U, T> &&                     \
                                   (std::is_same_v<UTypes, Types> && ...)) &&  \
                                 std::is_assignable_v<T&, U&&>>>               \
  KOKKOS_INLINE_FUNCTION constexpr store& operator=(                           \
      store<Bools<b1, b2, b3, b4>, U, UTypes...>&& other) {                    \
    value = FWD(other.value);                                                  \
    rest  = FWD(other.rest);                                                   \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <bool b1, bool b2, bool b3, bool b4, class U, class... UTypes,      \
            class =                                                            \
                std::enable_if_t<sizeof...(UTypes) == sizeof...(Types) &&      \
                                 !(std::is_same_v<U, T> &&                     \
                                   (std::is_same_v<UTypes, Types> && ...)) &&  \
                                 std::is_assignable_v<const T&, U&&>>>         \
  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(                     \
      store<Bools<b1, b2, b3, b4>, U, UTypes...>&& other) const {              \
    value = FWD(other.value);                                                  \
    rest  = FWD(other.rest);                                                   \
    return *this;                                                              \
  }                                                                            \
  template <std::size_t I>                                                     \
  KOKKOS_INLINE_FUNCTION constexpr tuple_element_t<I, tuple<T, Types...>>&     \
  get_value() noexcept {                                                       \
    if constexpr (I == 0) {                                                    \
      return value;                                                            \
    } else {                                                                   \
      return rest.template get_value<I - 1>();                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <std::size_t I>                                                     \
  KOKKOS_INLINE_FUNCTION constexpr const tuple_element_t<I,                    \
                                                         tuple<T, Types...>>&  \
  get_value() const noexcept {                                                 \
    if constexpr (I == 0) {                                                    \
      return value;                                                            \
    } else {                                                                   \
      return rest.template get_value<I - 1>();                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <class Type>                                                        \
  KOKKOS_INLINE_FUNCTION constexpr Type& get_value() noexcept {                \
    if constexpr (std::is_same_v<Type, T>) {                                   \
      return value;                                                            \
    } else {                                                                   \
      return rest.template get_value<Type>();                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <class Type>                                                        \
  KOKKOS_INLINE_FUNCTION constexpr const Type& get_value() const noexcept {    \
    if constexpr (std::is_same_v<Type, T>) {                                   \
      return value;                                                            \
    } else {                                                                   \
      return rest.template get_value<Type>();                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <class U, class = std::enable_if_t<std::is_assignable_v<            \
                         T&, decltype(std::forward<U>(std::declval<U&&>()))>>> \
  KOKKOS_INLINE_FUNCTION constexpr void set_all(U&& u) {                       \
    value = u;                                                                 \
  }                                                                            \
                                                                               \
  template <class U, class... UTypes,                                          \
            class = std::enable_if_t<std::is_assignable_v<                     \
                T&, decltype(std::forward<U>(std::declval<U&&>()))>>>          \
  KOKKOS_INLINE_FUNCTION constexpr void set_all(U&& head, UTypes&&... tail) {  \
    value = head;                                                              \
    rest.set_all(FWD(tail)...);                                                \
  }                                                                            \
                                                                               \
  template <class UTuple>                                                      \
  inline constexpr void set(UTuple&& u) {                                      \
    set(FWD(u), std::make_index_sequence<1 + sizeof...(Types)>{});             \
  }                                                                            \
                                                                               \
  template <class UTuple, std::size_t... Ints>                                 \
  inline constexpr void set(UTuple&& u, std::index_sequence<Ints...>) {        \
    set_all(get<Ints>(FWD(u))...);                                             \
  }                                                                            \
                                                                               \
  KOKKOS_INLINE_FUNCTION constexpr void swap(store& rhs) noexcept(             \
      std::is_nothrow_swappable_v<T> &&                                        \
      (std::is_nothrow_swappable_v<Types> && ...)) {                           \
    using std::swap;                                                           \
    swap(value, rhs.value);                                                    \
    rest.swap(rhs.rest);                                                       \
  }                                                                            \
                                                                               \
  KOKKOS_INLINE_FUNCTION constexpr void swap(const store& rhs)                 \
      const noexcept(std::is_nothrow_swappable_v<const T> &&                   \
                     (std::is_nothrow_swappable_v<const Types> && ...)) {      \
    using std::swap;                                                           \
    swap(value, rhs.value);                                                    \
    rest.swap(rhs.rest);                                                       \
  }                                                                            \
  STORE_SPACESHIP_COMP

#define STORE_COPY_ASSIGN                                                 \
  KOKKOS_INLINE_FUNCTION constexpr store& operator=(const store& other) { \
    value = other.value;                                                  \
    rest  = other.rest;                                                   \
    return *this;                                                         \
  }                                                                       \
  KOKKOS_INLINE_FUNCTION constexpr store& operator=(store& other) {       \
    value = other.value;                                                  \
    rest  = other.rest;                                                   \
    return *this;                                                         \
  }
#define STORE_CONST_COPY_ASSIGN                                               \
  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(const store& other) \
      const {                                                                 \
    value = other.value;                                                      \
    rest  = other.rest;                                                       \
    return *this;                                                             \
  }                                                                           \
  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(store& other)       \
      const {                                                                 \
    value = other.value;                                                      \
    rest  = other.rest;                                                       \
    return *this;                                                             \
  }
#define STORE_MOVE_ASSIGN                                            \
  KOKKOS_INLINE_FUNCTION constexpr store& operator=(store&& other) { \
    value = std::move(other.value);                                  \
    rest  = std::move(other.rest);                                   \
    return *this;                                                    \
  }
#define STORE_CONST_MOVE_ASSIGN                                          \
  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(store&& other) \
      const {                                                            \
    value = std::move(other.value);                                      \
    rest  = std::move(other.rest);                                       \
    return *this;                                                        \
  }

#define DELETED_STORE_COPY_ASSIGN                    \
  constexpr store& operator=(store&)       = delete; \
  constexpr store& operator=(const store&) = delete;
#define DELETED_STORE_CONST_COPY_ASSIGN                          \
  constexpr const store& operator=(store&) const       = delete; \
  constexpr const store& operator=(const store&) const = delete;
#define DELETED_STORE_MOVE_ASSIGN constexpr store& operator=(store&&) = delete;
#define DELETED_STORE_CONST_MOVE_ASSIGN \
  constexpr const store& operator=(store&&) const = delete;

template <class T, class... Types>
struct store<Bools<false, false, false, false>, T, Types...> {
  STORE_COMMON_FUNCS
  DELETED_STORE_COPY_ASSIGN
  DELETED_STORE_CONST_COPY_ASSIGN
  DELETED_STORE_MOVE_ASSIGN
  DELETED_STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<false, false, false, true>, T, Types...> {
  STORE_COMMON_FUNCS
  DELETED_STORE_COPY_ASSIGN
  DELETED_STORE_CONST_COPY_ASSIGN
  DELETED_STORE_MOVE_ASSIGN
  STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<false, false, true, false>, T, Types...> {
  STORE_COMMON_FUNCS
  DELETED_STORE_COPY_ASSIGN
  DELETED_STORE_CONST_COPY_ASSIGN
  STORE_MOVE_ASSIGN
  DELETED_STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<false, false, true, true>, T, Types...> {
  STORE_COMMON_FUNCS
  DELETED_STORE_COPY_ASSIGN
  DELETED_STORE_CONST_COPY_ASSIGN
  STORE_MOVE_ASSIGN
  STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<false, true, false, false>, T, Types...> {
  STORE_COMMON_FUNCS
  DELETED_STORE_COPY_ASSIGN
  STORE_CONST_COPY_ASSIGN
  DELETED_STORE_MOVE_ASSIGN
  DELETED_STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<false, true, false, true>, T, Types...> {
  STORE_COMMON_FUNCS
  DELETED_STORE_COPY_ASSIGN
  STORE_CONST_COPY_ASSIGN
  DELETED_STORE_MOVE_ASSIGN
  STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<false, true, true, false>, T, Types...> {
  STORE_COMMON_FUNCS
  DELETED_STORE_COPY_ASSIGN
  STORE_CONST_COPY_ASSIGN
  STORE_MOVE_ASSIGN
  DELETED_STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<false, true, true, true>, T, Types...> {
  STORE_COMMON_FUNCS
  DELETED_STORE_COPY_ASSIGN
  STORE_CONST_COPY_ASSIGN
  STORE_MOVE_ASSIGN
  STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<true, false, false, false>, T, Types...> {
  STORE_COMMON_FUNCS
  STORE_COPY_ASSIGN
  DELETED_STORE_CONST_COPY_ASSIGN
  DELETED_STORE_MOVE_ASSIGN
  DELETED_STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<true, false, false, true>, T, Types...> {
  STORE_COMMON_FUNCS
  STORE_COPY_ASSIGN
  DELETED_STORE_CONST_COPY_ASSIGN
  DELETED_STORE_MOVE_ASSIGN
  STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<true, false, true, false>, T, Types...> {
  STORE_COMMON_FUNCS
  STORE_COPY_ASSIGN
  DELETED_STORE_CONST_COPY_ASSIGN
  STORE_MOVE_ASSIGN
  DELETED_STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<true, false, true, true>, T, Types...> {
  STORE_COMMON_FUNCS
  STORE_COPY_ASSIGN
  DELETED_STORE_CONST_COPY_ASSIGN
  STORE_MOVE_ASSIGN
  STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<true, true, false, false>, T, Types...> {
  STORE_COMMON_FUNCS
  STORE_COPY_ASSIGN
  STORE_CONST_COPY_ASSIGN
  DELETED_STORE_MOVE_ASSIGN
  DELETED_STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<true, true, false, true>, T, Types...> {
  STORE_COMMON_FUNCS
  STORE_COPY_ASSIGN
  STORE_CONST_COPY_ASSIGN
  DELETED_STORE_MOVE_ASSIGN
  STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<true, true, true, false>, T, Types...> {
  STORE_COMMON_FUNCS
  STORE_COPY_ASSIGN
  STORE_CONST_COPY_ASSIGN
  STORE_MOVE_ASSIGN
  DELETED_STORE_CONST_MOVE_ASSIGN
};

template <class T, class... Types>
struct store<Bools<true, true, true, true>, T, Types...> {
  STORE_COMMON_FUNCS
  STORE_COPY_ASSIGN
  STORE_CONST_COPY_ASSIGN
  STORE_MOVE_ASSIGN
  STORE_CONST_MOVE_ASSIGN
};

#if !defined(CEXA_HAS_CXX20)
template <bool bt1, bool bt2, bool bt3, bool bt4, class T, class... TTypes,
          bool bu1, bool bu2, bool bu3, bool bu4, class U, class... UTypes>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(
    const store<Bools<bt1, bt2, bt3, bt4>, T, TTypes...>& lhs,
    const store<Bools<bu1, bu2, bu3, bu4>, U, UTypes...>& rhs) {
  return lhs.value == rhs.value && lhs.rest == rhs.rest;
}

template <bool bt1, bool bt2, bool bt3, bool bt4, class T, class... TTypes,
          bool bu1, bool bu2, bool bu3, bool bu4, class U, class... UTypes>
KOKKOS_INLINE_FUNCTION constexpr bool operator<(
    const store<Bools<bt1, bt2, bt3, bt4>, T, TTypes...>& lhs,
    const store<Bools<bu1, bu2, bu3, bu4>, U, UTypes...>& rhs) {
  if (lhs.value < rhs.value) {
    return true;
  } else if (lhs.value > rhs.value) {
    return false;
  } else {
    return lhs.rest < rhs.rest;
  }
}
#endif

#undef STORE_COMMON_FUNCS
#undef STORE_COPY_ASSIGN
#undef STORE_CONST_COPY_ASSIGN
#undef STORE_MOVE_ASSIGN
#undef STORE_CONST_MOVE_ASSIGN
#undef DELETED_STORE_COPY_ASSIGN
#undef DELETED_STORE_CONST_COPY_ASSIGN
#undef DELETED_STORE_MOVE_ASSIGN
#undef DELETED_STORE_CONST_MOVE_ASSIGN

#define TUPLE_ASSIGN_HELPER_CONSTRUCTORS                                  \
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple_assign_helper() = default;    \
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper(                   \
      const tuple_assign_helper&) {}                                      \
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper(                   \
      tuple_assign_helper&&) {}                                           \
  KOKKOS_DEFAULTED_FUNCTION CEXA_CONSTEXPR_CXX20 ~tuple_assign_helper() = \
      default;

template <bool copy_assignable, bool const_copy_assignable,
          bool move_assignable, bool const_move_assignable>
struct tuple_assign_helper;

template <>
struct tuple_assign_helper<false, false, false, false> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&) = delete;
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&) const =
      delete;
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&&) = delete;
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&&) const =
      delete;
};

template <>
struct tuple_assign_helper<false, false, false, true> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&) = delete;
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&) const =
      delete;
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&&) = delete;
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&&) const {
    return *this;
  }
};

template <>
struct tuple_assign_helper<false, false, true, false> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&) = delete;
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&) const =
      delete;
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&&) {
    return *this;
  }
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&&) const =
      delete;
};

template <>
struct tuple_assign_helper<false, false, true, true> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&) = delete;
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&) const =
      delete;
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&&) {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&&) const {
    return *this;
  }
};

template <>
struct tuple_assign_helper<false, true, false, false> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&) = delete;
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&) const {
    return *this;
  }
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&&) = delete;
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&&) const =
      delete;
};

template <>
struct tuple_assign_helper<false, true, false, true> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&) = delete;
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&) const {
    return *this;
  }
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&&) = delete;
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&&) const {
    return *this;
  }
};

template <>
struct tuple_assign_helper<false, true, true, false> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&) = delete;
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&) const {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&&) {
    return *this;
  }
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&&) const =
      delete;
};

template <>
struct tuple_assign_helper<false, true, true, true> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&) = delete;
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&) const {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&&) {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&&) const {
    return *this;
  }
};

template <>
struct tuple_assign_helper<true, false, false, false> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&) {
    return *this;
  }
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&) const =
      delete;
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&&) = delete;
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&&) const =
      delete;
};

template <>
struct tuple_assign_helper<true, false, false, true> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&) {
    return *this;
  }
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&) const =
      delete;
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&&) = delete;
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&&) const {
    return *this;
  }
};

template <>
struct tuple_assign_helper<true, false, true, false> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&) {
    return *this;
  }
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&) const =
      delete;
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&&) {
    return *this;
  }
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&&) const =
      delete;
};

template <>
struct tuple_assign_helper<true, false, true, true> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&) {
    return *this;
  }
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&) const =
      delete;
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&&) {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&&) const {
    return *this;
  }
};

template <>
struct tuple_assign_helper<true, true, false, false> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&) {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&) const {
    return *this;
  }
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&&) = delete;
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&&) const =
      delete;
};

template <>
struct tuple_assign_helper<true, true, false, true> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&) {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&) const {
    return *this;
  }
  constexpr tuple_assign_helper& operator=(tuple_assign_helper&&) = delete;
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&&) const {
    return *this;
  }
};

template <>
struct tuple_assign_helper<true, true, true, false> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&) {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&) const {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&&) {
    return *this;
  }
  constexpr const tuple_assign_helper& operator=(tuple_assign_helper&&) const =
      delete;
};

template <>
struct tuple_assign_helper<true, true, true, true> {
  TUPLE_ASSIGN_HELPER_CONSTRUCTORS
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&) {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&) const {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr tuple_assign_helper& operator=(
      tuple_assign_helper&&) {
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const tuple_assign_helper& operator=(
      tuple_assign_helper&&) const {
    return *this;
  }
};

#undef TUPLE_ASSIGN_HELPER_CONSTRUCTORS
}  // namespace impl

#if defined(CEXA_HAS_CXX20)
#define CEXA_EXPLICIT(expr) explicit(expr)
#else
// #define CEXA_EXPLICIT(expr) explicit
#define CEXA_EXPLICIT(expr)
#endif

template <typename... Types>
class tuple;

template <>
class tuple<> {
 public:
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple() = default;

  KOKKOS_DEFAULTED_FUNCTION constexpr tuple(tuple& u)       = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple(const tuple& u) = default;

  KOKKOS_DEFAULTED_FUNCTION CEXA_CONSTEXPR_CXX20 ~tuple() = default;

  KOKKOS_INLINE_FUNCTION constexpr void swap(tuple&) noexcept {}
  KOKKOS_INLINE_FUNCTION constexpr void swap(const tuple&) const noexcept {}
#if defined(CEXA_HAS_CXX20)
  KOKKOS_DEFAULTED_FUNCTION auto operator<=>(const tuple&) const = default;
#endif
};

#if !defined(CEXA_HAS_CXX20)
KOKKOS_INLINE_FUNCTION constexpr bool operator==(const tuple<>&,
                                                 const tuple<>&) {
  return true;
}

KOKKOS_INLINE_FUNCTION constexpr bool operator<(const tuple<>&,
                                                const tuple<>&) {
  return false;
}
#endif

template <typename... Types>
class tuple
    : impl::tuple_assign_helper<
          std::conjunction_v<std::is_copy_assignable<Types>...>,
          std::conjunction_v<std::is_copy_assignable<const Types>...>,
          std::conjunction_v<std::is_move_assignable<Types>...>,
          std::conjunction_v<std::is_assignable<const Types&, Types&&>...>> {
 private:
  template <typename... Ts>
  using T0 = typename impl::nth_type<0, Ts...>::type;
  template <typename... Ts>
  using T1 = typename impl::nth_type<sizeof...(Ts) == 1 ? 0 : 1, Ts...>::type;

  struct converting_tag {};
  struct tuple_like_tag {};

  template <class... UTypes>
  friend class tuple;

  impl::store_<Types...> values;

#define IMPL_CONVERTING_TUPLE_CONSTRUCTOR(CONST, REF)                          \
 public:                                                                       \
  template <                                                                   \
      typename... UTypes,                                                      \
      class = std::enable_if_t<std::conjunction_v<                             \
          std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,           \
          impl::all_types_constructible<tuple<Types...>,                       \
                                        CONST tuple<UTypes...> REF>,           \
          std::disjunction<                                                    \
              std::bool_constant<sizeof...(Types) != 1>,                       \
              std::negation<std::disjunction<                                  \
                  std::is_convertible<                                         \
                      decltype(std::declval<CONST tuple<UTypes...> REF>()),    \
                      T0<Types...>>,                                           \
                  std::is_constructible<                                       \
                      T0<Types...>,                                            \
                      decltype(std::declval<CONST tuple<UTypes...> REF>())>,   \
                  std::is_same<T0<Types...>, T0<UTypes...>>>>>,                \
          std::negation<impl::any_types_reference_constructs_from_temporary<   \
              tuple<Types...>, CONST tuple<UTypes...> REF>>>>>                 \
  KOKKOS_INLINE_FUNCTION CEXA_EXPLICIT(                                        \
      !(impl::all_types_convertible_v<                                         \
          CONST tuple<UTypes...> REF,                                          \
          tuple<Types...>>)) constexpr tuple(CONST tuple<UTypes...> REF other) \
      : tuple(converting_tag{}, FWD(other),                                    \
              std::make_index_sequence<sizeof...(Types)>{}) {}

#define IMPL_PAIR_CONSTRUCTOR(CONST, REF)                                    \
  template <class U1, class U2,                                              \
            class = std::enable_if_t<                                        \
                sizeof...(Types) == 2 &&                                     \
                std::is_constructible_v<                                     \
                    T0<Types...>,                                            \
                    decltype(std::get<0>(FWD(                                \
                        (std::declval<CONST std::pair<U1, U2> REF>()))))> && \
                std::is_constructible_v<                                     \
                    T1<Types...>,                                            \
                    decltype(std::get<1>(FWD(                                \
                        (std::declval<CONST std::pair<U1, U2> REF>()))))> && \
                !impl::reference_constructs_from_temporary_v<                \
                    T0<Types...>,                                            \
                    decltype(std::get<0>(FWD(                                \
                        (std::declval<CONST std::pair<U1, U2> REF>()))))> && \
                !impl::reference_constructs_from_temporary_v<                \
                    T1<Types...>,                                            \
                    decltype(std::get<1>(FWD(                                \
                        (std::declval<CONST std::pair<U1, U2> REF>()))))>>>  \
  inline constexpr CEXA_EXPLICIT(                                            \
      (!std::is_convertible_v<                                               \
           decltype(std::get<0>(                                             \
               FWD((std::declval<CONST std::pair<U1, U2> REF>())))),         \
           T0<Types...>> ||                                                  \
       !std::is_convertible_v<                                               \
           decltype(std::get<1>(                                             \
               FWD((std::declval<CONST std::pair<U1, U2> REF>())))),         \
           T1<Types...>>)) tuple(CONST std::pair<U1, U2> REF u)              \
      : values(std::get<0>(FWD(u)), std::get<1>(FWD(u))) {}

  template <class UTuple, std::size_t... Ints>
  KOKKOS_INLINE_FUNCTION constexpr tuple(converting_tag, UTuple&& u,
                                         std::index_sequence<Ints...>)
      : values(get<Ints>(FWD(u))...) {}

  template <class UTuple, std::size_t... Ints>
  inline constexpr tuple(tuple_like_tag, UTuple&& u,
                         std::index_sequence<Ints...>)
      : values(std::get<Ints>(FWD(u))...) {}

 public:
  // tuple.cnstr
  template <
      class Dummy = void,
      class       = std::enable_if_t<std::conjunction_v<
                std::is_same<Dummy, void>, std::is_default_constructible<Types>...>>>
  KOKKOS_INLINE_FUNCTION CEXA_EXPLICIT(
      (!impl::empty_copy_list_initializable_v<Types> ||
       ...)) constexpr tuple() noexcept((std::
                                             is_nothrow_default_constructible_v<
                                                 Types> &&
                                         ...))
      : values{} {}

  template <
      class Dummy = void,
      class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                     (sizeof...(Types) >= 1 &&
                                (std::is_copy_constructible_v<Types> && ...))>>
  KOKKOS_INLINE_FUNCTION
  CEXA_EXPLICIT((!std::is_constructible_v<Types, const Types&> || ...)) constexpr tuple(
      const Types&... vals) noexcept((std::
                                          is_nothrow_copy_constructible_v<
                                              Types> &&
                                      ...))
      : values(vals...) {}

  template <
      class... UTypes,
      class = std::enable_if_t<std::conjunction_v<
          std::bool_constant<sizeof...(Types) == sizeof...(UTypes) &&
                             sizeof...(Types) >= 1>,
          std::negation<
              std::conjunction<std::is_same<UTypes&&, const Types&>...>>,
          std::conditional_t<
              sizeof...(Types) == 1,
              std::negation<std::is_same<impl::remove_cvref_t<T0<UTypes...>>,
                                         tuple<Types...>>>,
              std::true_type>,
          std::conditional_t<
              sizeof...(Types) == 2 ||
                  sizeof...(Types) == 3,
              std::disjunction<std::negation<std::is_same<
                                   impl::remove_cvref_t<T0<UTypes...>>,
                                   std::allocator_arg_t>>,
                               std::is_same<impl::remove_cvref_t<T0<Types...>>,
                                            std::allocator_arg_t>>,
              std::true_type>,
          std::negation<
              impl::reference_constructs_from_temporary<Types, UTypes&&>>...,
          std::is_constructible<Types, UTypes>...>>>
  KOKKOS_INLINE_FUNCTION CEXA_EXPLICIT(
      (!std::is_convertible_v<UTypes&&, Types> ||
       ...)) constexpr tuple(UTypes&&... args)
      : values(FWD(args)...) {}

  KOKKOS_DEFAULTED_FUNCTION constexpr tuple(const tuple& u) = default;
  template <
      class Dummy = void,
      class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                     (std::is_move_constructible_v<Types> && ...)>>
  KOKKOS_INLINE_FUNCTION constexpr tuple(tuple&& u) : values(FWD(u).values) {}

  IMPL_CONVERTING_TUPLE_CONSTRUCTOR(, &)
  IMPL_CONVERTING_TUPLE_CONSTRUCTOR(, &&)
  IMPL_CONVERTING_TUPLE_CONSTRUCTOR(const, &)
  IMPL_CONVERTING_TUPLE_CONSTRUCTOR(const, &&)

  IMPL_PAIR_CONSTRUCTOR(, &)
  IMPL_PAIR_CONSTRUCTOR(, &&)
  IMPL_PAIR_CONSTRUCTOR(const, &)
  IMPL_PAIR_CONSTRUCTOR(const, &&)

  template <
      class UTuple,
      class = std::enable_if_t<std::conjunction_v<
          impl::is_tuple_like<UTuple>,
          std::bool_constant<
              sizeof...(Types) ==
              tuple_size<std::remove_reference_t<UTuple>>::value>,
          impl::all_types_constructible<tuple<Types...>, UTuple&&>,
          std::negation<impl::is_tuple<impl::remove_cvref_t<UTuple>>>,
          std::negation<impl::is_subrange<impl::remove_cvref_t<UTuple>>>,
          std::negation<impl::any_types_reference_constructs_from_temporary<
              tuple<Types...>, UTuple>>,
          std::conjunction<std::bool_constant<sizeof...(Types) != 1>,
                           std::negation<std::disjunction<
                               std::is_convertible<UTuple, T0<Types...>>,
                               std::is_constructible<T0<Types...>, UTuple>>>>>>>
  inline constexpr CEXA_EXPLICIT(
      (!impl::all_types_convertible_v<UTuple&&, tuple<Types...>>))
      tuple(UTuple&& u)
      : tuple(tuple_like_tag{}, FWD(u),
              std::make_index_sequence<sizeof...(Types)>{}) {}

  KOKKOS_DEFAULTED_FUNCTION
#if defined(CEXA_HAS_CXX20)
  constexpr
#endif
      ~tuple() = default;

#undef IMPL_CONVERTING_TUPLE_CONSTRUCTOR
#undef IMPL_PAIR_CONSTRUCTOR

  // tuple.assign
  template <class... UTypes,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,
                std::is_assignable<Types&, const UTypes&>...>>>
  KOKKOS_INLINE_FUNCTION constexpr tuple&
  operator=(const tuple<UTypes...>& other) noexcept(
      (std::is_nothrow_assignable_v<Types&, const UTypes&> && ...)) {
    values = other.values;
    return *this;
  }

  template <class... UTypes,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,
                std::is_assignable<const Types&, const UTypes&>...>>>
  KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(
      const tuple<UTypes...>& other) const
      noexcept((std::is_nothrow_assignable_v<const Types&, const UTypes&> &&
                ...)) {
    values = other.values;
    return *this;
  }

  template <class... UTypes,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,
                std::is_assignable<Types&, UTypes>...>>>
  KOKKOS_INLINE_FUNCTION constexpr tuple&
  operator=(tuple<UTypes...>&& other) noexcept(
      (std::is_nothrow_assignable_v<Types&, UTypes> && ...)) {
    values = FWD(other).values;
    return *this;
  }

  template <class... UTypes,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,
                std::is_assignable<const Types&, UTypes>...>>>
  KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(
      tuple<UTypes...>&& other) const
      noexcept((std::is_nothrow_assignable_v<const Types&, UTypes> && ...)) {
    values = FWD(other).values;
    return *this;
  }

  template <class U1, class U2,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == 2>,
                std::is_assignable<T0<Types...>&, const U1&>,
                std::is_assignable<T1<Types...>&, const U2&>>>>
  inline constexpr tuple& operator=(const std::pair<U1, U2>& p) noexcept(
      std::is_nothrow_assignable_v<T0<Types...>&, const U1&> &&
      std::is_nothrow_assignable_v<T1<Types...>&, const U2&>) {
    values.value      = p.first;
    values.rest.value = p.second;
    return *this;
  }

  template <class U1, class U2,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == 2>,
                std::is_assignable<const T0<Types...>&, const U1&>,
                std::is_assignable<const T1<Types...>&, const U2&>>>>
  inline constexpr const tuple& operator=(const std::pair<U1, U2>& p) const
      noexcept(std::is_nothrow_assignable_v<const T0<Types...>&, const U1&> &&
               std::is_nothrow_assignable_v<const T1<Types...>&, const U2&>) {
    values.value      = p.first;
    values.rest.value = p.second;
    return *this;
  }

  template <class U1, class U2,
            class = std::enable_if_t<
                std::conjunction_v<std::bool_constant<sizeof...(Types) == 2>,
                                   std::is_assignable<T0<Types...>&, U1>,
                                   std::is_assignable<T1<Types...>&, U2>>>>
  inline constexpr tuple& operator=(std::pair<U1, U2>&& p) noexcept(
      std::is_nothrow_assignable_v<T0<Types...>&, U1> &&
      std::is_nothrow_assignable_v<T1<Types...>&, U2>) {
    values.value      = std::forward<U1>(p.first);
    values.rest.value = std::forward<U2>(p.second);
    return *this;
  }

  template <class U1, class U2,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == 2>,
                std::is_assignable<const T0<Types...>&, U1>,
                std::is_assignable<const T1<Types...>&, U2>>>>
  inline constexpr const tuple& operator=(std::pair<U1, U2>&& p) const
      noexcept(std::is_nothrow_assignable_v<const T0<Types...>&, U1> &&
               std::is_nothrow_assignable_v<const T1<Types...>&, U2>) {
    values.value      = std::forward<U1>(p.first);
    values.rest.value = std::forward<U2>(p.second);
    return *this;
  }

  // TODO: use conjunction_v here
  template <class UTuple,
            class = std::enable_if_t<
                impl::is_tuple_like_v<UTuple> &&
                !impl::is_tuple_v<impl::remove_cvref_t<UTuple>> &&
                !impl::is_pair<impl::remove_cvref_t<UTuple>>::value &&
                impl::is_different_from_v<UTuple, tuple> &&
                !impl::is_subrange_v<UTuple> &&
                sizeof...(Types) ==
                    tuple_size<std::remove_reference_t<UTuple>>::value>>
  // The check for is_assignable is delegated to store.set_all()
  constexpr tuple& operator=(UTuple&& u) {
    values.set(FWD(u));
    return *this;
  }

  template <class UTuple,
            class = std::enable_if_t<
                impl::is_tuple_like_v<UTuple> &&
                !impl::is_tuple_v<impl::remove_cvref_t<UTuple>> &&
                !impl::is_pair<impl::remove_cvref_t<UTuple>>::value &&
                impl::is_different_from_v<UTuple, tuple> &&
                !impl::is_subrange_v<impl::remove_cvref_t<UTuple>> &&
                sizeof...(Types) ==
                    tuple_size<std::remove_reference_t<UTuple>>::value>>
  // The check for is_assignable is delegated to store.set_all()
  constexpr const tuple& operator=(UTuple&& u) const {
    values.set(FWD(u));
    return *this;
  }
#undef FWD

  KOKKOS_INLINE_FUNCTION constexpr void swap(tuple& rhs) noexcept(
      (std::is_nothrow_swappable_v<Types> && ...)) {
    return values.swap(rhs.values);
  }

  KOKKOS_INLINE_FUNCTION constexpr void swap(const tuple& rhs) const
      noexcept((std::is_nothrow_swappable_v<const Types> && ...)) {
    return values.swap(rhs.values);
  }

  template <std::size_t I, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (I < sizeof...(Ts)), typename tuple_element<I, tuple<Ts...>>::type&>
  get(tuple<Ts...>& t) noexcept;
  template <std::size_t I, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (I < sizeof...(Ts)), typename tuple_element<I, tuple<Ts...>>::type&&>
  get(tuple<Ts...>&& t) noexcept;
  template <std::size_t I, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (I < sizeof...(Ts)), const typename tuple_element<I, tuple<Ts...>>::type&>
  get(const tuple<Ts...>& t) noexcept;
  template <std::size_t I, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (I < sizeof...(Ts)),
      const typename tuple_element<I, tuple<Ts...>>::type&&>
  get(const tuple<Ts...>&& t) noexcept;
  template <class T, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (std::is_same_v<T, Ts> || ...), T&>
  get(tuple<Ts...>& t) noexcept;
  template <class T, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (std::is_same_v<T, Ts> || ...), T&&>
  get(tuple<Ts...>&& t) noexcept;
  template <class T, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (std::is_same_v<T, Ts> || ...), const T&>
  get(const tuple<Ts...>& t) noexcept;
  template <class T, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr const std::enable_if_t<
      (std::is_same_v<T, Ts> || ...), const T&&>
  get(const tuple<Ts...>&& t) noexcept;

  // tuple.rel
#if defined(CEXA_HAS_CXX20)
  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr auto operator<=>(
      const tuple<UTypes...>& rhs) const {
    return values <=> rhs.values;
  }

  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(
      const tuple<UTypes...>& rhs) const {
    return (values <=> rhs.values) == 0;
  }
#else
  template <class... TTypes, class... UTypes>
  KOKKOS_INLINE_FUNCTION friend constexpr bool operator==(
      const tuple<TTypes...>& lhs, const tuple<UTypes...>& rhs);

  template <class... TTypes, class... UTypes>
  KOKKOS_INLINE_FUNCTION friend constexpr bool operator<(
      const tuple<TTypes...>& lhs, const tuple<UTypes...>& rhs);
#endif
};

// deduction guides
template <class... UTypes>
KOKKOS_DEDUCTION_GUIDE tuple(UTypes...) -> tuple<UTypes...>;
template <class T1, class T2>
tuple(std::pair<T1, T2>) -> tuple<T1, T2>;

// tuple.elem
template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)), typename tuple_element<I, tuple<Types...>>::type&>
get(tuple<Types...>& t) noexcept {
  return t.values.template get_value<I>();
}
template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)), typename tuple_element<I, tuple<Types...>>::type&&>
get(tuple<Types...>&& t) noexcept {
  return static_cast<tuple_element_t<I, tuple<Types...>>&&>(
      t.values.template get_value<I>());
}
template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)),
    const typename tuple_element<I, tuple<Types...>>::type&>
get(const tuple<Types...>& t) noexcept {
  return t.values.template get_value<I>();
}
template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)),
    const typename tuple_element<I, tuple<Types...>>::type&&>
get(const tuple<Types...>&& t) noexcept {
  return static_cast<const tuple_element_t<I, tuple<Types...>>&&>(
      t.values.template get_value<I>());
}
template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (std::is_same_v<T, Types> || ...), T&>
get(tuple<Types...>& t) noexcept {
  return t.values.template get_value<T>();
}
template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (std::is_same_v<T, Types> || ...), T&&>
get(tuple<Types...>&& t) noexcept {
  return static_cast<T&&>(t.values.template get_value<T>());
}
template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (std::is_same_v<T, Types> || ...), const T&>
get(const tuple<Types...>& t) noexcept {
  return t.values.template get_value<T>();
}
template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr const std::enable_if_t<
    (std::is_same_v<T, Types> || ...), const T&&>
get(const tuple<Types...>&& t) noexcept {
  return static_cast<const T&&>(t.values.template get_value<T>());
}

// tuple.rel
#if !defined(CEXA_HAS_CXX20)
template <class... TTypes, class... UTypes>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(const tuple<TTypes...>& lhs,
                                                 const tuple<UTypes...>& rhs) {
  static_assert(sizeof...(TTypes) == sizeof...(UTypes), "");
  return lhs.values == rhs.values;
}

template <class... TTypes, class... UTypes>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const tuple<TTypes...>& lhs,
                                                 const tuple<UTypes...>& rhs) {
  return !(lhs == rhs);
}

template <class... TTypes, class... UTypes>
KOKKOS_INLINE_FUNCTION constexpr bool operator<(const tuple<TTypes...>& lhs,
                                                const tuple<UTypes...>& rhs) {
  static_assert(sizeof...(TTypes) == sizeof...(UTypes), "");
  return lhs.values < rhs.values;
}

template <class... TTypes, class... UTypes>
KOKKOS_INLINE_FUNCTION constexpr bool operator<=(const tuple<TTypes...>& lhs,
                                                 const tuple<UTypes...>& rhs) {
  return !(rhs < lhs);
}

template <class... TTypes, class... UTypes>
KOKKOS_INLINE_FUNCTION constexpr bool operator>(const tuple<TTypes...>& lhs,
                                                const tuple<UTypes...>& rhs) {
  return rhs < lhs;
}

template <class... TTypes, class... UTypes>
KOKKOS_INLINE_FUNCTION constexpr bool operator>=(const tuple<TTypes...>& lhs,
                                                 const tuple<UTypes...>& rhs) {
  return !(lhs < rhs);
}
#endif

template <class... Types,
          class = std::enable_if_t<(std::is_swappable_v<Types> && ...)>>
KOKKOS_INLINE_FUNCTION constexpr void swap(tuple<Types...>& lhs,
                                           tuple<Types...>& rhs) {
  lhs.swap(rhs);
}

template <class... Types,
          class = std::enable_if_t<(std::is_swappable_v<const Types> && ...)>>
KOKKOS_INLINE_FUNCTION constexpr void swap(const tuple<Types...>& lhs,
                                           const tuple<Types...>& rhs) {
  lhs.swap(rhs);
}

}  // namespace cexa

#undef CEXA_EXPLICIT
#undef CEXA_CONSTEXPR_CXX20
