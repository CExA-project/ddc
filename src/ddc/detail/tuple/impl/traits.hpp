#pragma once

#include <array>
#include <type_traits>
#include <functional>

#include "tuple_fwd.hpp"

namespace cexa::impl {
#if defined(CEXA_HAS_CXX20)
template <class T>
using remove_cvref = std::remove_cvref<T>;
#else
template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};
#endif
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

#if defined(CEXA_HAS_CXX20)
template <class T>
using unwrap_ref_decay = std::unwrap_ref_decay<T>;
#else
template <class T>
struct unwrap_reference {
  using type = T;
};

template <class T>
struct unwrap_reference<std::reference_wrapper<T>> {
  using type = T&;
};

template <class T>
struct unwrap_ref_decay {
  using type = typename unwrap_reference<std::decay_t<T>>::type;
};
#endif
template <class T>
using unwrap_ref_decay_t = typename unwrap_ref_decay<T>::type;

// NOTE: Since const T& t = {} is not an expression, we check that a member of
// type T can be initialized with empty braces when calling a constructor.
template <class U, class = void>
struct empty_copy_list_initializable_helper : std::false_type {};
template <class U>
struct empty_copy_list_initializable_helper<U, std::void_t<decltype(U({}))>>
    : std::true_type {};

template <class T>
struct empty_copy_list_initializable {
  struct helper {
    helper(const T&) {}
  };

  static inline constexpr bool value =
      empty_copy_list_initializable_helper<helper>::value;
};

template <class T>
inline constexpr bool empty_copy_list_initializable_v =
    empty_copy_list_initializable<T>::value;

// is_tuple_like
// tells if a type is tuple-like, tuple-like types include array, pair,
// tuple and subrange

// TODO: replace by this (simpler) impl
// template <typename T, class = void>
// struct is_tuple_like_impl : std::false_type {};
//
// template <typename T>
// struct is_tuple_like_impl<T, std::void_t<decltype(std::tuple_size_v<T>)>> :
// std::true_type {};

template <typename T>
struct is_tuple_like_impl : std::false_type {};

template <typename T, std::size_t N>
struct is_tuple_like_impl<std::array<T, N>> : std::true_type {};

template <typename T0, typename T1>
struct is_tuple_like_impl<std::pair<T0, T1>> : std::true_type {};

template <typename... Types>
struct is_tuple_like_impl<std::tuple<Types...>> : std::true_type {};

template <typename... Types>
struct is_tuple_like_impl<cexa::tuple<Types...>> : std::true_type {};

template <typename T>
struct is_tuple_like : is_tuple_like_impl<impl::remove_cvref_t<T>> {};

template <typename T>
inline constexpr bool is_tuple_like_v = is_tuple_like<T>::value;

// is_tuple
template <class T>
struct is_tuple : std::false_type {};

template <class... Types>
struct is_tuple<tuple<Types...>> : std::true_type {};

template <class T>
inline constexpr bool is_tuple_v = is_tuple<T>::value;

// is_different_from
template <class T, class U>
struct is_different_from {
  static inline constexpr bool value =
      !std::is_same_v<remove_cvref_t<T>, remove_cvref_t<U>>;
};

template <class T, class U>
inline constexpr bool is_different_from_v = is_different_from<T, U>::value;

// reference_constructs_from_temporary
template <class T, class U>
struct reference_constructs_from_temporary : std::false_type {};

template <class T, class U>
struct reference_constructs_from_temporary<const T&, U>
    : std::integral_constant<
          bool, (std::is_same_v<remove_cvref_t<T>, remove_cvref_t<U>> &&
                 !std::is_reference_v<U>) ||
                    (std::conjunction_v<
                        std::negation<
                            std::is_same<remove_cvref_t<T>, remove_cvref_t<U>>>,
                        std::is_convertible<
                            std::conditional_t<std::is_scalar_v<U> ||
                                                   std::is_void_v<U>,
                                               std::remove_cv_t<U>, U>,
                            T>>)> {};

template <class T, class U>
struct reference_constructs_from_temporary<T&&, U>
    : std::integral_constant<
          bool, (std::is_same_v<remove_cvref_t<T>, remove_cvref_t<U>> &&
                 !std::is_reference_v<U>) ||
                    (std::conjunction_v<
                        std::negation<
                            std::is_same<remove_cvref_t<T>, remove_cvref_t<U>>>,
                        std::is_convertible<
                            std::conditional_t<std::is_scalar_v<U> ||
                                                   std::is_void_v<U>,
                                               std::remove_cv_t<U>, U>,
                            T>>)> {};

template <class T, class U>
constexpr inline bool reference_constructs_from_temporary_v =
    reference_constructs_from_temporary<T, U>::value;

// common_reference helper
#if defined(CEXA_HAS_CXX20)
template <class TTuple, class UTuple, template <class> class TQual,
          template <class> class UQual, class IndexSeq>
struct common_reference_helper;

template <class TTuple, class UTuple, template <class> class TQual,
          template <class> class UQual, std::size_t... Ints>
struct common_reference_helper<TTuple, UTuple, TQual, UQual,
                               std::index_sequence<Ints...>> {
  using type = cexa::tuple<
      std::common_reference_t<TQual<std::tuple_element_t<Ints, TTuple>>,
                              UQual<std::tuple_element_t<Ints, UTuple>>>...>;
};
#endif

// common_type helper
template <class TTuple, class UTuple, class IndexSeq>
struct common_type_helper;

template <class TTuple, class UTuple, std::size_t... Ints>
struct common_type_helper<TTuple, UTuple, std::index_sequence<Ints...>> {
  using type =
      cexa::tuple<std::common_type_t<std::tuple_element_t<Ints, TTuple>,
                                     std::tuple_element_t<Ints, UTuple>>...>;
};

}  // namespace cexa::impl

// tuple.common.ref
#if defined(CEXA_HAS_CXX20)
template <class... TTypes, class UTuple, template <class> class TQual,
          template <class> class UQual>
struct std::basic_common_reference<cexa::tuple<TTypes...>, UTuple, TQual,
                                   UQual> {
  static_assert(std::is_same_v<cexa::tuple<TTypes...>,
                               std::decay_t<cexa::tuple<TTypes...>>>);
  static_assert(std::is_same_v<UTuple, std::decay_t<UTuple>>);
  static_assert(sizeof...(TTypes) ==
                std::tuple_size_v<std::remove_reference_t<UTuple>>);

  using type = typename cexa::impl::common_reference_helper<
      cexa::tuple<TTypes...>, UTuple, TQual, UQual,
      decltype(std::index_sequence_for<TTypes...>{})>::type;
};

template <class TTuple, class... UTypes, template <class> class TQual,
          template <class> class UQual>
struct std::basic_common_reference<TTuple, cexa::tuple<UTypes...>, TQual,
                                   UQual> {
  static_assert(std::is_same_v<TTuple, std::decay_t<TTuple>>);
  static_assert(std::is_same_v<cexa::tuple<UTypes...>,
                               std::decay_t<cexa::tuple<UTypes...>>>);
  static_assert(std::tuple_size_v<std::remove_reference_t<TTuple>> ==
                sizeof...(UTypes));

  using type = typename cexa::impl::common_reference_helper<
      TTuple, cexa::tuple<UTypes...>, TQual, UQual,
      decltype(std::index_sequence_for<UTypes...>{})>::type;
};
#endif

template <class... TTypes, class UTuple>
struct std::common_type<cexa::tuple<TTypes...>, UTuple> {
  static_assert(std::is_same_v<cexa::tuple<TTypes...>,
                               std::decay_t<cexa::tuple<TTypes...>>>);
  static_assert(std::is_same_v<UTuple, std::decay_t<UTuple>>);
  static_assert(sizeof...(TTypes) ==
                std::tuple_size_v<std::remove_reference_t<UTuple>>);

  using type = typename cexa::impl::common_type_helper<
      cexa::tuple<TTypes...>, UTuple,
      decltype(std::index_sequence_for<TTypes...>{})>::type;
};

template <class TTuple, class... UTypes>
struct std::common_type<TTuple, cexa::tuple<UTypes...>> {
  static_assert(std::is_same_v<TTuple, std::decay_t<TTuple>>);
  static_assert(std::is_same_v<cexa::tuple<UTypes...>,
                               std::decay_t<cexa::tuple<UTypes...>>>);
  static_assert(std::tuple_size_v<std::remove_reference_t<TTuple>> ==
                sizeof...(UTypes));

  using type = typename cexa::impl::common_type_helper<
      TTuple, cexa::tuple<UTypes...>,
      decltype(std::index_sequence_for<UTypes...>{})>::type;
};
