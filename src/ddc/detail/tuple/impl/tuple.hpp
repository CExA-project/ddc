// NOLINTBEGIN(readability-identifier-naming)
// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include <Kokkos_Macros.hpp>
#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
#    include <compare>
#endif

#include "helper.hpp"
#include "macros.hpp"
#include "traits.hpp"
#include "tuple_fwd.hpp"

namespace cexa {

namespace impl {

template <std::size_t I, typename... Ts>
struct nth_type;
template <typename T, typename... Ts>
struct nth_type<0, T, Ts...>
{
    using type = T;
};
template <std::size_t I, typename T, typename... Ts>
struct nth_type<I, T, Ts...> : nth_type<I - 1, Ts...>
{
};

#define FWD(x) std::forward<decltype(x)>(x)
// #define FWD(x) static_cast<decltype(x)>(x)

template <class Tuple, class UTuple>
struct all_types_constructible : std::false_type
{
};

template <class... Types, class UTuple>
struct all_types_constructible<tuple<Types...>, UTuple>
{
    using t = void;
    template <class Seq>
    struct all_types_constructible_helper;
    template <std::size_t... Ints>
    struct all_types_constructible_helper<std::index_sequence<Ints...>>
    {
        static constexpr bool value
                = (std::is_constructible_v<Types, decltype(get<Ints>(std::declval<UTuple>()))>
                   && ...);
    };

    static constexpr bool value
            = all_types_constructible_helper<decltype(std::index_sequence_for<Types...> {})>::value;
};

template <class Tuple, class UTuple>
inline constexpr bool all_types_constructible_v = all_types_constructible<Tuple, UTuple>::value;

template <class UTuple, class Tuple>
struct all_types_convertible : std::false_type
{
};

template <class... Types, class UTuple>
struct all_types_convertible<UTuple, tuple<Types...>>
{
    template <class Seq>
    struct all_types_convertible_helper;
    template <std::size_t... Ints>
    struct all_types_convertible_helper<std::index_sequence<Ints...>>
    {
        static constexpr bool value
                = (std::is_convertible_v<decltype(get<Ints>(FWD(std::declval<UTuple>()))), Types>
                   && ...);
    };

    static constexpr bool value
            = all_types_convertible_helper<decltype(std::index_sequence_for<Types...> {})>::value;
};

template <class UTuple, class Tuple>
inline constexpr bool all_types_convertible_v = all_types_convertible<UTuple, Tuple>::value;

template <class T>
struct is_pair : std::false_type
{
};

template <class T, class U>
struct is_pair<std::pair<T, U>> : std::true_type
{
};

template <class Tuple, class UTuple>
struct any_types_reference_constructs_from_temporary : std::false_type
{
};

template <class... Types, class UTuple>
struct any_types_reference_constructs_from_temporary<tuple<Types...>, UTuple>
{
    template <class Seq>
    struct any_types_reference_constructs_from_temporary_helper;
    template <std::size_t... Ints>
    struct any_types_reference_constructs_from_temporary_helper<std::index_sequence<Ints...>>
    {
        static constexpr bool value
                = (impl::reference_constructs_from_temporary_v<
                           Types,
                           decltype(get<Ints>(FWD(std::declval<UTuple>())))>
                   || ...);
    };

    static constexpr bool value = any_types_reference_constructs_from_temporary_helper<
            decltype(std::index_sequence_for<Types...> {})>::value;
};

template <class Tuple, class UTuple>
inline constexpr bool any_types_reference_constructs_from_temporary_v
        = any_types_reference_constructs_from_temporary<Tuple, UTuple>::value;

template <typename... Types>
struct store;

template <class T>
struct is_store : std::false_type
{
};

template <class... Types>
struct is_store<store<Types...>> : std::true_type
{
};

template <class T>
inline constexpr bool is_store_v = is_store<std::remove_cvref_t<T>>::value;

template <>
struct store<>
{
    KOKKOS_DEFAULTED_FUNCTION constexpr store() = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr store(store const&) = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr store(store&&) = default;
#if defined(CEXA_HAS_CXX23)
    KOKKOS_DEFAULTED_FUNCTION constexpr store(store&) noexcept = default;
    KOKKOS_INLINE_FUNCTION constexpr store(store const&&) noexcept {};
#endif
    KOKKOS_DEFAULTED_FUNCTION constexpr ~store() = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr store& operator=(store const&) = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr store& operator=(store&&) = default;

#if defined(CEXA_HAS_CXX23)
    // NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature,cert-oop54-cpp)
    KOKKOS_INLINE_FUNCTION constexpr const store& operator=(store const&) const noexcept
    {
        return *this;
    }
    // NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
    KOKKOS_INLINE_FUNCTION constexpr const store& operator=(store&&) const noexcept
    {
        return *this;
    }
#endif

    KOKKOS_INLINE_FUNCTION constexpr void swap(store&) noexcept {}
#if defined(CEXA_HAS_CXX23)
    KOKKOS_INLINE_FUNCTION constexpr void swap(store const&) const noexcept {}
#endif

#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
    KOKKOS_DEFAULTED_FUNCTION auto operator<=>(store const&) const = default;
#else
    KOKKOS_INLINE_FUNCTION friend constexpr bool operator==(store<> const&, store<> const&)
    {
        return true;
    }
    KOKKOS_INLINE_FUNCTION friend constexpr bool operator<(store<> const&, store<> const&)
    {
        return false;
    }
#endif
};

template <class T, class... Types>
struct store<T, Types...>
{
    T value {};
    store<Types...> rest;

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    KOKKOS_DEFAULTED_FUNCTION constexpr store() = default;

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    KOKKOS_DEFAULTED_FUNCTION constexpr store(store const& other) = default;

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    KOKKOS_INLINE_FUNCTION constexpr store(store&& other) noexcept(
            std::is_nothrow_move_constructible_v<T>
            && (std::is_nothrow_move_constructible_v<Types> && ...))
        : value(FWD(other.value))
        , rest(FWD(other.rest))
    {
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <typename U, typename... UTypes>
        requires(!is_store_v<U> && (!is_store_v<UTypes> && ...))
    KOKKOS_INLINE_FUNCTION constexpr explicit store(U&& u, UTypes&&... args) noexcept(
            std::is_nothrow_move_constructible_v<T>
            && (std::is_nothrow_move_constructible_v<Types> && ...))
        : value(FWD(u))
        , rest(FWD(args)...)
    {
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    KOKKOS_DEFAULTED_FUNCTION constexpr ~store() = default;

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <class U, class... UTypes>
        requires(
                sizeof...(UTypes) == sizeof...(Types)
                && !(std::is_same_v<U, T> && (std::is_same_v<UTypes, Types> && ...))
                && std::is_assignable_v<T&, U const&>)
    KOKKOS_INLINE_FUNCTION constexpr store& operator=(store<U, UTypes...> const& other)
    {
        value = other.value;
        rest = other.rest;
        return *this;
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <class U, class... UTypes>
        requires(
                sizeof...(UTypes) == sizeof...(Types)
                && !(std::is_same_v<U, T> && (std::is_same_v<UTypes, Types> && ...))
                && std::is_assignable_v<T&, U &&>)
    KOKKOS_INLINE_FUNCTION constexpr store& operator=(store<U, UTypes...>&& other)
    {
        value = FWD(other.value);
        rest = std::move(other.rest);
        return *this;
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    // FIXME: defaulting this operator leads to compile errors where the
    // generated constructor would be ill-formed
    // NOLINTNEXTLINE(hicpp-use-equals-default, modernize-use-equals-default)
    KOKKOS_INLINE_FUNCTION constexpr store& operator=(store const& other) noexcept(
            std::is_nothrow_copy_assignable_v<T>
            && (std::is_nothrow_copy_assignable_v<Types> && ...))
        requires(std::is_copy_assignable_v<T>)
    {
        if (this != &other) {
            value = other.value;
            rest = other.rest;
        }
        return *this;
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    KOKKOS_INLINE_FUNCTION constexpr store& operator=(store&& other) noexcept(
            std::is_nothrow_move_assignable_v<T>
            && (std::is_nothrow_move_assignable_v<Types> && ...))
        requires(std::is_move_assignable_v<T>)
    {
        value = std::move(other.value);
        rest = std::move(other.rest);
        return *this;
    }

#if defined(CEXA_HAS_CXX23)
    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    KOKKOS_INLINE_FUNCTION constexpr const store& operator=(store const& other) const
        requires(std::is_copy_assignable_v<T const>)
    {
        if (this != &other) {
            value = other.value;
            rest = other.rest;
        }
        return *this;
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    KOKKOS_INLINE_FUNCTION constexpr const store& operator=(store&& other) const noexcept(
            std::is_nothrow_move_assignable_v<T const>
            && (std::is_nothrow_move_assignable_v<Types const> && ...))
        requires(std::is_assignable_v<T const&, T &&>)
    {
        value = std::move(other.value);
        rest = std::move(other.rest);
        return *this;
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <class U, class... UTypes>
        requires(
                sizeof...(UTypes) == sizeof...(Types)
                && !(std::is_same_v<U, T> && (std::is_same_v<UTypes, Types> && ...))
                && std::is_assignable_v<T const&, U const&>)
    KOKKOS_INLINE_FUNCTION constexpr const store& operator=(store<U, UTypes...> const& other) const
    {
        value = other.value;
        rest = other.rest;
        return *this;
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <class U, class... UTypes>
        requires(
                sizeof...(UTypes) == sizeof...(Types)
                && !(std::is_same_v<U, T> && (std::is_same_v<UTypes, Types> && ...))
                && std::is_assignable_v<T const&, U &&>)
    KOKKOS_INLINE_FUNCTION constexpr const store& operator=(store<U, UTypes...>&& other) const
    {
        value = FWD(other.value);
        rest = std::move(other.rest);
        return *this;
    }
#endif

    template <std::size_t I>
    KOKKOS_INLINE_FUNCTION constexpr tuple_element_t<I, tuple<T, Types...>>& get_value() noexcept
    {
        if constexpr (I == 0) {
            return value;
        } else {
            return rest.template get_value<I - 1>();
        }
    }

    template <std::size_t I>
    KOKKOS_INLINE_FUNCTION constexpr const tuple_element_t<I, tuple<T, Types...>>& get_value()
            const noexcept
    {
        if constexpr (I == 0) {
            return value;
        } else {
            return rest.template get_value<I - 1>();
        }
    }

    template <class Type>
    KOKKOS_INLINE_FUNCTION constexpr Type& get_value() noexcept
    {
        if constexpr (std::is_same_v<Type, T>) {
            return value;
        } else {
            return rest.template get_value<Type>();
        }
    }

    template <class Type>
    KOKKOS_INLINE_FUNCTION constexpr const Type& get_value() const noexcept
    {
        if constexpr (std::is_same_v<Type, T>) {
            return value;
        } else {
            return rest.template get_value<Type>();
        }
    }

    template <class U>
        requires std::is_assignable_v<T&, decltype(std::forward<U>(std::declval<U&&>()))>
    KOKKOS_INLINE_FUNCTION constexpr void set_all(U&& u)
    {
        value = u;
    }

    template <class U, class... UTypes>
        requires std::is_assignable_v<T&, decltype(std::forward<U>(std::declval<U&&>()))>
    KOKKOS_INLINE_FUNCTION constexpr void set_all(U&& head, UTypes&&... tail)
    {
        value = head;
        rest.set_all(FWD(tail)...);
    }

    template <class UTuple>
    constexpr void set(UTuple&& u)
    {
        set(FWD(u), std::make_index_sequence<1 + sizeof...(Types)> {});
    }

    template <class UTuple, std::size_t... Ints>
    constexpr void set(UTuple&& u, std::index_sequence<Ints...>)
    {
        set_all(get<Ints>(FWD(u))...);
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    KOKKOS_INLINE_FUNCTION constexpr void swap(store& rhs) noexcept(
            std::is_nothrow_swappable_v<T> && (std::is_nothrow_swappable_v<Types> && ...))
    {
        using std::swap;
        swap(value, rhs.value);
        rest.swap(rhs.rest);
    }

#if defined(CEXA_HAS_CXX23)
    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    KOKKOS_INLINE_FUNCTION constexpr void swap(store const& rhs) const noexcept(
            std::is_nothrow_swappable_v<T const>
            && (std::is_nothrow_swappable_v<Types const> && ...))
    {
        using std::swap;
        swap(value, rhs.value);
        rest.swap(rhs.rest);
    }

#endif

#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <class U, class... UTypes>
        requires std::three_way_comparable_with<T, U>
    KOKKOS_INLINE_FUNCTION constexpr auto operator<=>(store<U, UTypes...> const& rhs) const
            -> std::common_comparison_category_t<
                    decltype(value <=> rhs.value),
                    decltype(rest <=> rhs.rest)>
    {
        auto res = value <=> rhs.value;
        return res != 0 ? res : rest <=> rhs.rest;
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <class U, class... UTypes>
    KOKKOS_INLINE_FUNCTION constexpr std::weak_ordering operator<=>(
            store<U, UTypes...> const& rhs) const
    {
        if (value < rhs.value) {
            return std::weak_ordering::less;
        }
        if (rhs.value < value) {
            return std::weak_ordering::greater;
        }
        return static_cast<std::weak_ordering>(rest <=> rhs.rest);
    }

    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <class U, class... UTypes>
    KOKKOS_INLINE_FUNCTION constexpr bool operator==(store<U, UTypes...> const& rhs) const
    {
        return operator<=>(rhs) == 0;
    }
#else
    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <class U, class... UTypes>
    KOKKOS_INLINE_FUNCTION constexpr bool operator==(store<U, UTypes...> const& rhs) const
    {
        return value == rhs.value && rest == rhs.rest;
    }
    CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
    template <class U, class... UTypes>
    KOKKOS_INLINE_FUNCTION constexpr bool operator<(store<U, UTypes...> const& rhs) const
    {
        return value < rhs.value || (value == rhs.value && rest < rhs.rest);
    }
#endif
};
} // namespace impl

template <class... Types>
class tuple;

template <>
class tuple<>
{
public:
    KOKKOS_DEFAULTED_FUNCTION constexpr tuple() noexcept = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr tuple(tuple const& u) noexcept = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr tuple(tuple&& u) noexcept = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr ~tuple() = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(tuple const& u) noexcept = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(tuple&& u) noexcept = default;
#if defined(CEXA_HAS_CXX23)
    // We don't care about self assignment since this is an empty class
    // NOLINTNEXTLINE(cert-oop54-cpp)
    KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(tuple const& u) const noexcept
    {
        return *this;
    }
    KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(tuple&& u) const noexcept
    {
        return *this;
    }
#endif

    KOKKOS_INLINE_FUNCTION constexpr void swap(tuple&) noexcept {}
#if defined(CEXA_HAS_CXX23)
    KOKKOS_INLINE_FUNCTION constexpr void swap(tuple const&) const noexcept {}
#endif

#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
    KOKKOS_DEFAULTED_FUNCTION auto operator<=>(tuple const&) const = default;
#endif
};

#if !defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
KOKKOS_INLINE_FUNCTION constexpr bool operator==(tuple<> const&, tuple<> const&)
{
    return true;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(tuple<> const&, tuple<> const&)
{
    return false;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator<(tuple<> const&, tuple<> const&)
{
    return false;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator<=(tuple<> const&, tuple<> const&)
{
    return true;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator>(tuple<> const&, tuple<> const&)
{
    return false;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator>=(tuple<> const&, tuple<> const&)
{
    return true;
}
#endif

namespace impl {
// We use sfinae instead of a concept since the concept would depend on itself
// NOLINTBEGIN(bugprone-macro-parentheses)
#define CONVERTING_TUPLE_CTOR_CONSTRAINTS(CONST, REF)                                              \
    class = std::enable_if_t<std::conjunction_v<                                                   \
            std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,                             \
            impl::all_types_constructible<tuple<Types...>, CONST tuple<UTypes...> REF>,            \
            std::disjunction<                                                                      \
                    std::bool_constant<sizeof...(Types) != 1>,                                     \
                    std::negation<std::disjunction<                                                \
                            std::is_convertible<                                                   \
                                    decltype(std::declval<CONST tuple<UTypes...> REF>()),          \
                                    T0<Types...>>,                                                 \
                            std::is_constructible<                                                 \
                                    T0<Types...>,                                                  \
                                    decltype(std::declval<CONST tuple<UTypes...> REF>())>,         \
                            std::is_same<T0<Types...>, T0<UTypes...>>>>>,                          \
            std::negation<impl::any_types_reference_constructs_from_temporary<                     \
                    tuple<Types...>,                                                               \
                    CONST tuple<UTypes...> REF>>>>
// NOLINTEND(bugprone-macro-parentheses)

template <class Tuple, class UPair>
concept pair_constructible = tuple_size_v<Tuple> == 2
                             && std::constructible_from<
                                     tuple_element_t<0, std::remove_cvref_t<Tuple>>,
                                     decltype(std::get<0>(FWD((std::declval<UPair>()))))>
                             && std::constructible_from<
                                     tuple_element_t<1, std::remove_cvref_t<Tuple>>,
                                     decltype(std::get<1>(FWD((std::declval<UPair>()))))>
                             && !impl::reference_constructs_from_temporary_v<
                                     tuple_element_t<0, std::remove_cvref_t<Tuple>>,
                                     decltype(std::get<0>(FWD((std::declval<UPair>()))))>
                             && !impl::reference_constructs_from_temporary_v<
                                     tuple_element_t<1, std::remove_cvref_t<Tuple>>,
                                     decltype(std::get<1>(FWD((std::declval<UPair>()))))>;

template <class Tuple, class UTuple>
concept tuple_like_constructible
        = impl::is_tuple_like_v<UTuple>
          && tuple_size_v<Tuple> == tuple_size_v<std::remove_reference_t<UTuple>>
          && impl::all_types_constructible_v<Tuple, UTuple&&>
          && !impl::is_tuple_v<std::remove_cvref_t<UTuple>>
          && !impl::is_subrange_v<std::remove_cvref_t<UTuple>>
          && !impl::any_types_reference_constructs_from_temporary_v<Tuple, UTuple>
          && (tuple_size_v<Tuple> != 1
              || !(std::convertible_to<UTuple, tuple_element_t<0, Tuple>>
                   || std::constructible_from<tuple_element_t<0, Tuple>, UTuple>));
} // namespace impl

template <typename... Types>
class tuple
{
private:
    template <typename... Ts>
    using T0 = typename impl::nth_type<0, Ts...>::type;
    template <typename... Ts>
    using T1 = typename impl::nth_type<sizeof...(Ts) == 1 ? 0 : 1, Ts...>::type;

    struct converting_tag
    {
    };
    struct tuple_like_tag
    {
    };

    template <class... UTypes>
    friend class tuple;

    impl::store<Types...> m_values;

    template <class UTuple, std::size_t... Ints>
    KOKKOS_INLINE_FUNCTION constexpr tuple(converting_tag, UTuple&& u, std::index_sequence<Ints...>)
        : m_values(get<Ints>(FWD(u))...)
    {
    }

    template <class UTuple, std::size_t... Ints>
    constexpr tuple(tuple_like_tag, UTuple&& u, std::index_sequence<Ints...>)
        : m_values(std::get<Ints>(FWD(u))...)
    {
    }

public:
    // tuple.cnstr
    KOKKOS_INLINE_FUNCTION explicit((
            !impl::is_empty_copy_list_initializable_v<Types>
            || ...)) constexpr tuple() noexcept((std::is_nothrow_default_constructible_v<Types> && ...))
        requires(std::is_default_constructible_v<Types> && ...)
        : m_values {}
    {
    }

    // NOTE: We have to use sfinae for these constructors, as using a requires
    // clause would lead to a compile error about the atomic constraint depending
    // on itself
    // NOLINTBEGIN(modernize-use-constraints)
    template <
            class Dummy = void,
            class = std::enable_if_t<
                    std::is_same_v<Dummy, void> && (sizeof...(Types) >= 1)
                    && (std::is_copy_constructible_v<Types> && ...)>>
    KOKKOS_INLINE_FUNCTION explicit((!std::is_convertible_v<Types const&, Types> || ...)) constexpr tuple(
            Types const&... vals) noexcept((std::is_nothrow_copy_constructible_v<Types> && ...))
        : m_values(vals...)
    {
    }

    template <
            class... UTypes,
            class = std::enable_if_t<std::conjunction_v<
                    std::bool_constant<
                            sizeof...(Types) == sizeof...(UTypes) && sizeof...(Types) >= 1>,
                    std::negation<std::conjunction<std::is_same<UTypes&&, Types const&>...>>,
                    std::conditional_t<
                            sizeof...(Types) == 1,
                            std::negation<std::is_same<
                                    std::remove_cvref_t<T0<UTypes...>>,
                                    tuple<Types...>>>,
                            std::true_type>,
                    std::conditional_t<
                            sizeof...(Types) == 2 || sizeof...(Types) == 3,
                            std::disjunction<
                                    std::negation<std::is_same<
                                            std::remove_cvref_t<T0<UTypes...>>,
                                            std::allocator_arg_t>>,
                                    std::is_same<
                                            std::remove_cvref_t<T0<Types...>>,
                                            std::allocator_arg_t>>,
                            std::true_type>,
                    std::negation<impl::reference_constructs_from_temporary<Types, UTypes&&>>...,
                    std::is_constructible<Types, UTypes>...>>>
    KOKKOS_INLINE_FUNCTION explicit(
            (!std::is_convertible_v<UTypes&&, Types> || ...)) constexpr tuple(UTypes&&... args)
        : m_values(FWD(args)...)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION constexpr tuple(tuple const& u) = default;

    KOKKOS_INLINE_FUNCTION constexpr tuple(tuple&& u) noexcept(
            (std::is_nothrow_move_assignable_v<Types> && ...))
        requires(std::move_constructible<Types> && ...)
        : m_values(std::move(u.m_values))
    {
    }

    template <class... UTypes, CONVERTING_TUPLE_CTOR_CONSTRAINTS(const, &)>
    KOKKOS_INLINE_FUNCTION explicit(
            !(impl::all_types_convertible_v<
                    tuple<UTypes...> const&,
                    tuple<Types...>>)) constexpr tuple(tuple<UTypes...> const& other)
        : tuple(converting_tag {}, FWD(other), std::make_index_sequence<sizeof...(Types)> {})
    {
    }

    template <class... UTypes, CONVERTING_TUPLE_CTOR_CONSTRAINTS(, &&)>
    KOKKOS_INLINE_FUNCTION explicit(!(impl::all_types_convertible_v<
                                      tuple<UTypes...>&&,
                                      tuple<Types...>>)) constexpr tuple(tuple<UTypes...>&& other)
        : tuple(converting_tag {}, std::move(other), std::make_index_sequence<sizeof...(Types)> {})
    {
    }

#if defined(CEXA_HAS_CXX23)
    template <class... UTypes, CONVERTING_TUPLE_CTOR_CONSTRAINTS(, &)>
    KOKKOS_INLINE_FUNCTION explicit(!(impl::all_types_convertible_v<
                                      tuple<UTypes...>&,
                                      tuple<Types...>>)) constexpr tuple(tuple<UTypes...>& other)
        : tuple(converting_tag {}, FWD(other), std::make_index_sequence<sizeof...(Types)> {})
    {
    }

    template <class... UTypes, CONVERTING_TUPLE_CTOR_CONSTRAINTS(const, &&)>
    KOKKOS_INLINE_FUNCTION explicit(
            !(impl::all_types_convertible_v<
                    tuple<UTypes...> const&&,
                    tuple<Types...>>)) constexpr tuple(tuple<UTypes...> const&& other)
        : tuple(converting_tag {}, std::move(other), std::make_index_sequence<sizeof...(Types)> {})
    {
    }
#endif
    // NOLINTEND(modernize-use-constraints)

    template <class U1, class U2>
        requires impl::pair_constructible<tuple, std::pair<U1, U2> const&>
    constexpr explicit(
            (!std::is_convertible_v<
                     decltype(std::get<0>(FWD((std::declval<std::pair<U1, U2> const&>())))),
                     T0<Types...>>
             || !std::is_convertible_v<
                     decltype(std::get<1>(FWD((std::declval<std::pair<U1, U2> const&>())))),
                     T1<Types...>>)) tuple(std::pair<U1, U2> const& u)
        : m_values(u.first, u.second)
    {
    }

    template <class U1, class U2>
        requires impl::pair_constructible<tuple, std::pair<U1, U2>&&>
    constexpr explicit(
            (!std::is_convertible_v<
                     decltype(std::get<0>(FWD((std::declval<std::pair<U1, U2>&&>())))),
                     T0<Types...>>
             || !std::is_convertible_v<
                     decltype(std::get<1>(FWD((std::declval<std::pair<U1, U2>&&>())))),
                     T1<Types...>>)) tuple(std::pair<U1, U2>&& u)
        : m_values(std::move(u.first), std::move(u.second))
    {
    } // TODO: see if we should use forward instead

#if defined(CEXA_HAS_CXX23)
    template <class U1, class U2>
        requires impl::pair_constructible<tuple, std::pair<U1, U2>&>
    constexpr explicit(
            (!std::is_convertible_v<
                     decltype(std::get<0>(FWD((std::declval<std::pair<U1, U2>&>())))),
                     T0<Types...>>
             || !std::is_convertible_v<
                     decltype(std::get<1>(FWD((std::declval<std::pair<U1, U2>&>())))),
                     T1<Types...>>)) tuple(std::pair<U1, U2>& u)
        : m_values(u.first, u.second)
    {
    }

    template <class U1, class U2>
        requires impl::pair_constructible<tuple, std::pair<U1, U2>&>
    constexpr explicit(
            (!std::is_convertible_v<
                     decltype(std::get<0>(FWD((std::declval<std::pair<U1, U2> const&&>())))),
                     T0<Types...>>
             || !std::is_convertible_v<
                     decltype(std::get<1>(FWD((std::declval<std::pair<U1, U2> const&&>())))),
                     T1<Types...>>)) tuple(std::pair<U1, U2> const&& u)
        : m_values(std::move(u.first), std::move(u.second))
    {
    }
#endif

    template <class UTuple>
        requires impl::tuple_like_constructible<tuple, UTuple>
    constexpr explicit((!impl::all_types_convertible_v<UTuple&&, tuple<Types...>>))
            tuple(UTuple&& u)
        : tuple(tuple_like_tag {}, FWD(u), std::make_index_sequence<sizeof...(Types)> {})
    {
    }

    KOKKOS_DEFAULTED_FUNCTION
    constexpr ~tuple() = default;

    // tuple.assign
    KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(tuple const& u)
        requires(std::is_copy_assignable_v<Types> && ...)
    = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(tuple&& u) noexcept(
            (std::is_nothrow_move_assignable_v<Types> && ...))
        requires(std::is_move_assignable_v<Types> && ...)
    = default;

#if defined(CEXA_HAS_CXX23)
    KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(tuple const& u) const
        requires(std::is_copy_assignable_v<Types const> && ...)
    {
        if (this != &u) {
            m_values = u.m_values;
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(tuple&& u) const
            noexcept((std::is_nothrow_move_assignable_v<Types const> && ...))
        requires(std::is_assignable_v<Types const&, Types> && ...)
    {
        m_values = std::move(u.m_values);
        return *this;
    }
#endif

    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
                && std::conjunction_v<std::is_assignable<Types&, UTypes const&>...>
    KOKKOS_INLINE_FUNCTION constexpr tuple& operator=(tuple<UTypes...> const& other) noexcept(
            (std::is_nothrow_assignable_v<Types&, UTypes const&> && ...))
    {
        m_values = other.m_values;
        return *this;
    }

    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
                && std::conjunction_v<std::is_assignable<Types&, UTypes>...>
    KOKKOS_INLINE_FUNCTION constexpr tuple& operator=(tuple<UTypes...>&& other) noexcept(
            (std::is_nothrow_assignable_v<Types&, UTypes> && ...))
    {
        m_values = std::move(other.m_values);
        return *this;
    }

#if defined(CEXA_HAS_CXX23)
    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
                && std::conjunction_v<std::is_assignable<Types const&, UTypes const&>...>
    KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(tuple<UTypes...> const& other) const
            noexcept((std::is_nothrow_assignable_v<Types const&, UTypes const&> && ...))
    {
        m_values = other.m_values;
        return *this;
    }

    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
                && std::conjunction_v<std::is_assignable<Types const&, UTypes>...>
    KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(tuple<UTypes...>&& other) const
            noexcept((std::is_nothrow_assignable_v<Types const&, UTypes> && ...))
    {
        m_values = std::move(other.m_values);
        return *this;
    }
#endif

    template <class U1, class U2>
        requires(sizeof...(Types) == 2)
                && std::conjunction_v<
                        std::is_assignable<T0<Types...>&, const U1&>,
                        std::is_assignable<T1<Types...>&, const U2&>>
    constexpr tuple& operator=(std::pair<U1, U2> const& p) noexcept(
            std::is_nothrow_assignable_v<T0<Types...>&, const U1&>
            && std::is_nothrow_assignable_v<T1<Types...>&, const U2&>)
    {
        m_values.value = p.first;
        m_values.rest.value = p.second;
        return *this;
    }

    template <class U1, class U2>
        requires(sizeof...(Types) == 2)
                && std::conjunction_v<
                        std::is_assignable<T0<Types...>&, U1>,
                        std::is_assignable<T1<Types...>&, U2>>
    // NOTE: we use forward in order to not move out of references contained
    // inside the pair
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    constexpr tuple& operator=(std::pair<U1, U2>&& p) noexcept(
            std::is_nothrow_assignable_v<T0<Types...>&, U1>
            && std::is_nothrow_assignable_v<T1<Types...>&, U2>)
    {
        m_values.value = FWD(p.first);
        m_values.rest.value = FWD(p.second);
        return *this;
    }

#if defined(CEXA_HAS_CXX23)
    template <class U1, class U2>
        requires(sizeof...(Types) == 2)
                && std::conjunction_v<
                        std::is_assignable<const T0<Types...>&, const U1&>,
                        std::is_assignable<const T1<Types...>&, const U2&>>
    // NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
    constexpr tuple const& operator=(std::pair<U1, U2> const& p) const noexcept(
            std::is_nothrow_assignable_v<const T0<Types...>&, const U1&>
            && std::is_nothrow_assignable_v<const T1<Types...>&, const U2&>)
    {
        m_values.value = p.first;
        m_values.rest.value = p.second;
        return *this;
    }

    template <class U1, class U2>
        requires(sizeof...(Types) == 2)
                && std::conjunction_v<
                        std::is_assignable<const T0<Types...>&, U1>,
                        std::is_assignable<const T1<Types...>&, U2>>
    // NOTE: we use forward in order to not move out of references contained
    // inside the pair
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved,misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
    constexpr tuple const& operator=(std::pair<U1, U2>&& p) const noexcept(
            std::is_nothrow_assignable_v<const T0<Types...>&, U1>
            && std::is_nothrow_assignable_v<const T1<Types...>&, U2>)
    {
        m_values.value = FWD(p.first);
        m_values.rest.value = FWD(p.second);
        return *this;
    }
#endif

#if defined(CEXA_HAS_CXX23)
    template <class UTuple>
        requires std::conjunction_v<
                impl::is_tuple_like<UTuple>,
                std::negation<impl::is_tuple<std::remove_cvref_t<UTuple>>>,
                std::negation<impl::is_pair<std::remove_cvref_t<UTuple>>>,
                impl::is_different_from<UTuple, tuple>,
                std::negation<impl::is_subrange<UTuple>>,
                std::bool_constant<
                        sizeof...(Types) == tuple_size<std::remove_reference_t<UTuple>>::value>>
    // The check for is_assignable is delegated to store.set_all()
    constexpr tuple& operator=(UTuple&& u)
    {
        m_values.set(FWD(u));
        return *this;
    }

    template <class UTuple>
        requires std::conjunction_v<
                impl::is_tuple_like<UTuple>,
                std::negation<impl::is_tuple<std::remove_cvref_t<UTuple>>>,
                std::negation<impl::is_pair<std::remove_cvref_t<UTuple>>>,
                impl::is_different_from<UTuple, tuple>,
                std::negation<impl::is_subrange<UTuple>>,
                std::bool_constant<
                        sizeof...(Types) == tuple_size<std::remove_reference_t<UTuple>>::value>>
    // The check for is_assignable is delegated to store.set_all()
    // NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
    constexpr tuple const& operator=(UTuple&& u) const
    {
        m_values.set(FWD(u));
        return *this;
    }
#endif
#undef FWD

    KOKKOS_INLINE_FUNCTION constexpr void swap(tuple& rhs) noexcept(
            (std::is_nothrow_swappable_v<Types> && ...))
    {
        return m_values.swap(rhs.m_values);
    }

#if defined(CEXA_HAS_CXX23)
    KOKKOS_INLINE_FUNCTION constexpr void swap(tuple const& rhs) const
            noexcept((std::is_nothrow_swappable_v<Types const> && ...))
    {
        return m_values.swap(rhs.m_values);
    }
#endif

    template <std::size_t I, class... Ts>
        requires(I < sizeof...(Ts))
    KOKKOS_INLINE_FUNCTION friend constexpr tuple_element_t<I, tuple<Ts...>>& get(
            tuple<Ts...>& t) noexcept;
    template <std::size_t I, class... Ts>
        requires(I < sizeof...(Ts))
    KOKKOS_INLINE_FUNCTION friend constexpr tuple_element_t<I, tuple<Ts...>>&& get(
            tuple<Ts...>&& t) noexcept;
    template <std::size_t I, class... Ts>
        requires(I < sizeof...(Ts))
    KOKKOS_INLINE_FUNCTION friend constexpr const tuple_element_t<I, tuple<Ts...>>& get(
            tuple<Ts...> const& t) noexcept;
    template <std::size_t I, class... Ts>
        requires(I < sizeof...(Ts))
    KOKKOS_INLINE_FUNCTION friend constexpr const tuple_element_t<I, tuple<Ts...>>&& get(
            tuple<Ts...> const&& t) noexcept;
    template <class T, class... Ts>
        requires(std::is_same_v<T, Ts> || ...)
    KOKKOS_INLINE_FUNCTION friend constexpr T& get(tuple<Ts...>& t) noexcept;
    template <class T, class... Ts>
        requires(std::is_same_v<T, Ts> || ...)
    KOKKOS_INLINE_FUNCTION friend constexpr T&& get(tuple<Ts...>&& t) noexcept;
    template <class T, class... Ts>
        requires(std::is_same_v<T, Ts> || ...)
    KOKKOS_INLINE_FUNCTION friend constexpr const T& get(tuple<Ts...> const& t) noexcept;
    template <class T, class... Ts>
        requires(std::is_same_v<T, Ts> || ...)
    KOKKOS_INLINE_FUNCTION friend constexpr const T&& get(tuple<Ts...> const&& t) noexcept;

    // tuple.rel
#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
    KOKKOS_INLINE_FUNCTION constexpr auto operator<=>(tuple<UTypes...> const& rhs) const
    {
        return m_values <=> rhs.m_values;
    }

    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
    KOKKOS_INLINE_FUNCTION constexpr bool operator==(tuple<UTypes...> const& rhs) const
    {
        return (m_values <=> rhs.m_values) == 0;
    }
#else
    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
    KOKKOS_INLINE_FUNCTION constexpr bool operator==(tuple<UTypes...> const& rhs) const
    {
        return m_values == rhs.m_values;
    }
    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
    KOKKOS_INLINE_FUNCTION constexpr bool operator!=(tuple<UTypes...> const& rhs) const
    {
        return !(m_values == rhs.m_values);
    }
    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
    KOKKOS_INLINE_FUNCTION constexpr bool operator<(tuple<UTypes...> const& rhs) const
    {
        return m_values < rhs.m_values;
    }
    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
    KOKKOS_INLINE_FUNCTION constexpr bool operator<=(tuple<UTypes...> const& rhs) const
    {
        return !(rhs.m_values < m_values);
    }
    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
    KOKKOS_INLINE_FUNCTION constexpr bool operator>(tuple<UTypes...> const& rhs) const
    {
        return rhs.m_values < m_values;
    }
    template <class... UTypes>
        requires(sizeof...(Types) == sizeof...(UTypes))
    KOKKOS_INLINE_FUNCTION constexpr bool operator>=(tuple<UTypes...> const& rhs) const
    {
        return !(m_values < rhs.m_values);
    }
#endif
};

// deduction guides
template <class... UTypes>
KOKKOS_DEDUCTION_GUIDE tuple(UTypes...) -> tuple<UTypes...>;
template <class T1, class T2>
tuple(std::pair<T1, T2>) -> tuple<T1, T2>;

// tuple.elem
template <std::size_t I, class... Types>
    requires(I < sizeof...(Types))
KOKKOS_INLINE_FUNCTION constexpr tuple_element_t<I, tuple<Types...>>& get(
        tuple<Types...>& t) noexcept
{
    return t.m_values.template get_value<I>();
}
template <std::size_t I, class... Types>
    requires(I < sizeof...(Types))
KOKKOS_INLINE_FUNCTION constexpr tuple_element_t<I, tuple<Types...>>&&
// NOTE: this doesn't work with std::move
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
get(tuple<Types...>&& t) noexcept
{
    return static_cast<tuple_element_t<I, tuple<Types...>>&&>(t.m_values.template get_value<I>());
}
template <std::size_t I, class... Types>
    requires(I < sizeof...(Types))
KOKKOS_INLINE_FUNCTION constexpr const tuple_element_t<I, tuple<Types...>>& get(
        tuple<Types...> const& t) noexcept
{
    return t.m_values.template get_value<I>();
}
template <std::size_t I, class... Types>
    requires(I < sizeof...(Types))
KOKKOS_INLINE_FUNCTION constexpr const tuple_element_t<I, tuple<Types...>>&& get(
        tuple<Types...> const&& t) noexcept
{
    return static_cast<tuple_element_t<I, tuple<Types...>> const&&>(
            t.m_values.template get_value<I>());
}
template <class T, class... Types>
    requires(std::is_same_v<T, Types> || ...)
KOKKOS_INLINE_FUNCTION constexpr T& get(tuple<Types...>& t) noexcept
{
    return t.m_values.template get_value<T>();
}
template <class T, class... Types>
    requires(std::is_same_v<T, Types> || ...)
KOKKOS_INLINE_FUNCTION constexpr T&&
// NOTE: this doesn't work with std::move
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
get(tuple<Types...>&& t) noexcept
{
    return static_cast<T&&>(t.m_values.template get_value<T>());
}
template <class T, class... Types>
    requires(std::is_same_v<T, Types> || ...)
KOKKOS_INLINE_FUNCTION constexpr const T& get(tuple<Types...> const& t) noexcept
{
    return t.m_values.template get_value<T>();
}
template <class T, class... Types>
    requires(std::is_same_v<T, Types> || ...)
KOKKOS_INLINE_FUNCTION constexpr const T&& get(tuple<Types...> const&& t) noexcept
{
    return static_cast<T const&&>(t.m_values.template get_value<T>());
}

template <class... Types>
    requires(std::is_swappable_v<Types> && ...)
KOKKOS_INLINE_FUNCTION constexpr void swap(tuple<Types...>& lhs, tuple<Types...>& rhs) noexcept(
        (std::is_nothrow_swappable_v<Types> && ...))
{
    lhs.swap(rhs);
}

#if defined(CEXA_HAS_CXX23)
template <class... Types>
    requires(std::is_swappable_v<Types const> && ...)
KOKKOS_INLINE_FUNCTION constexpr void swap(
        tuple<Types...> const& lhs,
        tuple<Types...> const& rhs) noexcept((std::is_nothrow_swappable_v<Types const> && ...))
{
    lhs.swap(rhs);
}
#endif

} // namespace cexa
// NOLINTEND(readability-identifier-naming)
