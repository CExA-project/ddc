// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <utility>

#include <Kokkos_Macros.hpp>

#include "detail/macros.hpp"
#include "detail/type_seq.hpp"

#include "discrete_vector.hpp"

namespace ddc {

template <class...>
class DiscreteElement;

template <class T>
struct is_discrete_element : std::false_type
{
};

template <class... Tags>
struct is_discrete_element<DiscreteElement<Tags...>> : std::true_type
{
};

template <class T>
inline constexpr bool is_discrete_element_v = is_discrete_element<T>::value;


namespace detail {

template <class... Tags>
struct ToTypeSeq<DiscreteElement<Tags...>>
{
    using type = TypeSeq<Tags...>;
};

} // namespace detail

/** A DiscreteCoordElement is a scalar that identifies an element of the discrete dimension
 */
using DiscreteElementType = std::size_t;

template <class Tag>
KOKKOS_FUNCTION constexpr DiscreteElementType const& uid(DiscreteElement<Tag> const& tuple) noexcept
{
    return tuple.uid();
}

template <class Tag>
KOKKOS_FUNCTION constexpr DiscreteElementType& uid(DiscreteElement<Tag>& tuple) noexcept
{
    return tuple.uid();
}

template <class QueryTag, class... Tags>
KOKKOS_FUNCTION constexpr DiscreteElementType const& uid(
        DiscreteElement<Tags...> const& tuple) noexcept
{
    return tuple.template uid<QueryTag>();
}

template <class QueryTag, class... Tags>
KOKKOS_FUNCTION constexpr DiscreteElementType& uid(DiscreteElement<Tags...>& tuple) noexcept
{
    return tuple.template uid<QueryTag>();
}

template <class QueryTag, class... Tags>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryTag> select_or(
        DiscreteElement<Tags...> const& arr,
        DiscreteElement<QueryTag> const& default_value) noexcept
{
    if constexpr (in_tags_v<QueryTag, detail::TypeSeq<Tags...>>) {
        return DiscreteElement<QueryTag>(arr);
    } else {
        return default_value;
    }
}

template <class... QueryTags, class... Tags>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryTags...> select(
        DiscreteElement<Tags...> const& arr) noexcept
{
    return DiscreteElement<QueryTags...>(arr);
}

template <class... QueryTags, class... Tags>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryTags...> select(
        DiscreteElement<Tags...>&& arr) noexcept
{
    return DiscreteElement<QueryTags...>(std::move(arr));
}

/// Returns a reference towards the DiscreteElement that contains the QueryTag
template <
        class QueryTag,
        class HeadDElem,
        class... TailDElems,
        std::enable_if_t<
                is_discrete_element_v<HeadDElem> && (is_discrete_element_v<TailDElems> && ...),
                int>
        = 1>
KOKKOS_FUNCTION constexpr auto const& take(HeadDElem const& head, TailDElems const&... tail)
{
    DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
    if constexpr (type_seq_contains_v<detail::TypeSeq<QueryTag>, to_type_seq_t<HeadDElem>>) {
        static_assert(
                (!type_seq_contains_v<detail::TypeSeq<QueryTag>, to_type_seq_t<TailDElems>> && ...),
                "ERROR: tag redundant");
        return head;
    } else {
        static_assert(sizeof...(TailDElems) > 0, "ERROR: tag not found");
        return take<QueryTag>(tail...);
    }
    DDC_IF_NVCC_THEN_POP
}

namespace detail {

/// Returns a reference to the underlying `std::array`
template <class... Tags>
KOKKOS_FUNCTION constexpr std::array<DiscreteElementType, sizeof...(Tags)>& array(
        DiscreteElement<Tags...>& v) noexcept
{
    return v.m_values;
}

/// Returns a reference to the underlying `std::array`
template <class... Tags>
KOKKOS_FUNCTION constexpr std::array<DiscreteElementType, sizeof...(Tags)> const& array(
        DiscreteElement<Tags...> const& v) noexcept
{
    return v.m_values;
}

} // namespace detail

template <class DDim>
constexpr DiscreteElement<DDim> create_reference_discrete_element() noexcept
{
    return DiscreteElement<DDim>(0);
}

/** A DiscreteElement identifies an element of the discrete dimension
 *
 * Each one is tagged by its associated dimensions.
 */
template <class... Tags>
class DiscreteElement
{
    using tags_seq = detail::TypeSeq<Tags...>;

    friend KOKKOS_FUNCTION constexpr std::array<DiscreteElementType, sizeof...(Tags)>& detail::
            array<Tags...>(DiscreteElement<Tags...>& v) noexcept;

    friend KOKKOS_FUNCTION constexpr std::array<DiscreteElementType, sizeof...(Tags)> const&
    detail::array<Tags...>(DiscreteElement<Tags...> const& v) noexcept;

private:
    std::array<DiscreteElementType, sizeof...(Tags)> m_values;

public:
    using value_type = DiscreteElementType;

    static KOKKOS_FUNCTION constexpr std::size_t size() noexcept
    {
        return sizeof...(Tags);
    }

public:
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteElement() = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteElement(DiscreteElement const&) = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteElement(DiscreteElement&&) = default;

    template <class... DElems, class = std::enable_if_t<(is_discrete_element_v<DElems> && ...)>>
    KOKKOS_FUNCTION constexpr explicit DiscreteElement(DElems const&... delems) noexcept
        : m_values {take<Tags>(delems...).template uid<Tags>()...}
    {
    }

    template <
            class... Params,
            class = std::enable_if_t<(!is_discrete_element_v<Params> && ...)>,
            class = std::enable_if_t<(std::is_convertible_v<Params, DiscreteElementType> && ...)>,
            class = std::enable_if_t<sizeof...(Params) == sizeof...(Tags)>>
    KOKKOS_FUNCTION constexpr explicit DiscreteElement(Params const&... params) noexcept
        : m_values {static_cast<DiscreteElementType>(params)...}
    {
    }

    KOKKOS_DEFAULTED_FUNCTION ~DiscreteElement() = default;

    KOKKOS_DEFAULTED_FUNCTION DiscreteElement& operator=(DiscreteElement const& other) = default;

    KOKKOS_DEFAULTED_FUNCTION DiscreteElement& operator=(DiscreteElement&& other) = default;

    template <class QueryTag>
    KOKKOS_FUNCTION constexpr value_type& uid() noexcept
    {
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteElement");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    template <class QueryTag>
    KOKKOS_FUNCTION constexpr value_type const& uid() const noexcept
    {
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteElement");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    template <std::size_t N = sizeof...(Tags)>
    KOKKOS_FUNCTION constexpr std::enable_if_t<N == 1, value_type&> uid() noexcept
    {
        return m_values[0];
    }

    template <std::size_t N = sizeof...(Tags)>
    KOKKOS_FUNCTION constexpr std::enable_if_t<N == 1, value_type const&> uid() const noexcept
    {
        return m_values[0];
    }

    template <std::size_t N = sizeof...(Tags), class = std::enable_if_t<N == 1>>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator++()
    {
        ++m_values[0];
        return *this;
    }

    template <std::size_t N = sizeof...(Tags), class = std::enable_if_t<N == 1>>
    KOKKOS_FUNCTION constexpr DiscreteElement operator++(int)
    {
        DiscreteElement const tmp = *this;
        ++m_values[0];
        return tmp;
    }

    template <std::size_t N = sizeof...(Tags), class = std::enable_if_t<N == 1>>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator--()
    {
        --m_values[0];
        return *this;
    }

    template <std::size_t N = sizeof...(Tags), class = std::enable_if_t<N == 1>>
    KOKKOS_FUNCTION constexpr DiscreteElement operator--(int)
    {
        DiscreteElement const tmp = *this;
        --m_values[0];
        return tmp;
    }

    template <class... OTags>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator+=(DiscreteVector<OTags...> const& rhs)
    {
        static_assert(((type_seq_contains_v<detail::TypeSeq<OTags>, tags_seq>) && ...));
        ((m_values[type_seq_rank_v<OTags, tags_seq>] += rhs.template get<OTags>()), ...);
        return *this;
    }

    template <
            class IntegralType,
            std::size_t N = sizeof...(Tags),
            class = std::enable_if_t<N == 1>,
            class = std::enable_if_t<std::is_integral_v<IntegralType>>>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator+=(IntegralType const& rhs)
    {
        m_values[0] += rhs;
        return *this;
    }

    template <class... OTags>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator-=(DiscreteVector<OTags...> const& rhs)
    {
        static_assert(((type_seq_contains_v<detail::TypeSeq<OTags>, tags_seq>) && ...));
        ((m_values[type_seq_rank_v<OTags, tags_seq>] -= rhs.template get<OTags>()), ...);
        return *this;
    }

    template <
            class IntegralType,
            std::size_t N = sizeof...(Tags),
            class = std::enable_if_t<N == 1>,
            class = std::enable_if_t<std::is_integral_v<IntegralType>>>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator-=(IntegralType const& rhs)
    {
        m_values[0] -= rhs;
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& out, DiscreteElement<> const&)
{
    out << "()";
    return out;
}

template <class Head, class... Tags>
std::ostream& operator<<(std::ostream& out, DiscreteElement<Head, Tags...> const& arr)
{
    out << "(";
    out << uid<Head>(arr);
    ((out << ", " << uid<Tags>(arr)), ...);
    out << ")";
    return out;
}


template <class... Tags, class... OTags>
KOKKOS_FUNCTION constexpr bool operator==(
        DiscreteElement<Tags...> const& lhs,
        DiscreteElement<OTags...> const& rhs) noexcept
{
    return ((lhs.template uid<Tags>() == rhs.template uid<Tags>()) && ...);
}

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
// In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
template <class... Tags, class... OTags>
KOKKOS_FUNCTION constexpr bool operator!=(
        DiscreteElement<Tags...> const& lhs,
        DiscreteElement<OTags...> const& rhs) noexcept
{
    return !(lhs == rhs);
}
#endif

template <class Tag>
KOKKOS_FUNCTION constexpr bool operator<(
        DiscreteElement<Tag> const& lhs,
        DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() < rhs.uid();
}

template <class Tag>
KOKKOS_FUNCTION constexpr bool operator<=(
        DiscreteElement<Tag> const& lhs,
        DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() <= rhs.uid();
}

template <class Tag>
KOKKOS_FUNCTION constexpr bool operator>(
        DiscreteElement<Tag> const& lhs,
        DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() > rhs.uid();
}

template <class Tag>
KOKKOS_FUNCTION constexpr bool operator>=(
        DiscreteElement<Tag> const& lhs,
        DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() >= rhs.uid();
}

/// right external binary operators: +, -

template <class... Tags, class... OTags>
KOKKOS_FUNCTION constexpr DiscreteElement<Tags...> operator+(
        DiscreteElement<Tags...> const& lhs,
        DiscreteVector<OTags...> const& rhs)
{
    using detail::TypeSeq;
    static_assert(((type_seq_contains_v<TypeSeq<OTags>, TypeSeq<Tags...>>) && ...));
    DiscreteElement<Tags...> result(lhs);
    result += rhs;
    return result;
}

template <
        class Tag,
        class IntegralType,
        class = std::enable_if_t<std::is_integral_v<IntegralType>>,
        class = std::enable_if_t<!is_discrete_vector_v<IntegralType>>>
KOKKOS_FUNCTION constexpr DiscreteElement<Tag> operator+(
        DiscreteElement<Tag> const& lhs,
        IntegralType const& rhs)
{
    DiscreteElement<Tag> result(lhs);
    result += rhs;
    return result;
}

template <class... Tags, class... OTags>
KOKKOS_FUNCTION constexpr DiscreteElement<Tags...> operator-(
        DiscreteElement<Tags...> const& lhs,
        DiscreteVector<OTags...> const& rhs)
{
    using detail::TypeSeq;
    static_assert(((type_seq_contains_v<TypeSeq<OTags>, TypeSeq<Tags...>>) && ...));
    DiscreteElement<Tags...> result(lhs);
    result -= rhs;
    return result;
}

template <
        class Tag,
        class IntegralType,
        class = std::enable_if_t<std::is_integral_v<IntegralType>>,
        class = std::enable_if_t<!is_discrete_vector_v<IntegralType>>>
KOKKOS_FUNCTION constexpr DiscreteElement<Tag> operator-(
        DiscreteElement<Tag> const& lhs,
        IntegralType const& rhs)
{
    DiscreteElement<Tag> result(lhs);
    result -= rhs;
    return result;
}

/// binary operator: -

template <class... Tags, class... OTags>
KOKKOS_FUNCTION constexpr DiscreteVector<Tags...> operator-(
        DiscreteElement<Tags...> const& lhs,
        DiscreteElement<OTags...> const& rhs)
{
    static_assert(type_seq_same_v<detail::TypeSeq<Tags...>, detail::TypeSeq<OTags...>>);
    return DiscreteVector<Tags...>((uid<Tags>(lhs) - uid<Tags>(rhs))...);
}

} // namespace ddc
