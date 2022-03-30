// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <ostream>
#include <utility>

#include "ddc/detail/tagged_vector.hpp"
#include "ddc/detail/type_seq.hpp"


template <class, class...>
class DiscreteElement;

template <class T>
struct IsDiscreteElement : std::false_type
{
};

template <class ElementType, class... Tags>
struct IsDiscreteElement<DiscreteElement<ElementType, Tags...>> : std::true_type
{
};

template <class T>
inline constexpr bool is_discrete_element_v = IsDiscreteElement<T>::value;


template <class QueryTag, class ElementType, class... Tags>
inline constexpr ElementType const& get(DiscreteElement<ElementType, Tags...> const& tuple) noexcept
{
    return tuple.template get<QueryTag>();
}

template <class QueryTag, class ElementType, class... Tags>
inline constexpr ElementType& get(DiscreteElement<ElementType, Tags...>& tuple) noexcept
{
    return tuple.template get<QueryTag>();
}

template <class QueryTag, class ElementType, class... Tags>
inline constexpr ElementType const& get_or(
        DiscreteElement<ElementType, Tags...> const& tuple,
        ElementType const& default_value) noexcept
{
    return tuple.template get_or<QueryTag>(default_value);
}

template <class... QueryTags, class ElementType, class... Tags>
inline constexpr DiscreteElement<ElementType, QueryTags...> select(
        DiscreteElement<ElementType, Tags...> const& arr) noexcept
{
    return DiscreteElement<ElementType, QueryTags...>(arr);
}

template <class... QueryTags, class ElementType, class... Tags>
inline constexpr DiscreteElement<ElementType, QueryTags...> select(
        DiscreteElement<ElementType, Tags...>&& arr) noexcept
{
    return DiscreteElement<ElementType, QueryTags...>(std::move(arr));
}

template <class QueryTag, class ElementType, class HeadTag, class... TailTags>
constexpr DiscreteElement<ElementType, QueryTag> const& take(
        DiscreteElement<ElementType, HeadTag> const& head,
        DiscreteElement<ElementType, TailTags> const&... tags)
{
    static_assert(
            !type_seq_contains_v<detail::TypeSeq<HeadTag>, detail::TypeSeq<TailTags...>>,
            "ERROR: tag redundant");
    if constexpr (std::is_same_v<QueryTag, HeadTag>) {
        return head;
    } else {
        static_assert(sizeof...(TailTags) > 0, "ERROR: tag not found");
        return take<QueryTag>(tags...);
    }
}


template <class ElementType, class... Tags>
class DiscreteElement
{
    static_assert(std::is_integral_v<ElementType>);
    using tags_seq = detail::TypeSeq<Tags...>;

private:
    std::array<ElementType, sizeof...(Tags)> m_values;

public:
    static constexpr std::size_t size() noexcept
    {
        return sizeof...(Tags);
    }

public:
    inline constexpr DiscreteElement() = default;

    inline constexpr DiscreteElement(DiscreteElement const&) = default;

    inline constexpr DiscreteElement(DiscreteElement&&) = default;

    template <class... OTags>
    explicit inline constexpr DiscreteElement(
            DiscreteElement<ElementType, OTags> const&... other) noexcept
        : m_values {take<Tags>(other...).uid()...}
    {
    }

    template <class OElementType, class... OTags>
    explicit inline constexpr DiscreteElement(
            DiscreteElement<OElementType, OTags...> const& other) noexcept
        : m_values {(static_cast<ElementType>(other.template get<Tags>()))...}
    {
    }

    template <
            class... Params,
            class = std::enable_if_t<(std::is_integral_v<Params> && ...)>,
            class = std::enable_if_t<(!is_discrete_element_v<Params> && ...)>,
            class = std::enable_if_t<sizeof...(Params) == sizeof...(Tags)>>
    explicit inline constexpr DiscreteElement(Params const&... params) noexcept
        : m_values {static_cast<ElementType>(params)...}
    {
    }

    constexpr inline DiscreteElement& operator=(DiscreteElement const& other) = default;

    constexpr inline DiscreteElement& operator=(DiscreteElement&& other) = default;

    template <class... OTags>
    constexpr inline DiscreteElement& operator=(
            DiscreteElement<ElementType, OTags...> const& other) noexcept
    {
        m_values = other.m_values;
        return *this;
    }

    template <class... OTags>
    constexpr inline DiscreteElement& operator=(
            DiscreteElement<ElementType, OTags...>&& other) noexcept
    {
        m_values = std::move(other.m_values);
        return *this;
    }

    /// Returns a reference to the underlying `std::array`
    constexpr inline std::array<ElementType, sizeof...(Tags)>& array() noexcept
    {
        return m_values;
    }

    /// Returns a const reference to the underlying `std::array`
    constexpr inline std::array<ElementType, sizeof...(Tags)> const& array() const noexcept
    {
        return m_values;
    }

    constexpr inline ElementType& operator[](size_t pos)
    {
        return m_values[pos];
    }

    constexpr inline ElementType const& operator[](size_t pos) const
    {
        return m_values[pos];
    }

    template <class QueryTag>
    inline constexpr ElementType& get() noexcept
    {
        using namespace detail;
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteElement");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    template <class QueryTag>
    inline constexpr ElementType const& get() const noexcept
    {
        using namespace detail;
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteElement");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    template <class QueryTag>
    ElementType const& get_or(ElementType const& default_value) const&
    {
        if constexpr (in_tags_v<QueryTag, tags_seq>) {
            return m_values[type_seq_rank_v<QueryTag, tags_seq>];
        } else {
            return default_value;
        }
    }

    template <std::size_t N = sizeof...(Tags)>
    constexpr inline std::enable_if_t<N == 1, ElementType&> uid() noexcept
    {
        return m_values[0];
    }

    template <std::size_t N = sizeof...(Tags)>
    constexpr inline std::enable_if_t<N == 1, ElementType const&> uid() const noexcept
    {
        return m_values[0];
    }
};

template <class ElementType>
std::ostream& operator<<(std::ostream& out, DiscreteElement<ElementType> const&)
{
    out << "()";
    return out;
}

template <class ElementType, class Head, class... Tags>
std::ostream& operator<<(std::ostream& out, DiscreteElement<ElementType, Head, Tags...> const& arr)
{
    out << "(";
    out << get<Head>(arr);
    ((out << ", " << get<Tags>(arr)), ...);
    out << ")";
    return out;
}


template <class ElementType, class... Tags, class... OTags>
constexpr inline bool operator==(
        DiscreteElement<ElementType, Tags...> const& lhs,
        DiscreteElement<ElementType, OTags...> const& rhs) noexcept
{
    return ((lhs.template get<Tags>() == rhs.template get<Tags>()) && ...);
}

template <class ElementType, class... Tags, class... OTags>
constexpr inline bool operator!=(
        DiscreteElement<ElementType, Tags...> const& lhs,
        DiscreteElement<ElementType, OTags...> const& rhs) noexcept
{
    return !(lhs == rhs);
}

template <class ElementType, class Tag>
constexpr inline bool operator<(
        DiscreteElement<ElementType, Tag> const& lhs,
        DiscreteElement<ElementType, Tag> const& rhs)
{
    return lhs.uid() < rhs.uid();
}

template <class ElementType, class Tag>
constexpr inline bool operator<=(
        DiscreteElement<ElementType, Tag> const& lhs,
        DiscreteElement<ElementType, Tag> const& rhs)
{
    return lhs.uid() <= rhs.uid();
}

template <class ElementType, class Tag>
constexpr inline bool operator>(
        DiscreteElement<ElementType, Tag> const& lhs,
        DiscreteElement<ElementType, Tag> const& rhs)
{
    return lhs.uid() > rhs.uid();
}

template <class ElementType, class Tag>
constexpr inline bool operator>=(
        DiscreteElement<ElementType, Tag> const& lhs,
        DiscreteElement<ElementType, Tag> const& rhs)
{
    return lhs.uid() >= rhs.uid();
}

template <class ElementType, class... Tags, class OElementType, class... OTags>
constexpr inline auto operator+(
        DiscreteElement<ElementType, Tags...> const& lhs,
        detail::TaggedVector<OElementType, OTags...> const& rhs)
{
    static_assert(type_seq_same_v<detail::TypeSeq<Tags...>, detail::TypeSeq<OTags...>>);
    using RElementType = decltype(std::declval<ElementType>() + std::declval<OElementType>());
    return DiscreteElement<RElementType, Tags...>((get<Tags>(lhs) + get<Tags>(rhs))...);
}

template <
        class ElementType,
        class Tag,
        class OElementType,
        class = std::enable_if_t<std::is_integral_v<OElementType>>>
constexpr inline auto operator+(
        DiscreteElement<ElementType, Tag> const& lhs,
        OElementType const& rhs)
{
    using RElementType = decltype(std::declval<ElementType>() + std::declval<OElementType>());
    return DiscreteElement<RElementType, Tag>(get<Tag>(lhs) + rhs);
}

template <class ElementType, class... Tags, class OElementType, class... OTags>
constexpr inline auto operator-(
        DiscreteElement<ElementType, Tags...> const& lhs,
        detail::TaggedVector<OElementType, OTags...> const& rhs)
{
    static_assert(type_seq_same_v<detail::TypeSeq<Tags...>, detail::TypeSeq<OTags...>>);
    using RElementType = decltype(std::declval<ElementType>() - std::declval<OElementType>());
    return DiscreteElement<RElementType, Tags...>((get<Tags>(lhs) - get<Tags>(rhs))...);
}

template <
        class ElementType,
        class Tag,
        class OElementType,
        class = std::enable_if_t<std::is_integral_v<OElementType>>>
constexpr inline auto operator-(
        DiscreteElement<ElementType, Tag> const& lhs,
        OElementType const& rhs)
{
    using RElementType = decltype(std::declval<ElementType>() + std::declval<OElementType>());
    return DiscreteElement<RElementType, Tag>(get<Tag>(lhs) - rhs);
}

template <class ElementType, class... Tags, class OElementType, class... OTags>
constexpr inline auto operator-(
        DiscreteElement<ElementType, Tags...> const& lhs,
        DiscreteElement<OElementType, OTags...> const& rhs)
{
    static_assert(type_seq_same_v<detail::TypeSeq<Tags...>, detail::TypeSeq<OTags...>>);
    using RElementType = decltype(std::declval<ElementType>() - std::declval<OElementType>());
    return detail::TaggedVector<RElementType, Tags...>((get<Tags>(lhs) - get<Tags>(rhs))...);
}
