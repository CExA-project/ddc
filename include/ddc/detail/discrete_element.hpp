// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <ostream>
#include <utility>

#include "ddc/detail/discrete_vector.hpp"
#include "ddc/detail/type_seq.hpp"


template <class...>
class DiscreteElement;

template <class T>
struct IsDiscreteElement : std::false_type
{
};

template <class... Tags>
struct IsDiscreteElement<DiscreteElement<Tags...>> : std::true_type
{
};

template <class T>
inline constexpr bool is_discrete_element_v = IsDiscreteElement<T>::value;


template <class Tag>
inline constexpr std::size_t const& uid(DiscreteElement<Tag> const& tuple) noexcept
{
    return tuple.uid();
}

template <class Tag>
inline constexpr std::size_t& uid(DiscreteElement<Tag>& tuple) noexcept
{
    return tuple.uid();
}

template <class QueryTag, class... Tags>
inline constexpr std::size_t const& uid(DiscreteElement<Tags...> const& tuple) noexcept
{
    return tuple.template uid<QueryTag>();
}

template <class QueryTag, class... Tags>
inline constexpr std::size_t& uid(DiscreteElement<Tags...>& tuple) noexcept
{
    return tuple.template uid<QueryTag>();
}

template <class QueryTag, class... Tags>
inline constexpr std::size_t const& uid_or(
        DiscreteElement<Tags...> const& tuple,
        std::size_t const& default_value) noexcept
{
    return tuple.template uid_or<QueryTag>(default_value);
}

template <class... QueryTags, class... Tags>
inline constexpr DiscreteElement<QueryTags...> select(DiscreteElement<Tags...> const& arr) noexcept
{
    return DiscreteElement<QueryTags...>(arr);
}

template <class... QueryTags, class... Tags>
inline constexpr DiscreteElement<QueryTags...> select(DiscreteElement<Tags...>&& arr) noexcept
{
    return DiscreteElement<QueryTags...>(std::move(arr));
}

template <class QueryTag, class HeadTag, class... TailTags>
constexpr DiscreteElement<QueryTag> const& take(
        DiscreteElement<HeadTag> const& head,
        DiscreteElement<TailTags> const&... tags)
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


template <class... Tags>
class DiscreteElement
{
    using tags_seq = detail::TypeSeq<Tags...>;

private:
    std::array<std::size_t, sizeof...(Tags)> m_values;

public:
    using value_type = std::size_t;

    static constexpr std::size_t size() noexcept
    {
        return sizeof...(Tags);
    }

public:
    inline constexpr DiscreteElement() = default;

    inline constexpr DiscreteElement(DiscreteElement const&) = default;

    inline constexpr DiscreteElement(DiscreteElement&&) = default;

    template <class... OTags>
    explicit inline constexpr DiscreteElement(DiscreteElement<OTags> const&... other) noexcept
        : m_values {take<Tags>(other...).uid()...}
    {
    }

    template <class... OTags>
    explicit inline constexpr DiscreteElement(DiscreteElement<OTags...> const& other) noexcept
        : m_values {other.template uid<Tags>()...}
    {
    }

    template <
            class... Params,
            class = std::enable_if_t<(std::is_integral_v<Params> && ...)>,
            class = std::enable_if_t<(!is_discrete_element_v<Params> && ...)>,
            class = std::enable_if_t<sizeof...(Params) == sizeof...(Tags)>>
    explicit inline constexpr DiscreteElement(Params const&... params) noexcept
        : m_values {static_cast<value_type>(params)...}
    {
    }

    constexpr inline DiscreteElement& operator=(DiscreteElement const& other) = default;

    constexpr inline DiscreteElement& operator=(DiscreteElement&& other) = default;

    template <class... OTags>
    constexpr inline DiscreteElement& operator=(DiscreteElement<OTags...> const& other) noexcept
    {
        m_values = other.m_values;
        return *this;
    }

    template <class... OTags>
    constexpr inline DiscreteElement& operator=(DiscreteElement<OTags...>&& other) noexcept
    {
        m_values = std::move(other.m_values);
        return *this;
    }

    /// Returns a reference to the underlying `std::array`
    constexpr inline std::array<value_type, sizeof...(Tags)>& array() noexcept
    {
        return m_values;
    }

    /// Returns a const reference to the underlying `std::array`
    constexpr inline std::array<value_type, sizeof...(Tags)> const& array() const noexcept
    {
        return m_values;
    }

    constexpr inline value_type& operator[](size_t pos)
    {
        return m_values[pos];
    }

    constexpr inline value_type const& operator[](size_t pos) const
    {
        return m_values[pos];
    }

    template <class QueryTag>
    value_type const& uid_or(value_type const& default_value) const&
    {
        if constexpr (in_tags_v<QueryTag, tags_seq>) {
            return m_values[type_seq_rank_v<QueryTag, tags_seq>];
        } else {
            return default_value;
        }
    }

    template <class QueryTag>
    inline constexpr value_type& uid() noexcept
    {
        using namespace detail;
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteElement");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    template <class QueryTag>
    inline constexpr value_type const& uid() const noexcept
    {
        using namespace detail;
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteElement");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    template <std::size_t N = sizeof...(Tags)>
    constexpr inline std::enable_if_t<N == 1, value_type&> uid() noexcept
    {
        return m_values[0];
    }

    template <std::size_t N = sizeof...(Tags)>
    constexpr inline std::enable_if_t<N == 1, value_type const&> uid() const noexcept
    {
        return m_values[0];
    }

    template <class... OTags>
    constexpr inline DiscreteElement& operator+=(DiscreteVector<OTags...> const& rhs)
    {
        static_assert(type_seq_same_v<tags_seq, detail::TypeSeq<OTags...>>);
        ((m_values[type_seq_rank_v<Tags, tags_seq>] += rhs.template get<Tags>()), ...);
        return *this;
    }

    template <class IntegralType, class = std::enable_if_t<std::is_integral_v<IntegralType>>>
    constexpr inline DiscreteElement& operator+=(IntegralType const& rhs)
    {
        ((m_values[type_seq_rank_v<Tags, tags_seq>] += rhs), ...);
        return *this;
    }

    template <class... OTags>
    constexpr inline DiscreteElement& operator-=(DiscreteVector<OTags...> const& rhs)
    {
        static_assert(type_seq_same_v<tags_seq, detail::TypeSeq<OTags...>>);
        ((m_values[type_seq_rank_v<Tags, tags_seq>] -= rhs.template get<Tags>()), ...);
        return *this;
    }

    template <class IntegralType, class = std::enable_if_t<std::is_integral_v<IntegralType>>>
    constexpr inline DiscreteElement& operator-=(IntegralType const& rhs)
    {
        ((m_values[type_seq_rank_v<Tags, tags_seq>] -= rhs), ...);
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
constexpr inline bool operator==(
        DiscreteElement<Tags...> const& lhs,
        DiscreteElement<OTags...> const& rhs) noexcept
{
    return ((lhs.template uid<Tags>() == rhs.template uid<Tags>()) && ...);
}

template <class... Tags, class... OTags>
constexpr inline bool operator!=(
        DiscreteElement<Tags...> const& lhs,
        DiscreteElement<OTags...> const& rhs) noexcept
{
    return !(lhs == rhs);
}

template <class Tag>
constexpr inline bool operator<(DiscreteElement<Tag> const& lhs, DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() < rhs.uid();
}

template <class Tag>
constexpr inline bool operator<=(DiscreteElement<Tag> const& lhs, DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() <= rhs.uid();
}

template <class Tag>
constexpr inline bool operator>(DiscreteElement<Tag> const& lhs, DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() > rhs.uid();
}

template <class Tag>
constexpr inline bool operator>=(DiscreteElement<Tag> const& lhs, DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() >= rhs.uid();
}

/// right external binary operators: +, -

template <class... Tags, class... OTags>
constexpr inline auto operator+(
        DiscreteElement<Tags...> const& lhs,
        DiscreteVector<OTags...> const& rhs)
{
    static_assert(type_seq_same_v<detail::TypeSeq<Tags...>, detail::TypeSeq<OTags...>>);
    return DiscreteElement<Tags...>((uid<Tags>(lhs) + get<Tags>(rhs))...);
}

template <
        class Tag,
        class IntegralType,
        class = std::enable_if_t<std::is_integral_v<IntegralType>>,
        class = std::enable_if_t<!is_discrete_vector_v<IntegralType>>>
constexpr inline auto operator+(DiscreteElement<Tag> const& lhs, IntegralType const& rhs)
{
    return DiscreteElement<Tag>(uid<Tag>(lhs) + rhs);
}

template <class... Tags, class... OTags>
constexpr inline auto operator-(
        DiscreteElement<Tags...> const& lhs,
        DiscreteVector<OTags...> const& rhs)
{
    static_assert(type_seq_same_v<detail::TypeSeq<Tags...>, detail::TypeSeq<OTags...>>);
    return DiscreteElement<Tags...>((uid<Tags>(lhs) - get<Tags>(rhs))...);
}

template <
        class Tag,
        class IntegralType,
        class = std::enable_if_t<std::is_integral_v<IntegralType>>,
        class = std::enable_if_t<!is_discrete_vector_v<IntegralType>>>
constexpr inline auto operator-(DiscreteElement<Tag> const& lhs, IntegralType const& rhs)
{
    return DiscreteElement<Tag>(uid<Tag>(lhs) - rhs);
}

/// binary operator: -

template <class... Tags, class... OTags>
constexpr inline auto operator-(
        DiscreteElement<Tags...> const& lhs,
        DiscreteElement<OTags...> const& rhs)
{
    static_assert(type_seq_same_v<detail::TypeSeq<Tags...>, detail::TypeSeq<OTags...>>);
    return DiscreteVector<Tags...>((uid<Tags>(lhs) - uid<Tags>(rhs))...);
}
