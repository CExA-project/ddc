// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>

#include <Kokkos_Macros.hpp>

#include "ddc/detail/type_seq.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_vector.hpp"

namespace ddc {

template <class DDim>
struct DiscreteDomainIterator;

template <class... DDims>
class DiscreteDomain;

template <class T>
struct is_discrete_domain : std::false_type
{
};

template <class... Tags>
struct is_discrete_domain<DiscreteDomain<Tags...>> : std::true_type
{
};

template <class T>
inline constexpr bool is_discrete_domain_v = is_discrete_domain<T>::value;


namespace detail {

template <class... Tags>
struct ToTypeSeq<DiscreteDomain<Tags...>>
{
    using type = TypeSeq<Tags...>;
};

template <class T, class U>
struct RebindDomain;

template <class... DDims, class... ODDims>
struct RebindDomain<DiscreteDomain<DDims...>, detail::TypeSeq<ODDims...>>
{
    using type = DiscreteDomain<ODDims...>;
};

} // namespace detail

template <class... DDims>
class DiscreteDomain
{
    template <class...>
    friend class DiscreteDomain;

public:
    using discrete_element_type = DiscreteElement<DDims...>;

    using discrete_vector_type = DiscreteVector<DDims...>;

private:
    DiscreteElement<DDims...> m_element_begin;

    DiscreteElement<DDims...> m_element_end;

public:
    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return sizeof...(DDims);
    }

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain() = default;

    /// Construct a DiscreteDomain by copies and merge of domains
    template <class... DDoms, class = std::enable_if_t<(is_discrete_domain_v<DDoms> && ...)>>
    KOKKOS_FUNCTION constexpr explicit DiscreteDomain(DDoms const&... domains)
        : m_element_begin(domains.front()...)
        , m_element_end((domains.front() + domains.extents())...)
    {
    }

    /** Construct a DiscreteDomain starting from element_begin with size points.
     * @param element_begin the lower bound in each direction
     * @param size the number of points in each direction
     */
    KOKKOS_FUNCTION constexpr DiscreteDomain(
            discrete_element_type const& element_begin,
            discrete_vector_type const& size)
        : m_element_begin(element_begin)
        , m_element_end(element_begin + size)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain(DiscreteDomain const& x) = default;

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain(DiscreteDomain&& x) = default;

    KOKKOS_DEFAULTED_FUNCTION ~DiscreteDomain() = default;

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain& operator=(DiscreteDomain const& x) = default;

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain& operator=(DiscreteDomain&& x) = default;

    template <class... ODims>
    KOKKOS_FUNCTION constexpr bool operator==(DiscreteDomain<ODims...> const& other) const
    {
        if (empty() && other.empty()) {
            return true;
        }
        return m_element_begin == other.m_element_begin && m_element_end == other.m_element_end;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    template <class... ODims>
    KOKKOS_FUNCTION constexpr bool operator!=(DiscreteDomain<ODims...> const& other) const
    {
        return !(*this == other);
    }
#endif

    KOKKOS_FUNCTION constexpr std::size_t size() const
    {
        return (1UL * ... * extent<DDims>().value());
    }

    KOKKOS_FUNCTION constexpr discrete_vector_type extents() const noexcept
    {
        return m_element_end - m_element_begin;
    }

    template <class QueryDDim>
    KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDim> extent() const noexcept
    {
        return DiscreteElement<QueryDDim>(m_element_end)
               - DiscreteElement<QueryDDim>(m_element_begin);
    }

    KOKKOS_FUNCTION constexpr discrete_element_type front() const noexcept
    {
        return m_element_begin;
    }

    KOKKOS_FUNCTION constexpr discrete_element_type back() const noexcept
    {
        return discrete_element_type((DiscreteElement<DDims>(m_element_end) - 1)...);
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain take_first(discrete_vector_type n) const
    {
        return DiscreteDomain(front(), n);
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain take_last(discrete_vector_type n) const
    {
        return DiscreteDomain(front() + (extents() - n), n);
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain remove_first(discrete_vector_type n) const
    {
        return DiscreteDomain(front() + n, extents() - n);
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain remove_last(discrete_vector_type n) const
    {
        return DiscreteDomain(front(), extents() - n);
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain remove(
            discrete_vector_type n1,
            discrete_vector_type n2) const
    {
        return DiscreteDomain(front() + n1, extents() - n1 - n2);
    }

    KOKKOS_FUNCTION constexpr DiscreteElement<DDims...> operator()(
            DiscreteVector<DDims...> const& dvect) const noexcept
    {
        return m_element_begin + dvect;
    }

    template <class... ODDims>
    KOKKOS_FUNCTION constexpr auto restrict_with(DiscreteDomain<ODDims...> const& odomain) const
    {
        assert(((DiscreteElement<ODDims>(m_element_begin)
                 <= DiscreteElement<ODDims>(odomain.m_element_begin))
                && ...));
        assert(((DiscreteElement<ODDims>(m_element_end)
                 >= DiscreteElement<ODDims>(odomain.m_element_end))
                && ...));
        const DiscreteVector<DDims...> myextents = extents();
        const DiscreteVector<ODDims...> oextents = odomain.extents();
        return DiscreteDomain(
                DiscreteElement<DDims...>((select_or<DDims>(
                        odomain.m_element_begin,
                        DiscreteElement<DDims>(m_element_begin)))...),
                DiscreteVector<DDims...>(
                        (select_or<DDims>(oextents, DiscreteVector<DDims>(myextents)))...));
    }

    template <class... DElems>
    KOKKOS_FUNCTION bool contains(DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        return (((DiscreteElement<DDims>(take<DDims>(delems...))
                  >= DiscreteElement<DDims>(m_element_begin))
                 && ...)
                && ((DiscreteElement<DDims>(take<DDims>(delems...))
                     < DiscreteElement<DDims>(m_element_end))
                    && ...));
    }

    template <class... DElems>
    KOKKOS_FUNCTION DiscreteVector<DDims...> distance_from_front(
            DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        return DiscreteVector<DDims...>(
                (DiscreteElement<DDims>(take<DDims>(delems...))
                 - DiscreteElement<DDims>(m_element_begin))...);
    }

    KOKKOS_FUNCTION constexpr bool empty() const noexcept
    {
        return size() == 0;
    }

    KOKKOS_FUNCTION constexpr explicit operator bool()
    {
        return !empty();
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto begin() const
    {
        return DiscreteDomainIterator<DDim0>(front());
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto end() const
    {
        return DiscreteDomainIterator<DDim0>(m_element_end);
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto cbegin() const
    {
        return DiscreteDomainIterator<DDim0>(front());
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto cend() const
    {
        return DiscreteDomainIterator<DDim0>(m_element_end);
    }

    template <
            std::size_t N = sizeof...(DDims),
            class = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](std::size_t n)
    {
        return begin()[n];
    }

    template <
            std::size_t N = sizeof...(DDims),
            class = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](std::size_t n) const
    {
        return begin()[n];
    }
};

template <>
class DiscreteDomain<>
{
    template <class...>
    friend class DiscreteDomain;

public:
    using discrete_element_type = DiscreteElement<>;

    using discrete_vector_type = DiscreteVector<>;

    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return 0;
    }

    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteDomain() = default;

    /// Construct a DiscreteDomain by copies and merge of domains
    template <class... DDoms, class = std::enable_if_t<(is_discrete_domain_v<DDoms> && ...)>>
    KOKKOS_FUNCTION constexpr explicit DiscreteDomain([[maybe_unused]] DDoms const&... domains)
    {
    }

    /** Construct a DiscreteDomain starting from element_begin with size points.
     * @param element_begin the lower bound in each direction
     * @param size the number of points in each direction
     */
    KOKKOS_FUNCTION constexpr DiscreteDomain(
            [[maybe_unused]] discrete_element_type const& element_begin,
            [[maybe_unused]] discrete_vector_type const& size)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain(DiscreteDomain const& x) = default;

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain(DiscreteDomain&& x) = default;

    KOKKOS_DEFAULTED_FUNCTION ~DiscreteDomain() = default;

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain& operator=(DiscreteDomain const& x) = default;

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain& operator=(DiscreteDomain&& x) = default;

    KOKKOS_FUNCTION constexpr bool operator==([[maybe_unused]] DiscreteDomain const& other) const
    {
        return true;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    KOKKOS_FUNCTION constexpr bool operator!=(DiscreteDomain const& other) const
    {
        return !(*this == other);
    }
#endif

    static KOKKOS_FUNCTION constexpr std::size_t size()
    {
        return 1;
    }

    static KOKKOS_FUNCTION constexpr discrete_vector_type extents() noexcept
    {
        return {};
    }

    static KOKKOS_FUNCTION constexpr discrete_element_type front() noexcept
    {
        return {};
    }

    static KOKKOS_FUNCTION constexpr discrete_element_type back() noexcept
    {
        return {};
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain take_first(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain take_last(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain remove_first(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain remove_last(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomain remove(
            [[maybe_unused]] discrete_vector_type n1,
            [[maybe_unused]] discrete_vector_type n2) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteElement<> operator()(
            DiscreteVector<> const& /* dvect */) const noexcept
    {
        return {};
    }

    template <class... ODims>
    KOKKOS_FUNCTION constexpr DiscreteDomain restrict_with(
            DiscreteDomain<ODims...> const& /* odomain */) const
    {
        return *this;
    }

    static KOKKOS_FUNCTION bool contains() noexcept
    {
        return true;
    }

    static KOKKOS_FUNCTION bool contains(DiscreteElement<>) noexcept
    {
        return true;
    }

    static KOKKOS_FUNCTION DiscreteVector<> distance_from_front() noexcept
    {
        return {};
    }

    static KOKKOS_FUNCTION DiscreteVector<> distance_from_front(DiscreteElement<>) noexcept
    {
        return {};
    }

    static KOKKOS_FUNCTION constexpr bool empty() noexcept
    {
        return false;
    }

    KOKKOS_FUNCTION constexpr explicit operator bool()
    {
        return true;
    }
};

template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteDomain<QueryDDims...> select(
        DiscreteDomain<DDims...> const& domain)
{
    return DiscreteDomain<QueryDDims...>(
            DiscreteElement<QueryDDims...>(domain.front()),
            DiscreteVector<QueryDDims...>(domain.extents()));
}

namespace detail {

template <class T>
struct ConvertTypeSeqToDiscreteDomain;

template <class... DDims>
struct ConvertTypeSeqToDiscreteDomain<detail::TypeSeq<DDims...>>
{
    using type = DiscreteDomain<DDims...>;
};

template <class T>
using convert_type_seq_to_discrete_domain_t = typename ConvertTypeSeqToDiscreteDomain<T>::type;

} // namespace detail

// Computes the cartesian product of DiscreteDomain types
// Example usage : "using DDom = cartesian_prod_t<DDom1,DDom2,DDom3>;"
template <typename... DDoms>
struct cartesian_prod;

template <typename... DDims1, typename... DDims2, typename... DDomsTail>
struct cartesian_prod<DiscreteDomain<DDims1...>, DiscreteDomain<DDims2...>, DDomsTail...>
{
    using type = typename cartesian_prod<DiscreteDomain<DDims1..., DDims2...>, DDomsTail...>::type;
};

template <typename... DDims>
struct cartesian_prod<DiscreteDomain<DDims...>>
{
    using type = DiscreteDomain<DDims...>;
};

template <>
struct cartesian_prod<>
{
    using type = ddc::DiscreteDomain<>;
};

template <typename... DDoms>
using cartesian_prod_t = typename cartesian_prod<DDoms...>::type;

// Computes the subtraction DDom_a - DDom_b in the sense of linear spaces(retained dimensions are those in DDom_a which are not in DDom_b)
template <class... DDimsA, class... DDimsB>
KOKKOS_FUNCTION constexpr auto remove_dims_of(
        DiscreteDomain<DDimsA...> const& DDom_a,
        [[maybe_unused]] DiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    using TagSeqA = detail::TypeSeq<DDimsA...>;
    using TagSeqB = detail::TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_discrete_domain_t<type_seq_r>(DDom_a);
}

//! Remove the dimensions DDimsB from DDom_a
//! @param[in] DDom_a The discrete domain on which to remove dimensions
//! @return The discrete domain without DDimsB dimensions
template <class... DDimsB, class... DDimsA>
KOKKOS_FUNCTION constexpr auto remove_dims_of(DiscreteDomain<DDimsA...> const& DDom_a) noexcept
{
    using TagSeqA = detail::TypeSeq<DDimsA...>;
    using TagSeqB = detail::TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_discrete_domain_t<type_seq_r>(DDom_a);
}

// Remove dimensions from a domain type
template <typename DDom, typename... DDims>
using remove_dims_of_t = decltype(remove_dims_of<DDims...>(std::declval<DDom>()));

namespace detail {

// Checks if dimension of DDom_a is DDim1. If not, returns restriction to DDim2 of DDom_b. May not be useful in its own, it helps for replace_dim_of
template <typename DDim1, typename DDim2, typename DDimA, typename... DDimsB>
KOKKOS_FUNCTION constexpr std::conditional_t<
        std::is_same_v<DDimA, DDim1>,
        ddc::DiscreteDomain<DDim2>,
        ddc::DiscreteDomain<DDimA>>
replace_dim_of_1d(
        DiscreteDomain<DDimA> const& DDom_a,
        [[maybe_unused]] DiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    if constexpr (std::is_same_v<DDimA, DDim1>) {
        return ddc::DiscreteDomain<DDim2>(DDom_b);
    } else {
        return DDom_a;
    }
}

} // namespace detail

// Replace in DDom_a the dimension Dim1 by the dimension Dim2 of DDom_b
template <typename DDim1, typename DDim2, typename... DDimsA, typename... DDimsB>
KOKKOS_FUNCTION constexpr auto replace_dim_of(
        DiscreteDomain<DDimsA...> const& DDom_a,
        [[maybe_unused]] DiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    // TODO : static_asserts
    using TagSeqA = detail::TypeSeq<DDimsA...>;
    using TagSeqB = detail::TypeSeq<DDim1>;
    using TagSeqC = detail::TypeSeq<DDim2>;

    using type_seq_r = ddc::type_seq_replace_t<TagSeqA, TagSeqB, TagSeqC>;
    return ddc::detail::convert_type_seq_to_discrete_domain_t<type_seq_r>(
            detail::replace_dim_of_1d<
                    DDim1,
                    DDim2,
                    DDimsA,
                    DDimsB...>(ddc::DiscreteDomain<DDimsA>(DDom_a), DDom_b)...);
}

// Replace dimensions from a domain type
template <typename DDom, typename DDim1, typename DDim2>
using replace_dim_of_t = decltype(replace_dim_of<DDim1, DDim2>(
        std::declval<DDom>(),
        std::declval<typename detail::RebindDomain<DDom, detail::TypeSeq<DDim2>>::type>()));


template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDims...> extents(
        DiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteVector<QueryDDims...>(DiscreteDomain<QueryDDims>(domain).size()...);
}

template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> front(
        DiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(DiscreteDomain<QueryDDims>(domain).front()...);
}

template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> back(
        DiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(DiscreteDomain<QueryDDims>(domain).back()...);
}

template <class DDim>
struct DiscreteDomainIterator
{
private:
    DiscreteElement<DDim> m_value = DiscreteElement<DDim>();

public:
    using iterator_category = std::random_access_iterator_tag;

    using value_type = DiscreteElement<DDim>;

    using difference_type = std::ptrdiff_t;

    KOKKOS_DEFAULTED_FUNCTION DiscreteDomainIterator() = default;

    KOKKOS_FUNCTION constexpr explicit DiscreteDomainIterator(DiscreteElement<DDim> value)
        : m_value(value)
    {
    }

    KOKKOS_FUNCTION constexpr DiscreteElement<DDim> operator*() const noexcept
    {
        return m_value;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomainIterator& operator++()
    {
        ++m_value;
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomainIterator& operator--()
    {
        --m_value;
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator--(int)
    {
        auto tmp = *this;
        --*this;
        return tmp;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomainIterator& operator+=(difference_type n)
    {
        if (n >= difference_type(0)) {
            m_value += static_cast<DiscreteElementType>(n);
        } else {
            m_value -= static_cast<DiscreteElementType>(-n);
        }
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteDomainIterator& operator-=(difference_type n)
    {
        if (n >= difference_type(0)) {
            m_value -= static_cast<DiscreteElementType>(n);
        } else {
            m_value += static_cast<DiscreteElementType>(-n);
        }
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteElement<DDim> operator[](difference_type n) const
    {
        return m_value + n;
    }

    friend KOKKOS_FUNCTION constexpr bool operator==(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return xx.m_value == yy.m_value;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    friend KOKKOS_FUNCTION constexpr bool operator!=(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return xx.m_value != yy.m_value;
    }
#endif

    friend KOKKOS_FUNCTION constexpr bool operator<(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return xx.m_value < yy.m_value;
    }

    friend KOKKOS_FUNCTION constexpr bool operator>(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return yy < xx;
    }

    friend KOKKOS_FUNCTION constexpr bool operator<=(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return !(yy < xx);
    }

    friend KOKKOS_FUNCTION constexpr bool operator>=(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return !(xx < yy);
    }

    friend KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator+(
            DiscreteDomainIterator i,
            difference_type n)
    {
        return i += n;
    }

    friend KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator+(
            difference_type n,
            DiscreteDomainIterator i)
    {
        return i += n;
    }

    friend KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator-(
            DiscreteDomainIterator i,
            difference_type n)
    {
        return i -= n;
    }

    friend KOKKOS_FUNCTION constexpr difference_type operator-(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return (yy.m_value > xx.m_value) ? (-static_cast<difference_type>(yy.m_value - xx.m_value))
                                         : (xx.m_value - yy.m_value);
    }
};

} // namespace ddc
