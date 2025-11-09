// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>

#include <Kokkos_Macros.hpp>

#include "detail/type_seq.hpp"

#include "discrete_element.hpp"
#include "discrete_vector.hpp"

namespace ddc {

template <class DDim>
struct StridedDiscreteDomainIterator;

template <class... DDims>
class StridedDiscreteDomain;

template <class T>
struct is_strided_discrete_domain : std::false_type
{
};

template <class... Tags>
struct is_strided_discrete_domain<StridedDiscreteDomain<Tags...>> : std::true_type
{
};

template <class T>
inline constexpr bool is_strided_discrete_domain_v = is_strided_discrete_domain<T>::value;


namespace detail {

template <class... Tags>
struct ToTypeSeq<StridedDiscreteDomain<Tags...>>
{
    using type = TypeSeq<Tags...>;
};

template <class T, class U>
struct RebindDomain;

template <class... DDims, class... ODDims>
struct RebindDomain<StridedDiscreteDomain<DDims...>, detail::TypeSeq<ODDims...>>
{
    using type = StridedDiscreteDomain<ODDims...>;
};

} // namespace detail

template <class... ODDims>
KOKKOS_FUNCTION DiscreteVector<ODDims...> prod(
        DiscreteVector<ODDims...> const& lhs,
        DiscreteVector<ODDims...> const& rhs) noexcept
{
    return DiscreteVector<ODDims...>((get<ODDims>(lhs) * get<ODDims>(rhs))...);
}

template <class... DDims>
class StridedDiscreteDomain
{
    template <class...>
    friend class StridedDiscreteDomain;

public:
    using discrete_element_type = DiscreteElement<DDims...>;

    using discrete_vector_type = DiscreteVector<DDims...>;

private:
    DiscreteElement<DDims...> m_element_begin;

    DiscreteVector<DDims...> m_extents;

    DiscreteVector<DDims...> m_strides;

public:
    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return sizeof...(DDims);
    }

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain() = default;

    /// Construct a StridedDiscreteDomain by copies and merge of domains
    template <
            class... DDoms,
            class = std::enable_if_t<(is_strided_discrete_domain_v<DDoms> && ...)>>
    KOKKOS_FUNCTION constexpr explicit StridedDiscreteDomain(DDoms const&... domains)
        : m_element_begin(domains.front()...)
        , m_extents(domains.extents()...)
        , m_strides(domains.strides()...)
    {
    }

    /** Construct a StridedDiscreteDomain starting from element_begin with size points.
     * @param element_begin the lower bound in each direction
     * @param extents the number of points in each direction
     * @param strides the step between two elements
     */
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain(
            discrete_element_type const& element_begin,
            discrete_vector_type const& extents,
            discrete_vector_type const& strides)
        : m_element_begin(element_begin)
        , m_extents(extents)
        , m_strides(strides)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain(StridedDiscreteDomain const& x) = default;

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain(StridedDiscreteDomain&& x) = default;

    KOKKOS_DEFAULTED_FUNCTION ~StridedDiscreteDomain() = default;

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain& operator=(StridedDiscreteDomain const& x)
            = default;

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain& operator=(StridedDiscreteDomain&& x) = default;

    template <class... ODims>
    KOKKOS_FUNCTION constexpr bool operator==(StridedDiscreteDomain<ODims...> const& other) const
    {
        if (empty() && other.empty()) {
            return true;
        }
        return m_element_begin == other.m_element_begin && m_extents == other.m_extents
               && m_strides == other.m_strides;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    template <class... ODims>
    KOKKOS_FUNCTION constexpr bool operator!=(StridedDiscreteDomain<ODims...> const& other) const
    {
        return !(*this == other);
    }
#endif

    KOKKOS_FUNCTION constexpr std::size_t size() const
    {
        return (1UL * ... * get<DDims>(m_extents));
    }

    KOKKOS_FUNCTION constexpr discrete_vector_type extents() const noexcept
    {
        return m_extents;
    }

    KOKKOS_FUNCTION constexpr discrete_vector_type strides() const noexcept
    {
        return m_strides;
    }

    template <class QueryDDim>
    KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDim> extent() const noexcept
    {
        return DiscreteVector<QueryDDim>(m_extents);
    }

    KOKKOS_FUNCTION constexpr discrete_element_type front() const noexcept
    {
        return m_element_begin;
    }

    KOKKOS_FUNCTION constexpr discrete_element_type back() const noexcept
    {
        return discrete_element_type(
                (DiscreteElement<DDims>(m_element_begin)
                 + (get<DDims>(m_extents) - 1) * get<DDims>(m_strides))...);
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain take_first(discrete_vector_type n) const
    {
        return StridedDiscreteDomain(front(), n, m_strides);
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain take_last(discrete_vector_type n) const
    {
        return StridedDiscreteDomain(front() + prod(extents() - n, m_strides), n, m_strides);
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove_first(discrete_vector_type n) const
    {
        return StridedDiscreteDomain(front() + prod(n, m_strides), extents() - n, m_strides);
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove_last(discrete_vector_type n) const
    {
        return StridedDiscreteDomain(front(), extents() - n, m_strides);
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove(
            discrete_vector_type n1,
            discrete_vector_type n2) const
    {
        return StridedDiscreteDomain(front() + prod(n1, m_strides), extents() - n1 - n2, m_strides);
    }

    KOKKOS_FUNCTION constexpr DiscreteElement<DDims...> operator()(
            DiscreteVector<DDims...> const& dvect) const noexcept
    {
        return m_element_begin + prod(dvect, m_strides);
    }

    template <class... DElems>
    KOKKOS_FUNCTION bool contains(DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        auto const test1
                = ((DiscreteElement<DDims>(take<DDims>(delems...))
                    >= DiscreteElement<DDims>(m_element_begin))
                   && ...);
        auto const test2
                = ((DiscreteElement<DDims>(take<DDims>(delems...))
                    < (DiscreteElement<DDims>(m_element_begin)
                       + DiscreteVector<DDims>(m_extents) * DiscreteVector<DDims>(m_strides)))
                   && ...);
        auto const test3
                = ((((DiscreteElement<DDims>(take<DDims>(delems...))
                      - DiscreteElement<DDims>(m_element_begin))
                     % DiscreteVector<DDims>(m_strides))
                    == 0)
                   && ...);
        return test1 && test2 && test3;
    }

    template <class... DElems>
    KOKKOS_FUNCTION DiscreteVector<DDims...> distance_from_front(
            DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        KOKKOS_ASSERT(contains(delems...));
        return DiscreteVector<DDims...>(
                ((DiscreteElement<DDims>(take<DDims>(delems...))
                  - DiscreteElement<DDims>(m_element_begin))
                 / DiscreteVector<DDims>(m_strides))...);
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
        return StridedDiscreteDomainIterator<DDim0>(front(), m_strides);
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto end() const
    {
        return StridedDiscreteDomainIterator<
                DDim0>(m_element_begin + m_extents * m_strides, m_strides);
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto cbegin() const
    {
        return StridedDiscreteDomainIterator<DDim0>(front(), m_strides);
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto cend() const
    {
        return StridedDiscreteDomainIterator<
                DDim0>(m_element_begin + m_extents * m_strides, m_strides);
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
class StridedDiscreteDomain<>
{
    template <class...>
    friend class StridedDiscreteDomain;

public:
    using discrete_element_type = DiscreteElement<>;

    using discrete_vector_type = DiscreteVector<>;

    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return 0;
    }

    KOKKOS_DEFAULTED_FUNCTION constexpr StridedDiscreteDomain() = default;

    // Construct a StridedDiscreteDomain from a reordered copy of `domain`
    template <class... ODDims>
    KOKKOS_FUNCTION constexpr explicit StridedDiscreteDomain(
            [[maybe_unused]] StridedDiscreteDomain<ODDims...> const& domain)
    {
    }

    /** Construct a StridedDiscreteDomain starting from element_begin with size points.
     * @param element_begin the lower bound in each direction
     * @param size the number of points in each direction
     * @param strides the step between two elements
     */
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain(
            [[maybe_unused]] discrete_element_type const& element_begin,
            [[maybe_unused]] discrete_vector_type const& size,
            [[maybe_unused]] discrete_vector_type const& strides)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain(StridedDiscreteDomain const& x) = default;

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain(StridedDiscreteDomain&& x) = default;

    KOKKOS_DEFAULTED_FUNCTION ~StridedDiscreteDomain() = default;

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain& operator=(StridedDiscreteDomain const& x)
            = default;

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain& operator=(StridedDiscreteDomain&& x) = default;

    KOKKOS_FUNCTION constexpr bool operator==(
            [[maybe_unused]] StridedDiscreteDomain const& other) const
    {
        return true;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    KOKKOS_FUNCTION constexpr bool operator!=(StridedDiscreteDomain const& other) const
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

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain take_first(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain take_last(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove_first(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove_last(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove(
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
KOKKOS_FUNCTION constexpr StridedDiscreteDomain<QueryDDims...> select(
        StridedDiscreteDomain<DDims...> const& domain)
{
    return StridedDiscreteDomain<QueryDDims...>(
            DiscreteElement<QueryDDims...>(domain.front()),
            DiscreteVector<QueryDDims...>(domain.extents()),
            DiscreteVector<QueryDDims...>(domain.strides()));
}

namespace detail {

template <class T>
struct ConvertTypeSeqToStridedDiscreteDomain;

template <class... DDims>
struct ConvertTypeSeqToStridedDiscreteDomain<detail::TypeSeq<DDims...>>
{
    using type = StridedDiscreteDomain<DDims...>;
};

template <class T>
using convert_type_seq_to_strided_discrete_domain_t =
        typename ConvertTypeSeqToStridedDiscreteDomain<T>::type;

} // namespace detail

// Computes the subtraction DDom_a - DDom_b in the sense of linear spaces(retained dimensions are those in DDom_a which are not in DDom_b)
template <class... DDimsA, class... DDimsB>
KOKKOS_FUNCTION constexpr auto remove_dims_of(
        StridedDiscreteDomain<DDimsA...> const& DDom_a,
        [[maybe_unused]] StridedDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    using TagSeqA = detail::TypeSeq<DDimsA...>;
    using TagSeqB = detail::TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_strided_discrete_domain_t<type_seq_r>(DDom_a);
}

//! Remove the dimensions DDimsB from DDom_a
//! @param[in] DDom_a The discrete domain on which to remove dimensions
//! @return The discrete domain without DDimsB dimensions
template <class... DDimsB, class... DDimsA>
KOKKOS_FUNCTION constexpr auto remove_dims_of(
        StridedDiscreteDomain<DDimsA...> const& DDom_a) noexcept
{
    using TagSeqA = detail::TypeSeq<DDimsA...>;
    using TagSeqB = detail::TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_strided_discrete_domain_t<type_seq_r>(DDom_a);
}

namespace detail {

// Checks if dimension of DDom_a is DDim1. If not, returns restriction to DDim2 of DDom_b. May not be useful in its own, it helps for replace_dim_of
template <typename DDim1, typename DDim2, typename DDimA, typename... DDimsB>
KOKKOS_FUNCTION constexpr std::conditional_t<
        std::is_same_v<DDimA, DDim1>,
        ddc::StridedDiscreteDomain<DDim2>,
        ddc::StridedDiscreteDomain<DDimA>>
replace_dim_of_1d(
        StridedDiscreteDomain<DDimA> const& DDom_a,
        [[maybe_unused]] StridedDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    if constexpr (std::is_same_v<DDimA, DDim1>) {
        return ddc::StridedDiscreteDomain<DDim2>(DDom_b);
    } else {
        return DDom_a;
    }
}

} // namespace detail

// Replace in DDom_a the dimension Dim1 by the dimension Dim2 of DDom_b
template <typename DDim1, typename DDim2, typename... DDimsA, typename... DDimsB>
KOKKOS_FUNCTION constexpr auto replace_dim_of(
        StridedDiscreteDomain<DDimsA...> const& DDom_a,
        [[maybe_unused]] StridedDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    // TODO : static_asserts
    using TagSeqA = detail::TypeSeq<DDimsA...>;
    using TagSeqB = detail::TypeSeq<DDim1>;
    using TagSeqC = detail::TypeSeq<DDim2>;

    using type_seq_r = ddc::type_seq_replace_t<TagSeqA, TagSeqB, TagSeqC>;
    return ddc::detail::convert_type_seq_to_strided_discrete_domain_t<type_seq_r>(
            detail::replace_dim_of_1d<
                    DDim1,
                    DDim2,
                    DDimsA,
                    DDimsB...>(ddc::StridedDiscreteDomain<DDimsA>(DDom_a), DDom_b)...);
}

template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDims...> extents(
        StridedDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteVector<QueryDDims...>(StridedDiscreteDomain<QueryDDims>(domain).size()...);
}

template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> front(
        StridedDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(StridedDiscreteDomain<QueryDDims>(domain).front()...);
}

template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> back(
        StridedDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(StridedDiscreteDomain<QueryDDims>(domain).back()...);
}

template <class DDim>
struct StridedDiscreteDomainIterator
{
private:
    DiscreteElement<DDim> m_value = DiscreteElement<DDim>();

    DiscreteVector<DDim> m_stride = DiscreteVector<DDim>();

public:
    using iterator_category = std::random_access_iterator_tag;

    using value_type = DiscreteElement<DDim>;

    using difference_type = std::ptrdiff_t;

    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomainIterator() = default;

    KOKKOS_FUNCTION constexpr explicit StridedDiscreteDomainIterator(
            DiscreteElement<DDim> value,
            DiscreteVector<DDim> stride)
        : m_value(value)
        , m_stride(stride)
    {
    }

    KOKKOS_FUNCTION constexpr DiscreteElement<DDim> operator*() const noexcept
    {
        return m_value;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator& operator++()
    {
        m_value += m_stride;
        return *this;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator& operator--()
    {
        m_value -= m_stride;
        return *this;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator--(int)
    {
        auto tmp = *this;
        --*this;
        return tmp;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator& operator+=(difference_type n)
    {
        if (n >= difference_type(0)) {
            m_value += static_cast<DiscreteElementType>(n) * m_stride;
        } else {
            m_value -= static_cast<DiscreteElementType>(-n) * m_stride;
        }
        return *this;
    }

    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator& operator-=(difference_type n)
    {
        if (n >= difference_type(0)) {
            m_value -= static_cast<DiscreteElementType>(n) * m_stride;
        } else {
            m_value += static_cast<DiscreteElementType>(-n) * m_stride;
        }
        return *this;
    }

    KOKKOS_FUNCTION constexpr DiscreteElement<DDim> operator[](difference_type n) const
    {
        return m_value + n * m_stride;
    }

    friend KOKKOS_FUNCTION constexpr bool operator==(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return xx.m_value == yy.m_value;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    friend KOKKOS_FUNCTION constexpr bool operator!=(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return xx.m_value != yy.m_value;
    }
#endif

    friend KOKKOS_FUNCTION constexpr bool operator<(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return xx.m_value < yy.m_value;
    }

    friend KOKKOS_FUNCTION constexpr bool operator>(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return yy < xx;
    }

    friend KOKKOS_FUNCTION constexpr bool operator<=(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return !(yy < xx);
    }

    friend KOKKOS_FUNCTION constexpr bool operator>=(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return !(xx < yy);
    }

    friend KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator+(
            StridedDiscreteDomainIterator i,
            difference_type n)
    {
        return i += n;
    }

    friend KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator+(
            difference_type n,
            StridedDiscreteDomainIterator i)
    {
        return i += n;
    }

    friend KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator-(
            StridedDiscreteDomainIterator i,
            difference_type n)
    {
        return i -= n;
    }

    friend KOKKOS_FUNCTION constexpr difference_type operator-(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return (yy.m_value > xx.m_value) ? (-static_cast<difference_type>(yy.m_value - xx.m_value))
                                         : (xx.m_value - yy.m_value);
    }
};

} // namespace ddc
