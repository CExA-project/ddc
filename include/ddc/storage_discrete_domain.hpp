// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "ddc/detail/kokkos.hpp"
#include "ddc/detail/tagged_vector.hpp"
#include "ddc/detail/type_seq.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_vector.hpp"

namespace ddc {

template <class DDim>
struct StorageDiscreteDomainIterator;

template <class... DDims>
class StorageDiscreteDomain;

template <class T>
struct is_storage_discrete_domain : std::false_type
{
};

template <class... Tags>
struct is_storage_discrete_domain<StorageDiscreteDomain<Tags...>> : std::true_type
{
};

template <class T>
inline constexpr bool is_storage_discrete_domain_v = is_storage_discrete_domain<T>::value;


namespace detail {

template <class... Tags>
struct ToTypeSeq<StorageDiscreteDomain<Tags...>>
{
    using type = TypeSeq<Tags...>;
};

template <class InputIt1, class InputIt2>
KOKKOS_FUNCTION bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2)
{
    for (; first1 != last1; ++first1, ++first2) {
        if (!(*first1 == *first2)) {
            return false;
        }
    }

    return true;
}

template <
        class ForwardIt,
        class T = typename std::iterator_traits<ForwardIt>::value_type,
        class Compare>
KOKKOS_FUNCTION ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
{
    ForwardIt it;
    typename std::iterator_traits<ForwardIt>::difference_type count = last - first;
    while (count > 0) {
        it = first;
        typename std::iterator_traits<ForwardIt>::difference_type const step = count / 2;
        it += step;

        if (comp(*it, value)) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }

    return first;
}

template <
        class ForwardIt,
        class T = typename std::iterator_traits<ForwardIt>::value_type,
        class Compare>
bool binary_search(ForwardIt first, ForwardIt last, const T& value, Compare comp)
{
    first = ::ddc::detail::lower_bound(first, last, value, comp);
    return (!(first == last) && !(comp(value, *first)));
}

template <class DDim>
struct GetUidFn
{
    KOKKOS_FUNCTION DiscreteElementType
    operator()(DiscreteElement<DDim> const& delem) const noexcept
    {
        return delem.uid();
    }
};

} // namespace detail

template <class... DDims>
class StorageDiscreteDomain
{
    template <class...>
    friend class StorageDiscreteDomain;

public:
    using discrete_element_type = DiscreteElement<DDims...>;

    using discrete_vector_type = DiscreteVector<DDims...>;

private:
    detail::TaggedVector<Kokkos::View<DiscreteElementType*, Kokkos::SharedSpace>, DDims...> m_views;

public:
    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return sizeof...(DDims);
    }

    KOKKOS_DEFAULTED_FUNCTION StorageDiscreteDomain() = default;

    /// Construct a StorageDiscreteDomain by copies and merge of domains
    template <
            class... DDoms,
            class = std::enable_if_t<(is_storage_discrete_domain_v<DDoms> && ...)>>
    KOKKOS_FUNCTION constexpr explicit StorageDiscreteDomain(DDoms const&... domains)
        : m_views(domains...)
    {
    }

    /** Construct a StorageDiscreteDomain with Kokkos::View explicitly listing the discrete elements.
     * @param views list of Kokkos::View
     */
    explicit StorageDiscreteDomain(
            Kokkos::View<DiscreteElement<DDims>*, Kokkos::SharedSpace> const&... views)
    {
        ((m_views[type_seq_rank_v<DDims, detail::TypeSeq<DDims...>>]
          = Kokkos::View<DiscreteElementType*, Kokkos::SharedSpace>(views.label(), views.size())),
         ...);
        Kokkos::DefaultExecutionSpace const execution_space;
        ((Kokkos::Experimental::transform(
                 "StorageDiscreteDomainCtor",
                 execution_space,
                 views,
                 m_views[type_seq_rank_v<DDims, detail::TypeSeq<DDims...>>],
                 detail::GetUidFn<DDims>())),
         ...);
        execution_space.fence("StorageDiscreteDomainCtor");
    }

    KOKKOS_DEFAULTED_FUNCTION StorageDiscreteDomain(StorageDiscreteDomain const& x) = default;

    KOKKOS_DEFAULTED_FUNCTION StorageDiscreteDomain(StorageDiscreteDomain&& x) = default;

    KOKKOS_DEFAULTED_FUNCTION ~StorageDiscreteDomain() = default;

    KOKKOS_DEFAULTED_FUNCTION StorageDiscreteDomain& operator=(StorageDiscreteDomain const& x)
            = default;

    KOKKOS_DEFAULTED_FUNCTION StorageDiscreteDomain& operator=(StorageDiscreteDomain&& x) = default;

    template <class... ODims>
    KOKKOS_FUNCTION constexpr bool operator==(StorageDiscreteDomain<ODims...> const& other) const
    {
        if (empty() && other.empty()) {
            return true;
        }
        if (m_views.size() != other.m_views.size()) {
            return false;
        }
        for (std::size_t i = 0; i < m_views.size(); ++i) {
            if (m_views[i].size() != other.m_views[i].size()) {
                return false;
            }
            if (!detail::
                        equal(m_views[i].data(),
                              m_views[i].data() + m_views[i].size(),
                              other.m_views[i].data())) {
                return false;
            }
        }
        return true;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    template <class... ODims>
    KOKKOS_FUNCTION constexpr bool operator!=(StorageDiscreteDomain<ODims...> const& other) const
    {
        return !(*this == other);
    }
#endif

    KOKKOS_FUNCTION constexpr std::size_t size() const
    {
        return (1UL * ... * get<DDims>(m_views).size());
    }

    KOKKOS_FUNCTION constexpr discrete_vector_type extents() const noexcept
    {
        return discrete_vector_type(get<DDims>(m_views).size()...);
    }

    KOKKOS_FUNCTION constexpr auto discrete_elements() const noexcept
    {
        return m_views;
    }

    template <class QueryDDim>
    KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDim> extent() const noexcept
    {
        return DiscreteVector<QueryDDim>(get<QueryDDim>(m_views).size());
    }

    KOKKOS_FUNCTION constexpr discrete_element_type front() const noexcept
    {
        return discrete_element_type(get<DDims>(m_views)(0)...);
    }

    KOKKOS_FUNCTION constexpr discrete_element_type back() const noexcept
    {
        return discrete_element_type(get<DDims>(m_views)(get<DDims>(m_views).size() - 1)...);
    }

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain take_first(discrete_vector_type n) const
    {
        return StorageDiscreteDomain(front(), n);
    }

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain take_last(discrete_vector_type n) const
    {
        return StorageDiscreteDomain(front() + prod(extents() - n), n);
    }

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain remove_first(discrete_vector_type n) const
    {
        return StorageDiscreteDomain(front() + prod(n), extents() - n);
    }

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain remove_last(discrete_vector_type n) const
    {
        return StorageDiscreteDomain(front(), extents() - n);
    }

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain remove(
            discrete_vector_type n1,
            discrete_vector_type n2) const
    {
        return StorageDiscreteDomain(front() + prod(n1), extents() - n1 - n2);
    }

    KOKKOS_FUNCTION constexpr DiscreteElement<DDims...> operator()(
            DiscreteVector<DDims...> const& dvect) const noexcept
    {
        return m_views(get<DDims>(dvect)...);
    }

    template <class... DElems>
    KOKKOS_FUNCTION bool contains(DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        return (detail::binary_search(
                        get<DDims>(m_views).data(),
                        get<DDims>(m_views).data() + get<DDims>(m_views).size(),
                        uid<DDims>(take<DDims>(delems...)),
                        std::less {})
                && ...);
    }

    template <class... DElems>
    KOKKOS_FUNCTION DiscreteVector<DDims...> distance_from_front(
            DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        assert(contains(delems...));
        return DiscreteVector<DDims...>(
                (detail::lower_bound(
                         get<DDims>(m_views).data(),
                         get<DDims>(m_views).data() + get<DDims>(m_views).size(),
                         uid<DDims>(take<DDims>(delems...)),
                         std::less {})
                 - get<DDims>(m_views).data())...);
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
        return Kokkos::Experimental::begin(get<DDim0>(m_views));
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto end() const
    {
        return Kokkos::Experimental::end(get<DDim0>(m_views));
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto cbegin() const
    {
        return Kokkos::Experimental::cbegin(get<DDim0>(m_views));
    }

    template <
            std::size_t N = sizeof...(DDims),
            class DDim0 = std::enable_if_t<N == 1, std::tuple_element_t<0, std::tuple<DDims...>>>>
    KOKKOS_FUNCTION auto cend() const
    {
        return Kokkos::Experimental::cend(get<DDim0>(m_views));
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
class StorageDiscreteDomain<>
{
    template <class...>
    friend class StorageDiscreteDomain;

public:
    using discrete_element_type = DiscreteElement<>;

    using discrete_vector_type = DiscreteVector<>;

    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return 0;
    }

    KOKKOS_DEFAULTED_FUNCTION constexpr StorageDiscreteDomain() = default;

    // Construct a StorageDiscreteDomain from a reordered copy of `domain`
    template <class... ODDims>
    KOKKOS_FUNCTION constexpr explicit StorageDiscreteDomain(
            [[maybe_unused]] StorageDiscreteDomain<ODDims...> const& domain)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION StorageDiscreteDomain(StorageDiscreteDomain const& x) = default;

    KOKKOS_DEFAULTED_FUNCTION StorageDiscreteDomain(StorageDiscreteDomain&& x) = default;

    KOKKOS_DEFAULTED_FUNCTION ~StorageDiscreteDomain() = default;

    KOKKOS_DEFAULTED_FUNCTION StorageDiscreteDomain& operator=(StorageDiscreteDomain const& x)
            = default;

    KOKKOS_DEFAULTED_FUNCTION StorageDiscreteDomain& operator=(StorageDiscreteDomain&& x) = default;

    KOKKOS_FUNCTION constexpr bool operator==(
            [[maybe_unused]] StorageDiscreteDomain const& other) const
    {
        return true;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    KOKKOS_FUNCTION constexpr bool operator!=(StorageDiscreteDomain const& other) const
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

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain take_first(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain take_last(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain remove_first(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain remove_last(
            [[maybe_unused]] discrete_vector_type n) const
    {
        return *this;
    }

    KOKKOS_FUNCTION constexpr StorageDiscreteDomain remove(
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

    static bool contains() noexcept
    {
        return true;
    }

    static bool contains(DiscreteElement<>) noexcept
    {
        return true;
    }

    static DiscreteVector<> distance_from_front() noexcept
    {
        return {};
    }

    static DiscreteVector<> distance_from_front(DiscreteElement<>) noexcept
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
KOKKOS_FUNCTION constexpr StorageDiscreteDomain<QueryDDims...> select(
        StorageDiscreteDomain<DDims...> const& domain)
{
    return StorageDiscreteDomain<QueryDDims...>(
            DiscreteElement<QueryDDims...>(domain.front()),
            DiscreteElement<QueryDDims...>(domain.extents()));
}

namespace detail {

template <class T>
struct ConvertTypeSeqToStorageDiscreteDomain;

template <class... DDims>
struct ConvertTypeSeqToStorageDiscreteDomain<detail::TypeSeq<DDims...>>
{
    using type = StorageDiscreteDomain<DDims...>;
};

template <class T>
using convert_type_seq_to_storage_discrete_domain_t =
        typename ConvertTypeSeqToStorageDiscreteDomain<T>::type;

} // namespace detail

// Computes the subtraction DDom_a - DDom_b in the sense of linear spaces(retained dimensions are those in DDom_a which are not in DDom_b)
template <class... DDimsA, class... DDimsB>
KOKKOS_FUNCTION constexpr auto remove_dims_of(
        StorageDiscreteDomain<DDimsA...> const& DDom_a,
        [[maybe_unused]] StorageDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    using TagSeqA = detail::TypeSeq<DDimsA...>;
    using TagSeqB = detail::TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_storage_discrete_domain_t<type_seq_r>(DDom_a);
}

namespace detail {

// Checks if dimension of DDom_a is DDim1. If not, returns restriction to DDim2 of DDom_b. May not be useful in its own, it helps for replace_dim_of
template <typename DDim1, typename DDim2, typename DDimA, typename... DDimsB>
KOKKOS_FUNCTION constexpr std::conditional_t<
        std::is_same_v<DDimA, DDim1>,
        ddc::StorageDiscreteDomain<DDim2>,
        ddc::StorageDiscreteDomain<DDimA>>
replace_dim_of_1d(
        StorageDiscreteDomain<DDimA> const& DDom_a,
        [[maybe_unused]] StorageDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    if constexpr (std::is_same_v<DDimA, DDim1>) {
        return ddc::StorageDiscreteDomain<DDim2>(DDom_b);
    } else {
        return DDom_a;
    }
}

} // namespace detail

// Replace in DDom_a the dimension Dim1 by the dimension Dim2 of DDom_b
template <typename DDim1, typename DDim2, typename... DDimsA, typename... DDimsB>
KOKKOS_FUNCTION constexpr auto replace_dim_of(
        StorageDiscreteDomain<DDimsA...> const& DDom_a,
        [[maybe_unused]] StorageDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    // TODO : static_asserts
    using TagSeqA = detail::TypeSeq<DDimsA...>;
    using TagSeqB = detail::TypeSeq<DDim1>;
    using TagSeqC = detail::TypeSeq<DDim2>;

    using type_seq_r = ddc::type_seq_replace_t<TagSeqA, TagSeqB, TagSeqC>;
    return ddc::detail::convert_type_seq_to_storage_discrete_domain_t<type_seq_r>(
            detail::replace_dim_of_1d<
                    DDim1,
                    DDim2,
                    DDimsA,
                    DDimsB...>(ddc::StorageDiscreteDomain<DDimsA>(DDom_a), DDom_b)...);
}

template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDims...> extents(
        StorageDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteVector<QueryDDims...>(StorageDiscreteDomain<QueryDDims>(domain).size()...);
}

template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> front(
        StorageDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(StorageDiscreteDomain<QueryDDims>(domain).front()...);
}

template <class... QueryDDims, class... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> back(
        StorageDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(StorageDiscreteDomain<QueryDDims>(domain).back()...);
}

} // namespace ddc
