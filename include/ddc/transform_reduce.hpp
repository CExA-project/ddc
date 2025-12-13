// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "detail/macros.hpp"

#include "discrete_domain.hpp"
#include "discrete_element.hpp"

namespace ddc {

namespace detail {

/** A serial reduction over a nD domain
 * @param[in] begin iterator indicating the beginning of the domain
 * @param[in] end iterator indicating the end of the domain
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 * @param[in] dcoords discrete elements from dimensions already in a loop
 */
template <
        class Support,
        class T,
        class Element,
        std::size_t N,
        class BinaryReductionOp,
        class UnaryTransformOp,
        class... Is>
KOKKOS_FUNCTION T host_transform_reduce_serial(
        std::array<Element, N> const& begin,
        std::array<Element, N> const& end,
        [[maybe_unused]] T const neutral,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        return transform(Support(is...));
    } else {
        T result = neutral;
        for (Element ii = begin[I]; ii < end[I]; ++ii) {
            result = reduce(
                    host_transform_reduce_serial<
                            Support>(begin, end, neutral, reduce, transform, is..., ii),
                    result);
        }
        return result;
    }
}

/** A serial reduction over a nD domain. Can be called from a device kernel.
 * @param[in] begin iterator indicating the beginning of the domain
 * @param[in] end iterator indicating the end of the domain
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 * @param[in] dcoords discrete elements from dimensions already in a loop
 */
template <
        class Support,
        class T,
        class Element,
        std::size_t N,
        class BinaryReductionOp,
        class UnaryTransformOp,
        class... Is>
KOKKOS_FUNCTION T device_transform_reduce_serial(
        std::array<Element, N> const& begin,
        std::array<Element, N> const& end,
        [[maybe_unused]] T const neutral,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        return transform(Support(is...));
    } else {
        T result = neutral;
        for (Element ii = begin[I]; ii < end[I]; ++ii) {
            result = reduce(
                    device_transform_reduce_serial<
                            Support>(begin, end, neutral, reduce, transform, is..., ii),
                    result);
        }
        return result;
    }
}
} // namespace detail

template <class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
[[deprecated("Use host_transform_reduce instead")]]
T transform_reduce(
        DiscreteDomain<DDims...> const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    host_transform_reduce(
            domain,
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

/** A reduction over a nD domain in serial
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
T host_transform_reduce(
        DiscreteDomain<DDims...> const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    DiscreteElement<DDims...> const ddc_begin = domain.front();
    DiscreteElement<DDims...> const ddc_end = domain.front() + domain.extents();

    std::array const begin = detail::array(ddc_begin);
    std::array const end = detail::array(ddc_end);
    return detail::host_transform_reduce_serial<DiscreteElement<DDims...>>(
            begin,
            end,
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

/** A reduction over a nD domain in serial. Can be called from a device kernel.
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
KOKKOS_FUNCTION T device_transform_reduce(
        DiscreteDomain<DDims...> const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    DiscreteElement<DDims...> const ddc_begin = domain.front();
    DiscreteElement<DDims...> const ddc_end = domain.front() + domain.extents();

    std::array const begin = detail::array(ddc_begin);
    std::array const end = detail::array(ddc_end);
    return detail::device_transform_reduce_serial<DiscreteElement<DDims...>>(
            begin,
            end,
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

/** A reduction over a nD domain in serial. Can be called from a device kernel.
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
KOKKOS_FUNCTION T device_transform_reduce(
        StridedDiscreteDomain<DDims...> const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    using discrete_element_type = typename StridedDiscreteDomain<DDims...>::discrete_element_type;
    using discrete_vector_type = typename StridedDiscreteDomain<DDims...>::discrete_vector_type;
    discrete_element_type const ddc_begin = domain.front();
    discrete_element_type const ddc_end = domain.front() + domain.extents();

    std::array const begin = detail::array(ddc_begin);
    std::array const end = detail::array(ddc_end);
    return detail::device_transform_reduce_serial<discrete_vector_type>(
            begin,
            end,
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

/** A reduction over a nD domain in serial. Can be called from a device kernel.
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
KOKKOS_FUNCTION T device_transform_reduce(
        SparseDiscreteDomain<DDims...> const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    using discrete_element_type = typename SparseDiscreteDomain<DDims...>::discrete_element_type;
    using discrete_vector_type = typename SparseDiscreteDomain<DDims...>::discrete_vector_type;
    discrete_element_type const ddc_begin = domain.front();
    discrete_element_type const ddc_end = domain.front() + domain.extents();

    std::array const begin = detail::array(ddc_begin);
    std::array const end = detail::array(ddc_end);
    return detail::device_transform_reduce_serial<discrete_vector_type>(
            begin,
            end,
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

} // namespace ddc
