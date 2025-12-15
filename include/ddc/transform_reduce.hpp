// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "detail/macros.hpp"

#include "discrete_element.hpp"

namespace ddc {

namespace detail {

/** A serial reduction over a nD domain
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 * @param[in] dcoords discrete elements from dimensions already in a loop
 */
template <
        class Support,
        class Element,
        std::size_t N,
        class T,
        class BinaryReductionOp,
        class UnaryTransformOp,
        class... Is>
T host_transform_reduce_serial(
        Support const& domain,
        std::array<Element, N> const& size,
        [[maybe_unused]] T const neutral,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform,
        Is const&... is) noexcept
{
    DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        return transform(domain(typename Support::discrete_vector_type(is...)));
    } else {
        T result = neutral;
        for (Element ii = 0; ii < size[I]; ++ii) {
            result = reduce(
                    host_transform_reduce_serial(
                            domain,
                            size,
                            neutral,
                            reduce,
                            transform,
                            is...,
                            ii),
                    result);
        }
        return result;
    }
    DDC_IF_NVCC_THEN_POP
}

/** A serial reduction over a nD domain. Can be called from a device kernel.
 * @param[in] domain the range over which to apply the algorithm
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
        Support const& domain,
        std::array<Element, N> const& size,
        [[maybe_unused]] T const neutral,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform,
        Is const&... is) noexcept
{
    DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        return transform(domain(typename Support::discrete_vector_type(is...)));
    } else {
        T result = neutral;
        for (Element ii = 0; ii < size[I]; ++ii) {
            result = reduce(
                    device_transform_reduce_serial(
                            domain,
                            size,
                            neutral,
                            reduce,
                            transform,
                            is...,
                            ii),
                    result);
        }
        return result;
    }
    DDC_IF_NVCC_THEN_POP
}

} // namespace detail

/** A reduction over a nD domain in serial
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class Support, class T, class BinaryReductionOp, class UnaryTransformOp>
[[deprecated("Use host_transform_reduce instead")]]
T transform_reduce(
        Support const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return host_transform_reduce(
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
template <class Support, class T, class BinaryReductionOp, class UnaryTransformOp>
T host_transform_reduce(
        Support const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return detail::host_transform_reduce_serial(
            domain,
            detail::array(domain.extents()),
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
template <class Support, class T, class BinaryReductionOp, class UnaryTransformOp>
KOKKOS_FUNCTION T device_transform_reduce(
        Support const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return detail::device_transform_reduce_serial(
            domain,
            detail::array(domain.extents()),
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

} // namespace ddc
