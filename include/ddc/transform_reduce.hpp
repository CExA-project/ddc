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
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 * @param[in] dcoords discrete elements from dimensions already in a loop
 */
template <class Support, class T, class BinaryReductionOp, class UnaryTransformOp, class... DCoords>
T transform_reduce_serial(
        Support const& domain,
        [[maybe_unused]] T const neutral,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform,
        DCoords const&... dcoords) noexcept
{
    DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
    if constexpr (sizeof...(DCoords) == Support::rank()) {
        return transform(typename Support::discrete_element_type(dcoords...));
    } else {
        using CurrentDDim = type_seq_element_t<sizeof...(DCoords), to_type_seq_t<Support>>;
        T result = neutral;
        for (DiscreteElement<CurrentDDim> const ii : DiscreteDomain<CurrentDDim>(domain)) {
            result = reduce(
                    result,
                    transform_reduce_serial(domain, neutral, reduce, transform, dcoords..., ii));
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
        class... DDims,
        class T,
        class BinaryReductionOp,
        class UnaryTransformOp,
        class... DCoords>
KOKKOS_FUNCTION T annotated_transform_reduce_serial(
        DiscreteDomain<DDims...> const& domain,
        [[maybe_unused]] T const neutral,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform,
        DCoords const&... dcoords) noexcept
{
    DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
    if constexpr (sizeof...(DCoords) == sizeof...(DDims)) {
        return transform(DiscreteElement<DDims...>(dcoords...));
    } else {
        using CurrentDDim = type_seq_element_t<sizeof...(DCoords), detail::TypeSeq<DDims...>>;
        T result = neutral;
        for (DiscreteElement<CurrentDDim> const ii : select<CurrentDDim>(domain)) {
            result = reduce(
                    result,
                    annotated_transform_reduce_serial(
                            domain,
                            neutral,
                            reduce,
                            transform,
                            dcoords...,
                            ii));
        }
        return result;
    }
    DDC_IF_NVCC_THEN_POP
}

template <class Support, class T, class Element, std::size_t N, 
         class BinaryReductionOp,
         class UnaryTransformOp, class... Is>
KOKKOS_FUNCTION T annotated_transform_reduce_serial(
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
              annotated_transform_reduce_serial<Support>(begin, end, neutral, reduce, transform, is..., ii),
              result);
        }
        return result;
    }
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
T transform_reduce(
        Support const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return detail::transform_reduce_serial(
            domain,
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
KOKKOS_FUNCTION T annotated_transform_reduce(
        DiscreteDomain<DDims...> const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return detail::annotated_transform_reduce_serial(
            domain,
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
KOKKOS_FUNCTION T annotated_transform_reduce(
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
    return detail::annotated_transform_reduce_serial<discrete_vector_type>(
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
KOKKOS_FUNCTION T annotated_transform_reduce(
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
    return detail::annotated_transform_reduce_serial<discrete_vector_type>(
                begin, 
                end, 
                neutral,
                std::forward<BinaryReductionOp>(reduce), 
                std::forward<UnaryTransformOp>(transform));
}

} // namespace ddc
