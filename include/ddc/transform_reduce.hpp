// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/for_each.hpp"
#include "ddc/reducer.hpp"

namespace detail {

/** A serial reduction over a nD domain
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and init.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class... DDims, class BinaryReductionOp, class UnaryTransformOp, class... DCoords>
inline typename BinaryReductionOp::result_type transform_reduce_serial(
        DiscreteDomain<DDims...> const& domain,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform,
        DCoords const&... dcoords) noexcept
{
    if constexpr (sizeof...(DCoords) == sizeof...(DDims)) {
        return transform(DiscreteCoordinate<DDims...>(dcoords...));
    } else {
        using CurrentDDim = type_seq_element_t<sizeof...(DCoords), detail::TypeSeq<DDims...>>;
        typename BinaryReductionOp::result_type init;
        reduce.initialize(init);
        for (DiscreteCoordinate<CurrentDDim> const ii : select<CurrentDDim>(domain)) {
            reduce.reduce(init, transform_reduce_serial(domain, reduce, transform, dcoords..., ii));
        }
        return init;
    }
}

} // namespace detail

/** A reduction over a n-D domain using the OpenMP execution policy
 * @param[in] policy the execution policy to use
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and init.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class... DDims, class BinaryReductionOp, class UnaryTransformOp>
inline typename BinaryReductionOp::result_type transform_reduce(
        [[maybe_unused]] omp_policy policy,
        DiscreteDomain<DDims...> const& domain,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform) noexcept
{
#pragma omp declare reduction(OmpBinaryReductionOp                                                 \
                              : typename BinaryReductionOp::result_type                            \
                              : BinaryReductionOp::reduce(omp_out, omp_in))                        \
        initializer(BinaryReductionOp::initialize(omp_priv))

    typename BinaryReductionOp::result_type init;
    reduce.initialize(init);
    using FirstDDim = type_seq_element_t<0, detail::TypeSeq<DDims...>>;
    DiscreteDomainIterator<FirstDDim> const it_b = select<FirstDDim>(domain).begin();
    DiscreteDomainIterator<FirstDDim> const it_e = select<FirstDDim>(domain).end();

#pragma omp parallel for default(none) shared(it_b, it_e, domain, reduce, transform)               \
        reduction(OmpBinaryReductionOp                                                             \
                  : init)
    for (DiscreteDomainIterator<FirstDDim> it = it_b; it != it_e; ++it) {
        if constexpr (sizeof...(DDims) == 1) {
            reduce.reduce(init, transform(*it));
        } else {
            reduce.reduce(init, transform_reduce_serial(domain, reduce, transform, *it));
        }
    }
}

/** A reduction over a nD domain using the default execution policy
 * @param[in] policy the execution policy to use
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and init.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class... DDims, class BinaryReductionOp, class UnaryTransformOp>
inline auto transform_reduce(
        [[maybe_unused]] serial_policy policy,
        DiscreteDomain<DDims...> const& domain,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return detail::transform_reduce_serial(
            domain,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

/** A reduction over a nD domain using the default execution policy
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and init.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class... DDims, class BinaryReductionOp, class UnaryTransformOp>
inline auto transform_reduce(
        DiscreteDomain<DDims...> const& domain,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return transform_reduce(
            default_policy(),
            domain,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}
