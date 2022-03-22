// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/for_each.hpp"
#include "ddc/reducer.hpp"

namespace detail {

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

/** A parallel reduction over a nd domain
 * @param[in] domain    the nd domain to iterate
 * @param[in] reduce    a reduction operation
 * @param[in] transform a functor taking a nd discrete coordinate as parameter
 */
template <class... DDims, class BinaryReductionOp, class UnaryTransformOp>
inline typename BinaryReductionOp::result_type transform_reduce(
        omp_policy,
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
    return init;
}

/** A parallel reduction over a nd domain
 * @param[in] domain    the nd domain to iterate
 * @param[in] reduce    a reduction operation
 * @param[in] transform a functor taking a nd discrete coordinate as parameter
 */
template <class... DDims, class BinaryReductionOp, class UnaryTransformOp>
inline typename BinaryReductionOp::result_type transform_reduce(serial_policy,
        DiscreteDomain<DDims...> const& domain,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return detail::transform_reduce_serial(
            domain,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

/** A parallel reduction over a nd domain
 * @param[in] domain    the nd domain to iterate
 * @param[in] reduce    a reduction operation
 * @param[in] transform a functor taking a nd discrete coordinate as parameter
 */
template <class... DDims, class BinaryReductionOp, class UnaryTransformOp>
inline typename BinaryReductionOp::result_type transform_reduce(
        DiscreteDomain<DDims...> const& domain,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return detail::transform_reduce_serial(
            domain,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}
