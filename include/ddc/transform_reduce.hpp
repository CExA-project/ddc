// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/for_each.hpp"

namespace detail {

template <
        class... DDims,
        class T,
        class BinaryReductionOp,
        class UnaryTransformOp,
        class... DCoords>
inline T transform_reduce_serial(
        DiscreteDomain<DDims...> const& domain,
        T init,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform,
        DCoords const&... dcoords) noexcept
{
    if constexpr (sizeof...(DCoords) == sizeof...(DDims)) {
        return reduce(init, transform(DiscreteCoordinate<DDims...>(dcoords...)));
    } else {
        using CurrentDDim = type_seq_element_t<sizeof...(DCoords), detail::TypeSeq<DDims...>>;
        for (DiscreteCoordinate<CurrentDDim> const ii : select<CurrentDDim>(domain)) {
            init = transform_reduce_serial(domain, init, reduce, transform, dcoords..., ii);
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
template <class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
inline T transform_reduce(
        DiscreteDomain<DDims...> const& domain,
        T init,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return detail::transform_reduce_serial(
            domain,
            init,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}
