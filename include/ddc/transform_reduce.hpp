#pragma once

#include <utility>

#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/for_each.hpp"

/** iterates over a 1d domain
 * @param[in] domain the 1d domain to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class DDim, class T, class BinaryReductionOp, class BinaryTransformOp>
inline T transform_reduce(
        serial_policy,
        DiscreteDomain<DDim> const& domain,
        T init,
        BinaryReductionOp&& reduce,
        BinaryTransformOp&& transform) noexcept
{
    DiscreteDomainIterator<DDim> const it_b = domain.begin();
    DiscreteDomainIterator<DDim> const it_e = domain.end();
    for (DiscreteDomainIterator<DDim> it = it_b; it != it_e; ++it) {
        init = reduce(init, transform(*it));
    }
    return init;
}

/** iterates over a 2d domain
 * @param[in] domain the 2d domain to iterate
 * @param[in] f      a functor taking a 2-index as parameter
 */
template <class DDim1, class DDim2, class T, class BinaryReductionOp, class BinaryTransformOp>
inline T transform_reduce(
        serial_policy,
        DiscreteDomain<DDim1, DDim2> const& domain,
        T init,
        BinaryReductionOp&& reduce,
        BinaryTransformOp&& transform) noexcept
{
    DiscreteDomainIterator<DDim1> const it1_b = select<DDim1>(domain).begin();
    DiscreteDomainIterator<DDim1> const it1_e = select<DDim1>(domain).end();
    DiscreteDomainIterator<DDim2> const it2_b = select<DDim2>(domain).begin();
    DiscreteDomainIterator<DDim2> const it2_e = select<DDim2>(domain).end();
    for (DiscreteDomainIterator<DDim1> it1 = it1_b; it1 != it1_e; ++it1) {
        for (DiscreteDomainIterator<DDim2> it2 = it2_b; it2 != it2_e; ++it2) {
            init = reduce(init, transform(DiscreteCoordinate<DDim1, DDim2>(*it1, *it2)));
        }
    }
    return init;
}

template <class... DDims, class T, class BinaryReductionOp, class BinaryTransformOp>
inline T transform_reduce(
        DiscreteDomain<DDims...> const& domain,
        T init,
        BinaryReductionOp&& reduce,
        BinaryTransformOp&& transform) noexcept
{
    return transform_reduce(
            serial_policy(),
            domain,
            init,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<BinaryTransformOp>(transform));
}
