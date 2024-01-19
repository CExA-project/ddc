#pragma once

#include "view.hpp"

namespace ddc {
template <class DDim>
struct ConstantExtrapolationRule
{
private:
    ddc::DiscreteElement<DDim> m_eval_pos;

public:
    explicit ConstantExtrapolationRule(ddc::DiscreteElement<DDim> eval_pos) : m_eval_pos(eval_pos)
    {
    }

    template <class CoordType, class BSplines, class Layout, class MemorySpace>
    KOKKOS_INLINE_FUNCTION double operator()(
            CoordType,
            ddc::ChunkSpan<double const, ddc::DiscreteDomain<BSplines>, Layout, MemorySpace> const
                    spline_coef) const
    {
        std::array<double, BSplines::degree() + 1> vals;

        ddc::DiscreteElement<BSplines> idx
                = ddc::discrete_space<BSplines>().eval_basis(vals, m_eval_pos);

        double y = 0.0;
        for (std::size_t i = 0; i < BSplines::degree() + 1; ++i) {
            y += spline_coef(idx + i) * vals[i];
        }
        return y;
    }
};
} // namespace ddc
