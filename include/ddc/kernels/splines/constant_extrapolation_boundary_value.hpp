#pragma once

#include "spline_boundary_value.hpp"
#include "view.hpp"

namespace ddc {
template <class BSplines>
class ConstantExtrapolationBoundaryValue : public SplineBoundaryValue<BSplines>
{
public:
    using tag_type = typename BSplines::tag_type;
    using coord_type = ddc::Coordinate<tag_type>;

private:
    coord_type m_eval_pos;

public:
    explicit ConstantExtrapolationBoundaryValue(coord_type eval_pos) : m_eval_pos(eval_pos) {}

    ~ConstantExtrapolationBoundaryValue() override = default;

    template <class Layout, class MemorySpace>
    KOKKOS_FUNCTION double operator()(
            coord_type,
            ddc::ChunkSpan<double const, ddc::DiscreteDomain<BSplines>, Layout, MemorySpace> const
                    spline_coef) const final
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
