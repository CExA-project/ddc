#pragma once

#include "view.hpp"

namespace ddc {

template <class... Dim>
struct ConstantExtrapolationRule
{
};

template <class Dim>
struct ConstantExtrapolationRule<DDim>
{
private:
    ddc::Coordinate<Dim> m_eval_pos;

public:
    explicit ConstantExtrapolationRule(ddc::Coordinate<Dim> eval_pos) : m_eval_pos(eval_pos)
    {
    }

    template <class CoordType, class BSplines, class Layout, class MemorySpace>
    KOKKOS_FUNCTION double operator()(
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

template <class Dim1, class Dim2>
struct ConstantExtrapolationRule<Dim1,Dim2>
{
private:
    ddc::Coordinate<Dim1,Dim2> m_eval_pos;

public:
    explicit ConstantExtrapolationRule(ddc::Coordinate<Dim1,Dim2> eval_pos) : m_eval_pos(eval_pos)
    {
    }

    template <class CoordType, class BSplines, class Layout, class MemorySpace>
    KOKKOS_FUNCTION double operator()(
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
