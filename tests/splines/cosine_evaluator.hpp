// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <Kokkos_MathematicalConstants.hpp>

struct CosineEvaluator
{
    template <class DDim>
    class Evaluator
    {
    public:
        using Dim = DDim;

    private:
        static constexpr ddc::Real s_2_pi = 2 * Kokkos::numbers::pi;

        static constexpr ddc::Real s_pi_2 = Kokkos::numbers::pi / 2;

    private:
        ddc::Real m_coef0;

        ddc::Real m_coef1;

    public:
        template <class Domain>
        explicit Evaluator(Domain /*domain*/) : m_coef0(1.0)
                                              , m_coef1(0.0)
        {
        }

        Evaluator(ddc::Real coef0, ddc::Real coef1) : m_coef0(coef0), m_coef1(coef1) {}

        KOKKOS_FUNCTION ddc::Real operator()(ddc::Real const x) const noexcept
        {
            return eval(x, 0);
        }

        KOKKOS_FUNCTION void operator()(
                ddc::ChunkSpan<ddc::Real, ddc::DiscreteDomain<DDim>> chunk) const
        {
            ddc::DiscreteDomain<DDim> const domain = chunk.domain();

            for (ddc::DiscreteElement<DDim> const i : domain) {
                chunk(i) = eval(ddc::coordinate(i), 0);
            }
        }

        KOKKOS_FUNCTION ddc::Real deriv(ddc::Real const x, int const derivative) const noexcept
        {
            return eval(x, derivative);
        }

        KOKKOS_FUNCTION void deriv(
                ddc::ChunkSpan<ddc::Real, ddc::DiscreteDomain<DDim>> chunk,
                int const derivative) const
        {
            ddc::DiscreteDomain<DDim> const domain = chunk.domain();

            for (ddc::DiscreteElement<DDim> const i : domain) {
                chunk(i) = eval(ddc::coordinate(i), derivative);
            }
        }

        KOKKOS_FUNCTION ddc::Real max_norm(int diff = 0) const
        {
            return ddc::detail::ipow(s_2_pi * m_coef0, diff);
        }

    private:
        KOKKOS_FUNCTION ddc::Real eval(ddc::Real const x, int const derivative) const noexcept
        {
            return ddc::detail::ipow(s_2_pi * m_coef0, derivative)
                   * Kokkos::cos(s_pi_2 * derivative + s_2_pi * (m_coef0 * x + m_coef1));
        }
    };
};
