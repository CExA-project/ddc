// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

struct Evaluator3D
{
    template <class Eval1, class Eval2, class Eval3>
    class Evaluator
    {
    private:
        Eval1 m_eval_func1;
        Eval2 m_eval_func2;
        Eval3 m_eval_func3;

    public:
        template <class Domain>
        explicit Evaluator(Domain domain)
            : m_eval_func1(ddc::DiscreteDomain<typename Eval1::Dim>(domain))
            , m_eval_func2(ddc::DiscreteDomain<typename Eval2::Dim>(domain))
            , m_eval_func3(ddc::DiscreteDomain<typename Eval3::Dim>(domain))
        {
        }

        KOKKOS_FUNCTION double operator()(double const x, double const y, double const z)
                const noexcept
        {
            return m_eval_func1(x) * m_eval_func2(y) * m_eval_func3(z);
        }

        template <class DDim1, class DDim2, class DDim3>
        KOKKOS_FUNCTION double operator()(
                ddc::Coordinate<DDim1, DDim2, DDim3> const x) const noexcept
        {
            return m_eval_func1(ddc::get<DDim1>(x)) * m_eval_func2(ddc::get<DDim2>(x))
                   * m_eval_func3(ddc::get<DDim3>(x));
        }

        template <class DDim1, class DDim2, class DDim3>
        void operator()(
                ddc::ChunkSpan<double, ddc::DiscreteDomain<DDim1, DDim2, DDim3>> chunk) const
        {
            ddc::DiscreteDomain<DDim1, DDim2, DDim3> const domain = chunk.domain();

            for (ddc::DiscreteElement<DDim1> const i : ddc::DiscreteDomain<DDim1>(domain)) {
                for (ddc::DiscreteElement<DDim2> const j : ddc::DiscreteDomain<DDim2>(domain)) {
                    for (ddc::DiscreteElement<DDim3> const k : ddc::DiscreteDomain<DDim3>(domain)) {
                        chunk(i, j, k) = m_eval_func1(ddc::coordinate(i))
                                         * m_eval_func2(ddc::coordinate(j))
                                         * m_eval_func3(ddc::coordinate(k));
                    }
                }
            }
        }

        KOKKOS_FUNCTION double deriv(
                double const x,
                double const y,
                double const z,
                int const derivative_x,
                int const derivative_y,
                int const derivative_z) const noexcept
        {
            return m_eval_func1.deriv(x, derivative_x) * m_eval_func2.deriv(y, derivative_y)
                   * m_eval_func3.deriv(z, derivative_z);
        }

        template <class DDim1, class DDim2, class DDim3>
        KOKKOS_FUNCTION double deriv(
                ddc::Coordinate<DDim1, DDim2, DDim3> const x,
                int const derivative_x,
                int const derivative_y,
                int const derivative_z) const noexcept
        {
            return m_eval_func1.deriv(ddc::get<DDim1>(x), derivative_x)
                   * m_eval_func2.deriv(ddc::get<DDim2>(x), derivative_y)
                   * m_eval_func3.deriv(ddc::get<DDim3>(x), derivative_z);
        }

        template <class DDim1, class DDim2, class DDim3>
        void deriv(
                ddc::ChunkSpan<double, ddc::DiscreteDomain<DDim1, DDim2, DDim3>> chunk,
                int const derivative_x,
                int const derivative_y,
                int const derivative_z) const
        {
            ddc::DiscreteDomain<DDim1, DDim2, DDim3> const domain = chunk.domain();

            for (ddc::DiscreteElement<DDim1> const i : ddc::DiscreteDomain<DDim1>(domain)) {
                for (ddc::DiscreteElement<DDim2> const j : ddc::DiscreteDomain<DDim2>(domain)) {
                    for (ddc::DiscreteElement<DDim3> const k : ddc::DiscreteDomain<DDim3>(domain)) {
                        chunk(i, j, k) = m_eval_func1.deriv(ddc::coordinate(i), derivative_x)
                                         * m_eval_func2.deriv(ddc::coordinate(j), derivative_y)
                                         * m_eval_func3.deriv(ddc::coordinate(k), derivative_z);
                    }
                }
            }
        }

        KOKKOS_FUNCTION double max_norm(int diff1 = 0, int diff2 = 0, int diff3 = 0) const
        {
            return m_eval_func1.max_norm(diff1) * m_eval_func2.max_norm(diff2)
                   * m_eval_func3.max_norm(diff3);
        }
    };
};
