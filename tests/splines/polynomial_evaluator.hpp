// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

struct PolynomialEvaluator
{
    template <class DDim, std::size_t Degree>
    class Evaluator
    {
    public:
        using Dim = DDim;

    private:
        std::array<ddc::Real, Degree + 1> m_coeffs;

        ddc::Real m_xn;

    public:
        template <class Domain>
        explicit Evaluator(Domain domain)
            : m_xn(std::max(std::abs(rmin(domain)), std::abs(rmax(domain))))
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution dis(0., 1.);
            for (int i(0); i < Degree + 1; ++i) {
                m_coeffs[i] = dis(gen);
            }
        }

        KOKKOS_FUNCTION ddc::Real operator()(ddc::Real const x) const noexcept
        {
            return eval(x, 0);
        }

        void operator()(ddc::ChunkSpan<ddc::Real, ddc::DiscreteDomain<DDim>> chunk) const
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

        void deriv(ddc::ChunkSpan<ddc::Real, ddc::DiscreteDomain<DDim>> chunk, int const derivative)
                const
        {
            ddc::DiscreteDomain<DDim> const domain = chunk.domain();

            for (ddc::DiscreteElement<DDim> const i : domain) {
                chunk(i) = eval(ddc::coordinate(i), derivative);
            }
        }

        KOKKOS_FUNCTION ddc::Real max_norm(int diff = 0) const
        {
            return Kokkos::abs(deriv(m_xn, diff));
        }

    private:
        KOKKOS_FUNCTION ddc::Real eval(ddc::Real const x, int const derivative) const
        {
            ddc::Real result(0.0);
            int const start = derivative < 0 ? 0 : derivative;
            for (int i(start); i < Degree + 1; ++i) {
                ddc::Real const v = ddc::Real(falling_factorial(i, derivative))
                                    * Kokkos::pow(x, i - derivative);
                result += m_coeffs[i] * v;
            }
            return result;
        }

        KOKKOS_FUNCTION ddc::Real falling_factorial(int i, int d) const
        {
            ddc::Real c = 1.0;
            if (d >= 0) {
                for (int k(0); k < d; ++k) {
                    c *= (i - k);
                }
            } else {
                for (int k(-1); k > d - 1; --k) {
                    c /= (i - k);
                }
            }
            return c;
        }
    };
};
