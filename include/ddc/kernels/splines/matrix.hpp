// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "view.hpp"

namespace ddc::detail {

class Matrix
{
private:
    int m_n;

public:
    explicit Matrix(const int mat_size) : m_n(mat_size) {}

    virtual ~Matrix() = default;

    virtual double get_element(int i, int j) const = 0;

    virtual void set_element(int i, int j, double aij) = 0;

    virtual void factorize()
    {
        int const info = factorize_method();

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        } else if (info > 0) {
            std::cerr << "U(" << info << "," << info << ") is exactly zero.";
            std::cerr << " The factorization has been completed, but the factor";
            std::cerr << " U is exactly singular, and division by zero will occur "
                         "if "
                         "it is used to";
            std::cerr << " solve a system of equations.";
            // TODO: Add LOG_FATAL_ERROR
        }
    }

    virtual ddc::DSpan1D solve_inplace(ddc::DSpan1D const b) const
    {
        ddc::DSpan2D_left b_2d(b.data_handle(), b.extent(0), 1);
        solve_inplace(b_2d);
        return b;
    }

    virtual ddc::DSpan1D solve_transpose_inplace(ddc::DSpan1D const b) const
    {
        ddc::DSpan2D_left b_2d(b.data_handle(), b.extent(0), 1);
        solve_transpose_inplace(b_2d);
        return b;
    }

    virtual ddc::DSpan2D_stride solve_inplace(ddc::DSpan2D_stride const bx) const
    {
        assert(int(bx.extent(0)) == m_n);
        int const info = solve_inplace_method(bx, 'N');

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }

    virtual ddc::DSpan2D_stride solve_transpose_inplace(ddc::DSpan2D_stride const bx) const
    {
        assert(int(bx.extent(0)) == m_n);
        int const info = solve_inplace_method(bx, 'T');

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }

    int get_size() const
    {
        return m_n;
    }

    std::ostream& operator<<(std::ostream& os)
    {
        int const n = get_size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                os << std::fixed << std::setprecision(3) << std::setw(10) << get_element(i, j);
            }
            os << std::endl;
        }
        return os;
    }

protected:
    virtual int factorize_method() = 0;

    virtual int solve_inplace_method(ddc::DSpan2D_stride b, char transpose) const = 0;
};

} // namespace ddc::detail
