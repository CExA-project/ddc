#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "view.hpp"

namespace ddc::detail {

class Matrix
{
    int m_n;

public:
    explicit Matrix(const int mat_size) : m_n(mat_size) {}

    virtual ~Matrix() = default;

    virtual void reset() const = 0;

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
        assert(int(b.extent(0)) == m_n);
        int const info = solve_inplace_method(b.data_handle(), 'N', 1);

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return b;
    }

    virtual ddc::DSpan1D solve_transpose_inplace(ddc::DSpan1D const b) const
    {
        assert(int(b.extent(0)) == m_n);
        int const info = solve_inplace_method(b.data_handle(), 'T', 1);

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return b;
    }

    virtual ddc::DSpan2D solve_multiple_inplace(ddc::DSpan2D const bx) const
    {
        assert(int(bx.extent(1)) == m_n);
        int const info = solve_inplace_method(bx.data_handle(), 'N', bx.extent(0));

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }

    template <class MdspanType>
    MdspanType solve_multiple_inplace2(MdspanType const bx) const
    {
        assert(int(bx.extent(0)) == m_n);
        int const info = solve_inplace_method(
                bx.data_handle(),
                'N',
                bx.extent(1),
                std::max(bx.stride(0), bx.stride(1)));

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }

    virtual ddc::DSpan2D solve_multiple_transpose_inplace(ddc::DSpan2D const bx) const
    {
        assert(int(bx.extent(1)) == m_n);
        int const info = solve_inplace_method(bx.data_handle(), 'T', bx.extent(0));

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }

    template <class Layout, class... Args>
    Kokkos::View<double**, Layout, Args...> solve_multiple_rhs_inplace(
            Kokkos::View<double**, Layout, Args...> const bx) const
    {
        assert(int(bx.extent(0)) == m_n);
        int const info = solve_inplace_method(
                bx.data(),
                'N',
                bx.extent(1),
                std::max(bx.stride_0(), bx.stride_1()));

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

    virtual int solve_inplace_method(
            double* const b,
            char const transpose,
            int const n_equations,
            int const stride) const = 0;

    int solve_inplace_method(double* const b, char const transpose, int const n_equations) const
    {
        return solve_inplace_method(b, transpose, n_equations, get_size());
    };
};

} // namespace ddc::detail
