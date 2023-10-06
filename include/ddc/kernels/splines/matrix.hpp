#pragma once
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/ginkgo.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "ginkgo/core/matrix/dense.hpp"

#include "Kokkos_Core_fwd.hpp"
#include "view.hpp"

namespace ddc::detail {
class Matrix
{
public:
    Matrix(const int mat_size) : n(mat_size) {}
    virtual ~Matrix() = default;
    int n;
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
    virtual DSpan1D solve_inplace(DSpan1D const b) const
    {
        assert(int(b.extent(0)) == n);
        int const info = solve_inplace_method(b.data_handle(), 'N', 1);

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return b;
    }
    virtual DSpan1D solve_transpose_inplace(DSpan1D const b) const
    {
        assert(int(b.extent(0)) == n);
        int const info = solve_inplace_method(b.data_handle(), 'T', 1);

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return b;
    }
    virtual DSpan2D solve_multiple_inplace(DSpan2D const bx) const
    {
        assert(int(bx.extent(1)) == n);
        int const info = solve_inplace_method(bx.data_handle(), 'N', bx.extent(0));

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }
    template <class... Args>
    Kokkos::View<double**, Args...> solve_batch_inplace(
            Kokkos::View<double**, Args...> const bx) const
    {
        std::cout << "extent = " << bx.extent(1);
        assert(int(bx.extent(0)) == n);
        int const info = solve_inplace_method(bx.data(), 'N', bx.extent(1));

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }
    int get_size() const
    {
        return n;
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
    virtual int solve_inplace_method(double* b, char transpose, int n_equations) const = 0;
};
} // namespace ddc::detail
