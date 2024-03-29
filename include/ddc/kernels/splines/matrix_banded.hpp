// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>

#include "matrix.hpp"

namespace ddc::detail {
extern "C" int dgbtrf_(
        int const* m,
        int const* n,
        int const* kl,
        int const* ku,
        double* a_b,
        int const* lda_b,
        int* ipiv,
        int* info);
extern "C" int dgbtrs_(
        char const* trans,
        int const* n,
        int const* kl,
        int const* ku,
        int const* nrhs,
        double* a_b,
        int const* lda_b,
        int* ipiv,
        double* b,
        int const* ldb,
        int* info);

template <class ExecSpace>
class Matrix_Banded : public Matrix
{
protected:
    int const m_kl; // no. of subdiagonals
    int const m_ku; // no. of superdiagonals
    int const m_c; // no. of columns in q
    Kokkos::View<int*, Kokkos::HostSpace> m_ipiv; // pivot indices
    // TODO: double**
    Kokkos::View<double*, Kokkos::HostSpace> m_q; // banded matrix representation

public:
    Matrix_Banded(int const mat_size, int const kl, int const ku)
        : Matrix(mat_size)
        , m_kl(kl)
        , m_ku(ku)
        , m_c(2 * kl + ku + 1)
        , m_ipiv("ipiv", mat_size)
        , m_q("q", m_c * mat_size)
    {
        assert(mat_size > 0);
        assert(m_kl >= 0);
        assert(m_ku >= 0);
        assert(m_kl <= mat_size);
        assert(m_ku <= mat_size);

        /*
         * Given the linear system A*x=b, we assume that A is a square (n by n)
         * with ku super-diagonals and kl sub-diagonals.
         * All non-zero elements of A are stored in the rectangular matrix q, using
         * the format required by DGBTRF (LAPACK): diagonals of A are rows of q.
         * q has 2*kl rows for the subdiagonals, 1 row for the diagonal, and ku rows
         * for the superdiagonals. (The kl additional rows are needed for pivoting.)
         * The term A(i,j) of the full matrix is stored in q(i-j+2*kl+1,j).
         */

        Kokkos::deep_copy(m_q, 0.);
    }

    double get_element(int const i, int const j) const override
    {
        if (i >= std::max(0, j - m_ku) && i < std::min(get_size(), j + m_kl + 1)) {
            return m_q(j * m_c + m_kl + m_ku + i - j);
        } else {
            return 0.0;
        }
    }

    void set_element(int const i, int const j, double const aij) override
    {
        if (i >= std::max(0, j - m_ku) && i < std::min(get_size(), j + m_kl + 1)) {
            m_q(j * m_c + m_kl + m_ku + i - j) = aij;
        } else {
            assert(std::fabs(aij) < 1e-20);
        }
    }

protected:
    int factorize_method() override
    {
        int info;
        int const n = get_size();
        dgbtrf_(&n, &n, &m_kl, &m_ku, m_q.data(), &m_c, m_ipiv.data(), &info);
        return info;
    }
    int solve_inplace_method(ddc::DSpan2D_stride b, char const transpose) const override
    {
        assert(b.stride(0) == 1);
        int const n_equations = b.extent(1);
        int const stride = b.stride(1);

        Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space>
                b_view(b.data_handle(), Kokkos::LayoutStride(get_size(), 1, n_equations, stride));
        auto b_host = create_mirror_view(Kokkos::DefaultHostExecutionSpace(), b_view);
        for (int i = 0; i < n_equations; ++i) {
            Kokkos::deep_copy(
                    Kokkos::subview(b_host, Kokkos::ALL, i),
                    Kokkos::subview(b_view, Kokkos::ALL, i));
        }
        int info;
        int const n = get_size();
        dgbtrs_(&transpose,
                &n,
                &m_kl,
                &m_ku,
                &n_equations,
                m_q.data(),
                &m_c,
                m_ipiv.data(),
                b_host.data(),
                &stride,
                &info);
        for (int i = 0; i < n_equations; ++i) {
            Kokkos::deep_copy(
                    Kokkos::subview(b_view, Kokkos::ALL, i),
                    Kokkos::subview(b_host, Kokkos::ALL, i));
        }
        return info;
    }
};

} // namespace ddc::detail
