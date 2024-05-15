// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <memory>

#include <string.h>

#include "matrix.hpp"

namespace ddc::detail {

template <class ExecSpace>
class MatrixPDSBanded : public Matrix
{
    /*
     * Represents a real symmetric positive definite matrix
     * stored in a block format
     * */
protected:
    int const m_kd; // no. of columns in q
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>
            m_q; // pds banded matrix representation

public:
    MatrixPDSBanded(int const mat_size, int const kd)
        : Matrix(mat_size)
        , m_kd(kd)
        , m_q("q", kd + 1, mat_size)
    {
        Kokkos::deep_copy(m_q, 0.);
    }

    double get_element(int i, int j) const override
    {
        if (i == j) {
            return m_q(0, i);
        }
        if (i > j) {
            // inline swap i<->j
            int tmp = i;
            i = j;
            j = tmp;
        }
        for (int k = 1; k < m_kd + 1; ++k) {
            if (i + k == j) {
                return m_q(k, i);
            }
        }
        return 0.0;
    }
    void set_element(int i, int j, double const aij) override
    {
        if (i == j) {
            m_q(0, i) = aij;
            return;
        }
        if (i > j) {
            // inline swap i<->j
            int tmp = i;
            i = j;
            j = tmp;
        }
        for (int k = 1; k < m_kd + 1; ++k) {
            if (i + k == j) {
                m_q(k, i) = aij;
                return;
            }
        }
        assert(std::fabs(aij) < 1e-20);
        return;
    }

protected:
    int factorize_method() override
    {
        char const uplo = 'L';
        int const n = get_size();
        int const ldab = m_kd + 1;
        int const info = LAPACKE_dpbtrf(LAPACK_COL_MAJOR, uplo, n, m_kd, m_q.data(), ldab);
        return info;
    }

    int solve_inplace_method(ddc::DSpan2D_stride b, char const) const override
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
        char const uplo = 'L';
        int const n = get_size();
        int const ldab = m_kd + 1;
        int const info = LAPACKE_dpbtrs(
                LAPACK_COL_MAJOR,
                uplo,
                n,
                m_kd,
                n_equations,
                m_q.data(),
                ldab,
                b_host.data(),
                stride);
        for (int i = 0; i < n_equations; ++i) {
            Kokkos::deep_copy(
                    Kokkos::subview(b_view, Kokkos::ALL, i),
                    Kokkos::subview(b_host, Kokkos::ALL, i));
        }
        return info;
    }
};

} // namespace ddc::detail
