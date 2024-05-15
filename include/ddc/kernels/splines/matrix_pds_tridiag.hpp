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
class MatrixPDSTridiag : public Matrix
{
    /*
     * Represents a real symmetric positive definite matrix
     * stored in a block format
     * */
protected:
    Kokkos::View<double*, Kokkos::HostSpace> m_d; // diagonal
    Kokkos::View<double*, Kokkos::HostSpace> m_l; // lower diagonal

public:
    MatrixPDSTridiag(int const mat_size)
        : Matrix(mat_size)
        , m_d("d", mat_size)
        , m_l("l", mat_size - 1)
    {
        Kokkos::deep_copy(m_d, 0.);
        Kokkos::deep_copy(m_l, 0.);
    }

    double get_element(int i, int j) const override
    {
        if (i == j) {
            return m_d(i);
        }
        if (i > j) {
            // inline swap i<->j
            int tmp = i;
            i = j;
            j = tmp;
        }
        if (i + 1 == j) {
            return m_l(i);
        }
        return 0.0;
    }
    void set_element(int i, int j, double const aij) override
    {
        if (i == j) {
            m_d(i) = aij;
            return;
        }
        if (i > j) {
            // inline swap i<->j
            int tmp = i;
            i = j;
            j = tmp;
        }
        if (i + 1 != j) {
            assert(std::fabs(aij) < 1e-20);
        } else {
            m_l(i) = aij;
        }
    }

protected:
    int factorize_method() override
    {
        int const n = get_size();
        int const info = LAPACKE_dpttrf(n, m_d.data(), m_l.data());
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
        int const n = get_size();
        int const info = LAPACKE_dpttrs(
                LAPACK_COL_MAJOR,
                n,
                n_equations,
                m_d.data(),
                m_l.data(),
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
