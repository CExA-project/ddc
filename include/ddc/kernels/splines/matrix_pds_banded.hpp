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
extern "C" int dpbtrf_(
        char const* uplo,
        int const* n,
        int const* kd,
        double* ab,
        int const* ldab,
        int* info);
extern "C" int dpbtrs_(
        char const* uplo,
        int const* n,
        int const* kd,
        int const* nrhs,
        double const* ab,
        int const* ldab,
        double* b,
        int const* ldb,
        int* info);

template <class ExecSpace>
class Matrix_PDS_Banded : public Matrix
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
    Matrix_PDS_Banded(int const mat_size, int const kd)
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
    KOKKOS_FUNCTION int tbsv(
            char const uplo,
            char const trans,
            char const diag,
            int const n,
            int const k,
            Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space> const a,
            int const lda,
            Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space> const x,
            int const incx) const
    {
        if (trans == 'N') {
            for (int j = 0; j < n; ++j) {
                if (x(j) != 0) {
                    x(j) /= a(0, j);
                    for (int i = j + 1; i <= Kokkos::min(n, j + k); ++i) {
                        x(i) -= a(i - j, j) * x(j);
                    }
                }
            }
        } else if (trans == 'T') {
            for (int j = n - 1; j >= 0; --j) {
                for (int i = Kokkos::min(n, j + k); i >= j + 1; --i) {
                    x(j) -= a(i - j, j) * x(i);
                }
                x(j) /= a(0, j);
            }
        }
        return 0;
    }
    int factorize_method() override
    {
        int info;
        char const uplo = 'L';
        int const n = get_size();
        int const ldab = m_kd + 1;
        dpbtrf_(&uplo, &n, &m_kd, m_q.data(), &ldab, &info);
        return info;
    }

public:
    int solve_inplace_method(ddc::DSpan2D_stride b, char const) const override
    {
        assert(b.stride(0) == 1);
        int const n_equations = b.extent(1);
        int const stride = b.stride(1);

        Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space>
                b_view(b.data_handle(), Kokkos::LayoutStride(get_size(), 1, n_equations, stride));

        auto const kd_proxy = m_kd;
        auto const size_proxy = get_size();
        auto q_device = create_mirror_view_and_copy(ExecSpace(), m_q);
        Kokkos::parallel_for(
                "pbtrs",
                Kokkos::RangePolicy<ExecSpace>(0, n_equations),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    auto b_slice = Kokkos::subview(b_view, Kokkos::ALL, i);


                    int info;
                    info
                            = tbsv('L',
                                   'N',
                                   'N',
                                   size_proxy,
                                   kd_proxy,
                                   q_device,
                                   kd_proxy,
                                   b_slice,
                                   1);
                    Kokkos::fence();
                    info
                            = tbsv('L',
                                   'T',
                                   'N',
                                   size_proxy,
                                   kd_proxy,
                                   q_device,
                                   kd_proxy,
                                   b_slice,
                                   1);
                });
        return 0;
    }
};

} // namespace ddc::detail
