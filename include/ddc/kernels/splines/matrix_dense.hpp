// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>

#include "matrix.hpp"

namespace ddc::detail {
extern "C" int dgetrf_(int const* m, int const* n, double* a, int const* lda, int* ipiv, int* info);
extern "C" int dgetrs_(
        char const* trans,
        int const* n,
        int const* nrhs,
        double* a,
        int const* lda,
        int* ipiv,
        double* b,
        int const* ldb,
        int* info);

template <class ExecSpace>
class Matrix_Dense : public Matrix
{
protected:
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> m_a;
    Kokkos::View<int*, Kokkos::HostSpace> m_ipiv;

public:
    explicit Matrix_Dense(int const mat_size)
        : Matrix(mat_size)
        , m_a("a", mat_size, mat_size)
        , m_ipiv("ipiv", mat_size)
    {
        assert(mat_size > 0);

        Kokkos::deep_copy(m_a, 0.);
    }

    double get_element(int const i, int const j) const override
    {
        assert(i < get_size());
        assert(j < get_size());
        return m_a(i, j);
    }

    void set_element(int const i, int const j, double const aij) override
    {
        m_a(i, j) = aij;
    }

private:
    int factorize_method() override
    {
        int info;
        int const n = get_size();
        dgetrf_(&n, &n, m_a.data(), &n, m_ipiv.data(), &info);
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
        dgetrs_(&transpose,
                &n,
                &n_equations,
                m_a.data(),
                &n,
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
