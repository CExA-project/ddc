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
extern "C" int dpttrf_(int const* n, double* d, double* e, int* info);
extern "C" int dpttrs_(
        int const* n,
        int const* nrhs,
        double* d,
        double* e,
        double* b,
        int const* ldb,
        int* info);

template <class ExecSpace>
class Matrix_PDS_Tridiag : public Matrix
{
    /*
     * Represents a real symmetric positive definite matrix
     * stored in a block format
     * */
protected:
    Kokkos::View<double*, Kokkos::HostSpace> m_d; // diagonal
    Kokkos::View<double*, Kokkos::HostSpace> m_l; // lower diagonal

public:
    Matrix_PDS_Tridiag(int const mat_size)
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
        int info;
        int const n = get_size();
        dpttrf_(&n, m_d.data(), m_l.data(), &info);
        return info;
    }

public:
	int solve_inplace_method(ddc::DSpan2D_stride b, char const transpose) const override
    {
		assert(b.stride(0) == 1);
        int const n_equations = b.extent(1);
        int const stride = b.stride(1);

		Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space>
		                  b_view(b.data_handle(), Kokkos::LayoutStride(get_size(), 1, n_equations, stride));

		auto const size_proxy = get_size();
        auto d_device = create_mirror_view_and_copy(ExecSpace(), m_d);
        auto l_device = create_mirror_view_and_copy(ExecSpace(), m_l);
        Kokkos::parallel_for(
                "pbtrs",
                Kokkos::RangePolicy<ExecSpace>(0, n_equations),
                KOKKOS_LAMBDA(const int i) {
                    auto b_slice = Kokkos::subview(b_view, Kokkos::ALL, i);

                    for (int j = 1; j < size_proxy; ++j) {
                        b_slice(j) -= b_slice(j - 1) * d_device(j - 1);
                    }
                    b_slice(size_proxy - 1) /= d_device(size_proxy - 1);
                    for (int j = size_proxy - 2; j >= 0; --j) {
                        b_slice(j) = b_slice(j) / d_device(j) - b_slice(j + 1) * l_device(j);
                    }
                    int info;
                });
        return 0;
    }
};

} // namespace ddc::detail
