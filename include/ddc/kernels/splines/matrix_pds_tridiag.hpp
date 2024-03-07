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
    Kokkos::View<double*, typename ExecSpace::memory_space> m_d; // diagonal
    Kokkos::View<double*, typename ExecSpace::memory_space> m_l; // lower diagonal

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
            if constexpr (Kokkos::SpaceAccessibility<
                                  Kokkos::DefaultHostExecutionSpace,
                                  typename ExecSpace::memory_space>::accessible) {
                return m_d(i);
            } else {
                // Inefficient, usage is strongly discouraged
                double aij;
                Kokkos::deep_copy(
                        Kokkos::View<double, Kokkos::HostSpace>(&aij),
                        Kokkos::subview(m_d, i));
                return aij;
            }
        }
        if (i > j) {
            // inline swap i<->j
            int tmp = i;
            i = j;
            j = tmp;
        }
        if (i + 1 == j) {
            if constexpr (Kokkos::SpaceAccessibility<
                                  Kokkos::DefaultHostExecutionSpace,
                                  typename ExecSpace::memory_space>::accessible) {
                return m_l(i);
            } else {
                // Inefficient, usage is strongly discouraged
                double aij;
                Kokkos::deep_copy(
                        Kokkos::View<double, Kokkos::HostSpace>(&aij),
                        Kokkos::subview(m_l, i));
                return aij;
            }
        }
        return 0.0;
    }
    void set_element(int i, int j, double const aij) override
    {
        if (i == j) {
            if constexpr (Kokkos::SpaceAccessibility<
                                  Kokkos::DefaultHostExecutionSpace,
                                  typename ExecSpace::memory_space>::accessible) {
                m_d(i) = aij;
            } else {
                // Inefficient, usage is strongly discouraged
                Kokkos::deep_copy(
                        Kokkos::subview(m_d, i),
                        Kokkos::View<const double, Kokkos::HostSpace>(&aij));
            }
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
            if constexpr (Kokkos::SpaceAccessibility<
                                  Kokkos::DefaultHostExecutionSpace,
                                  typename ExecSpace::memory_space>::accessible) {
                m_l(i) = aij;
            } else {
                // Inefficient, usage is strongly discouraged
                Kokkos::deep_copy(
                        Kokkos::subview(m_l, i),
                        Kokkos::View<const double, Kokkos::HostSpace>(&aij));
            }
        }
    }

protected:
    int factorize_method() override
    {
        auto d_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_d);
        auto l_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_l);
        int info;
        int const n = get_size();
        dpttrf_(&n, d_host.data(), l_host.data(), &info);
        Kokkos::deep_copy(m_d, d_host);
        Kokkos::deep_copy(m_l, l_host);
        return info;
    }

    int solve_inplace_method(double* const b, char const, int const n_equations, int const stride)
            const override
    {
        auto d_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_d);
        auto l_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_l);
        Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space>
                b_view(b, Kokkos::LayoutStride(get_size(), 1, n_equations, stride));
        auto b_host = create_mirror_view(Kokkos::DefaultHostExecutionSpace(), b_view);
        for (int i = 0; i < n_equations; ++i) {
            Kokkos::deep_copy(
                    Kokkos::subview(b_host, Kokkos::ALL, i),
                    Kokkos::subview(b_view, Kokkos::ALL, i));
        }
        int info;
        int const n = get_size();
        dpttrs_(&n, &n_equations, d_host.data(), l_host.data(), b_host.data(), &stride, &info);
        for (int i = 0; i < n_equations; ++i) {
            Kokkos::deep_copy(
                    Kokkos::subview(b_view, Kokkos::ALL, i),
                    Kokkos::subview(b_host, Kokkos::ALL, i));
        }
        return info;
    }
};

} // namespace ddc::detail
