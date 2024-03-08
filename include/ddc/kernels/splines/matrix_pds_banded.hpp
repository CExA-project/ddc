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
    Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>
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
            KOKKOS_IF_ON_HOST(
                    if constexpr (Kokkos::SpaceAccessibility<
                                          Kokkos::DefaultHostExecutionSpace,
                                          typename ExecSpace::memory_space>::accessible) {
                        return m_q(0, i);
                    } else {
                        // Inefficient, usage is strongly discouraged
                        double aij;
                        Kokkos::deep_copy(
                                Kokkos::View<double, Kokkos::HostSpace>(&aij),
                                Kokkos::subview(m_q, 0, i));
                        return aij;
                    })
            KOKKOS_IF_ON_DEVICE(return m_q(0, i);)
        }
        if (i > j) {
            // inline swap i<->j
            int tmp = i;
            i = j;
            j = tmp;
        }
        for (int k = 1; k < m_kd + 1; ++k) {
            if (i + k == j) {
                KOKKOS_IF_ON_HOST(
                        if constexpr (Kokkos::SpaceAccessibility<
                                              Kokkos::DefaultHostExecutionSpace,
                                              typename ExecSpace::memory_space>::accessible) {
                            return m_q(k, i);
                        } else {
                            // Inefficient, usage is strongly discouraged
                            double aij;
                            Kokkos::deep_copy(
                                    Kokkos::View<double, Kokkos::HostSpace>(&aij),
                                    Kokkos::subview(m_q, k, i));
                            return aij;
                        })
                KOKKOS_IF_ON_DEVICE(return m_q(k, i);)
            }
        }
        return 0.0;
    }
    void set_element(int i, int j, double const aij) override
    {
        if (i == j) {
            KOKKOS_IF_ON_HOST(
                    if constexpr (Kokkos::SpaceAccessibility<
                                          Kokkos::DefaultHostExecutionSpace,
                                          typename ExecSpace::memory_space>::accessible) {
                        m_q(0, i) = aij;
                    } else {
                        // Inefficient, usage is strongly discouraged
                        Kokkos::deep_copy(
                                Kokkos::subview(m_q, 0, i),
                                Kokkos::View<const double, Kokkos::HostSpace>(&aij));
                    })
            KOKKOS_IF_ON_DEVICE(m_q(0, i) = aij;)
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
                KOKKOS_IF_ON_HOST(
                        if constexpr (Kokkos::SpaceAccessibility<
                                              Kokkos::DefaultHostExecutionSpace,
                                              typename ExecSpace::memory_space>::accessible) {
                            m_q(k, i) = aij;
                        } else {
                            // Inefficient, usage is strongly discouraged
                            Kokkos::deep_copy(
                                    Kokkos::subview(m_q, k, i),
                                    Kokkos::View<const double, Kokkos::HostSpace>(&aij));
                        })
                KOKKOS_IF_ON_DEVICE(m_q(k, i) = aij;)
                return;
            }
        }
        assert(std::fabs(aij) < 1e-20);
        return;
    }

protected:
    int factorize_method() override
    {
        auto q_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_q);
        int info;
        char const uplo = 'L';
        int const n = get_size();
        int const ldab = m_kd + 1;
        dpbtrf_(&uplo, &n, &m_kd, q_host.data(), &ldab, &info);
        Kokkos::deep_copy(m_q, q_host);
        return info;
    }

    int solve_inplace_method(ddc::DSpan2D_stride b, char const) const override
    {
        assert(b.stride(0) == 1);
        int const n_equations = b.extent(1);
        int const stride = b.stride(1);

        auto q_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_q);
        Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space>
                b_view(b.data_handle(), Kokkos::LayoutStride(get_size(), 1, n_equations, stride));
        auto b_host = create_mirror_view(Kokkos::DefaultHostExecutionSpace(), b_view);
        for (int i = 0; i < n_equations; ++i) {
            Kokkos::deep_copy(
                    Kokkos::subview(b_host, Kokkos::ALL, i),
                    Kokkos::subview(b_view, Kokkos::ALL, i));
        }
        int info;
        char const uplo = 'L';
        int const n = get_size();
        int const ldab = m_kd + 1;
        dpbtrs_(&uplo,
                &n,
                &m_kd,
                &n_equations,
                q_host.data(),
                &ldab,
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
