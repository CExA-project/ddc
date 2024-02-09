#ifndef MATRIX_PDS_BANDED_H
#define MATRIX_PDS_BANDED_H

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
    }

    void reset() const override
    {
        Kokkos::parallel_for(
                "fill_q",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {get_size(), m_kd + 1}),
                KOKKOS_CLASS_LAMBDA(const int i, const int j) { m_q(i, j) = 0; });
    }

    KOKKOS_FUNCTION double get_element(int i, int j) const override
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
    KOKKOS_FUNCTION void set_element(int i, int j, double const aij) const override
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
    KOKKOS_FUNCTION int tbsv(
            char const uplo,
            char const trans,
            char const diag,
            int const n,
            int const k,
            Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space> const a,
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
    }
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

public:
    int solve_inplace_method(double* const b, char const, int const n_equations, int const stride)
            const override
    {
        Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>
                b_view(b, get_size(), n_equations);

        Kokkos::parallel_for(
                "pbtrs",
                Kokkos::RangePolicy<ExecSpace>(0, n_equations),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>
                            b_slice = Kokkos::subview(b_view, Kokkos::ALL, i);

                    int info;
                    info = tbsv('L', 'N', 'N', get_size(), m_kd, m_q, m_kd, b_slice, 1);
                    info = tbsv('L', 'T', 'N', get_size(), m_kd, m_q, m_kd, b_slice, 1);
                });
        return 0;
    }
};

} // namespace ddc::detail
#endif // MATRIX_SYMMETRIC_BANDED_H
