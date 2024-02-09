#ifndef MATRIX_BANDED_H
#define MATRIX_BANDED_H
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
    Kokkos::View<int*, typename ExecSpace::memory_space> m_ipiv; // pivot indices
    // TODO: double**
    Kokkos::View<double*, typename ExecSpace::memory_space> m_q; // banded matrix representation

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
    }

    void reset() const override
    {
        Kokkos::parallel_for(
                "fill_q",
                Kokkos::RangePolicy<ExecSpace>(0, get_size()),
                KOKKOS_CLASS_LAMBDA(const int i) { m_q(i) = 0; });
    }

    KOKKOS_FUNCTION double get_element(int const i, int const j) const override
    {
        if (i >= std::max(0, j - m_ku) && i < std::min(get_size(), j + m_kl + 1)) {
            KOKKOS_IF_ON_HOST(
                    if constexpr (Kokkos::SpaceAccessibility<
                                          Kokkos::DefaultHostExecutionSpace,
                                          typename ExecSpace::memory_space>::accessible) {
                        return m_q(j * m_c + m_kl + m_ku + i - j);
                    } else {
                        // Inefficient, usage is strongly discouraged
                        double aij;
                        Kokkos::deep_copy(
                                Kokkos::View<double, Kokkos::HostSpace>(&aij),
                                Kokkos::subview(m_q, j * m_c + m_kl + m_ku + i - j));
                        return aij;
                    })
            KOKKOS_IF_ON_DEVICE(return m_q(j * m_c + m_kl + m_ku + i - j);)

        } else {
            return 0.0;
        }
    }

    KOKKOS_FUNCTION void set_element(int const i, int const j, double const aij) const override
    {
        if (i >= std::max(0, j - m_ku) && i < std::min(get_size(), j + m_kl + 1)) {
            KOKKOS_IF_ON_HOST(
                    if constexpr (Kokkos::SpaceAccessibility<
                                          Kokkos::DefaultHostExecutionSpace,
                                          typename ExecSpace::memory_space>::accessible) {
                        m_q(j * m_c + m_kl + m_ku + i - j) = aij;
                    } else {
                        // Inefficient, usage is strongly discouraged
                        Kokkos::deep_copy(
                                Kokkos::subview(m_q, j * m_c + m_kl + m_ku + i - j),
                                Kokkos::View<const double, Kokkos::HostSpace>(&aij));
                    })
            KOKKOS_IF_ON_DEVICE(m_q(j * m_c + m_kl + m_ku + i - j) = aij;)

        } else {
            assert(std::fabs(aij) < 1e-20);
        }
    }

protected:
    int factorize_method() override
    {
        // TODO : Rewrite using Kokkos-kernels
        auto q_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_q);
        auto ipiv_host = create_mirror_view(Kokkos::DefaultHostExecutionSpace(), m_ipiv);
        int info;
        int const n = get_size();
        dgbtrf_(&n, &n, &m_kl, &m_ku, q_host.data(), &m_c, ipiv_host.data(), &info);
        Kokkos::deep_copy(m_q, q_host);
        Kokkos::deep_copy(m_ipiv, ipiv_host);
        return info;
    }
    int solve_inplace_method(
            double* b,
            char const transpose,
            int const n_equations,
            int const stride) const override
    {
        // TODO : Rewrite using Kokkos-kernels
        auto q_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_q);
        auto ipiv_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_ipiv);
        Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space>
                b_view(b, Kokkos::LayoutStride(get_size(), 1, n_equations, stride));
        auto b_host = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), b_view);
        int info;
        int const n = get_size();
        dgbtrs_(&transpose,
                &n,
                &m_kl,
                &m_ku,
                &n_equations,
                q_host.data(),
                &m_c,
                ipiv_host.data(),
                b_host.data(),
                &stride,
                &info);
        Kokkos::deep_copy(b_view, b_host);
        return info;
    }
};

} // namespace ddc::detail
#endif // MATRIX_BANDED_H
