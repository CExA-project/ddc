#ifndef MATRIX_PDS_TRIDIAG_H
#define MATRIX_PDS_TRIDIAG_H

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
    }

    void reset() const override
    {
        Kokkos::parallel_for(
                "fill_d_l",
                Kokkos::RangePolicy<ExecSpace>(0, get_size()),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    m_d(i) = 0;
                    if (i < get_size() - 1) {
                        m_l(i) = 0;
                    }
                });
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
    void set_element(int i, int j, double const aij) const override
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

public:
    int solve_inplace_method(double* const b, char const, int const n_equations, int const stride)
            const override
    {
        Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space>
                b_view(b, Kokkos::LayoutStride(get_size(), 1, n_equations, stride));
        Kokkos::parallel_for(
                "pbtrs",
                Kokkos::RangePolicy<ExecSpace>(0, n_equations),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>
                            b_slice = Kokkos::subview(b_view, Kokkos::ALL, i);

                    for (int j = 1; j < get_size(); ++j) {
                        b_slice(j) -= b_slice(j - 1) * m_l(j - 1);
                    }
                    b_slice(get_size() - 1) /= m_d(get_size() - 1);
                    for (int j = get_size() - 2; j >= 0; --j) {
                        b_slice(j) = b_slice(j) / m_d(j) - b_slice(j + 1) * m_l(j);
                    }
                    int info;
                });
        return 0;
    }
};

} // namespace ddc::detail
#endif // MATRIX_PDS_TRIDIAG_H
