#ifndef MATRIX_PERIODIC_BANDED_H
#define MATRIX_PERIODIC_BANDED_H
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <utility>

#include <experimental/mdspan>

#include "matrix.hpp"
#include "matrix_corner_block.hpp"
#include "matrix_dense.hpp"
#include "view.hpp"


namespace ddc::detail {

template <class ExecSpace>
class Matrix_Periodic_Banded : public Matrix_Corner_Block<ExecSpace>
{
    // Necessary because we inherit from a template class, otherwise we should use this-> everywhere
    using Matrix_Corner_Block<ExecSpace>::get_size;
    using Matrix_Corner_Block<ExecSpace>::k;
    using Matrix_Corner_Block<ExecSpace>::nb;
    using Matrix_Corner_Block<ExecSpace>::m_q_block;
    using Matrix_Corner_Block<ExecSpace>::m_delta;
    using Matrix_Corner_Block<ExecSpace>::m_Abm_1_gamma;
    using Matrix_Corner_Block<ExecSpace>::m_lambda;

protected:
    int const kl; // no. of subdiagonals
    int const ku; // no. of superdiagonals

public:
    Matrix_Periodic_Banded(int const n, int const kl, int const ku, std::unique_ptr<Matrix> q)
        : Matrix_Corner_Block<ExecSpace>(
                n,
                std::max(kl, ku),
                std::move(q),
                std::max(kl, ku) + 1,
                std::max(kl, ku))
        , kl(kl)
        , ku(ku)
    {
    }

    double get_element(int const i, int j) const override
    {
        assert(i >= 0);
        assert(i < get_size());
        assert(j >= 0);
        assert(i < get_size());
        if (i >= nb && j < nb) {
            int d = j - i;
            if (d > get_size() / 2)
                d -= get_size();
            if (d < -get_size() / 2)
                d += get_size();

            if (d < -kl || d > ku)
                return 0.0;
            if (d > 0) {
                KOKKOS_IF_ON_HOST(
                        if constexpr (Kokkos::SpaceAccessibility<
                                              Kokkos::DefaultHostExecutionSpace,
                                              typename ExecSpace::memory_space>::accessible) {
                            return m_lambda(j, i - nb);
                        } else {
                            // Inefficient, usage is strongly discouraged
                            double aij;
                            Kokkos::deep_copy(
                                    Kokkos::View<double*, Kokkos::HostSpace>(&aij),
                                    Kokkos::View<double*, typename ExecSpace::memory_space>(
                                            &m_lambda(j, i - nb)));
                            return aij;
                        })
                KOKKOS_IF_ON_DEVICE(return m_lambda(j, i - nb);)
            } else {
                KOKKOS_IF_ON_HOST(
                        if constexpr (Kokkos::SpaceAccessibility<
                                              Kokkos::DefaultHostExecutionSpace,
                                              typename ExecSpace::memory_space>::accessible) {
                            return m_lambda(j - nb + k + 1, i - nb);
                        } else {
                            // Inefficient, usage is strongly discouraged
                            double aij;
                            Kokkos::deep_copy(
                                    Kokkos::View<double*, Kokkos::HostSpace>(&aij),
                                    Kokkos::View<double*, typename ExecSpace::memory_space>(
                                            &m_lambda(j - nb + k + 1, i - nb)));
                            return aij;
                        })
                KOKKOS_IF_ON_DEVICE(return m_lambda(j - nb + k + 1, i - nb);)
            }
        } else {
            return Matrix_Corner_Block<ExecSpace>::get_element(i, j);
        }
    }
    void set_element(int const i, int j, double const aij) const override
    {
        assert(i >= 0);
        assert(i < get_size());
        assert(j >= 0);
        assert(i < get_size());
        if (i >= nb && j < nb) {
            int d = j - i;
            if (d > get_size() / 2)
                d -= get_size();
            if (d < -get_size() / 2)
                d += get_size();

            if (d < -kl || d > ku) {
                assert(std::fabs(aij) < 1e-20);
                return;
            }

            if (d > 0) {
                KOKKOS_IF_ON_HOST(
                        if constexpr (Kokkos::SpaceAccessibility<
                                              Kokkos::DefaultHostExecutionSpace,
                                              typename ExecSpace::memory_space>::accessible) {
                            m_lambda(j, i - nb) = aij;
                        } else {
                            // Inefficient, usage is strongly discouraged
                            Kokkos::deep_copy(
                                    Kokkos::View<double*, typename ExecSpace::memory_space>(
                                            &m_lambda(j, i - nb)),
                                    Kokkos::View<const double*, Kokkos::HostSpace>(&aij));
                        })
                KOKKOS_IF_ON_DEVICE(m_lambda(j, i - nb) = aij;)
            } else {
                KOKKOS_IF_ON_HOST(
                        if constexpr (Kokkos::SpaceAccessibility<
                                              Kokkos::DefaultHostExecutionSpace,
                                              typename ExecSpace::memory_space>::accessible) {
                            m_lambda(j - nb + k + 1, i - nb) = aij;
                        } else {
                            // Inefficient, usage is strongly discouraged
                            Kokkos::deep_copy(
                                    Kokkos::View<double*, typename ExecSpace::memory_space>(
                                            &m_lambda(j - nb + k + 1, i - nb)),
                                    Kokkos::View<const double*, Kokkos::HostSpace>(&aij));
                        })
                KOKKOS_IF_ON_DEVICE(m_lambda(j - nb + k + 1, i - nb) = aij;)
            }
        } else {
            Matrix_Corner_Block<ExecSpace>::set_element(i, j, aij);
        }
    }

protected:
    void calculate_delta_to_factorize() override
    {
        Kokkos::parallel_for(
                "calculate_delta_to_factorize",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {k, k}),
                KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                    double val = 0.0;
                    // Upper diagonals in lambda, lower diagonals in Abm_1_gamma
                    for (int l = 0; l < i + 1; ++l) {
                        val += m_lambda(l, i) * m_Abm_1_gamma(j, l);
                    }
                    // Lower diagonals in lambda, upper diagonals in Abm_1_gamma
                    for (int l = i + 1; l < k + 1; ++l) {
                        int l_full = nb - 1 - k + l;
                        val += m_lambda(l, i) * m_Abm_1_gamma(j, l_full);
                    }
                    m_delta->set_element(i, j, m_delta->get_element(i, j) - val);
					                    std::cout << m_delta->get_element(i, j) << " ";
                });
    }
    ddc::DSpan1D solve_lambda_section(ddc::DSpan1D const v, DView1D const u) const override
    {
        Kokkos::parallel_for(
                "solve_lambda_section",
                Kokkos::RangePolicy<ExecSpace>(0, k),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    // Upper diagonals in lambda
                    for (int j = 0; j <= i; ++j) {
                        Kokkos::atomic_sub(&v(i), m_lambda(j, i) * u(j));
                    }
                    // Lower diagonals in lambda
                    for (int j = i + 1; j < k + 1; ++j) {
                        Kokkos::atomic_sub(&v(i), m_lambda(j, i) * u(nb - 1 - k + j));
                    }
                });
        return v;
    }
    ddc::DSpan1D solve_lambda_section_transpose(ddc::DSpan1D const u, DView1D const v)
            const override
    {
        Kokkos::parallel_for(
                "solve_lambda_section",
                Kokkos::RangePolicy<ExecSpace>(0, k),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    // Upper diagonals in lambda
                    for (int j = 0; j <= i; ++j) {
                        Kokkos::atomic_sub(&u(j), m_lambda(j, i) * v(i));
                    }
                    // Lower diagonals in lambda
                    for (int j = i + 1; j < k + 1; ++j) {
                        Kokkos::atomic_sub(&u(nb - 1 - k + j), m_lambda(j, i) * v(i));
                    }
                });
        return u;
    }
};

} // namespace ddc::detail
#endif // MATRIX_PERIODIC_BANDED_H
