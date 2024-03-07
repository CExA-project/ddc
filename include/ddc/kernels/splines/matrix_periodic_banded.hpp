#pragma once

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
    using Matrix_Corner_Block<ExecSpace>::m_k;
    using Matrix_Corner_Block<ExecSpace>::m_nb;
    using Matrix_Corner_Block<ExecSpace>::m_q_block;
    using Matrix_Corner_Block<ExecSpace>::m_delta;
    using Matrix_Corner_Block<ExecSpace>::m_Abm_1_gamma;
    using Matrix_Corner_Block<ExecSpace>::m_lambda;

protected:
    int const m_kl; // no. of subdiagonals
    int const m_ku; // no. of superdiagonals

public:
    Matrix_Periodic_Banded(int const n, int const m_kl, int const m_ku, std::unique_ptr<Matrix> q)
        : Matrix_Corner_Block<ExecSpace>(
                n,
                std::max(m_kl, m_ku),
                std::move(q),
                std::max(m_kl, m_ku),
                std::max(m_kl, m_ku) + 1)
        , m_kl(m_kl)
        , m_ku(m_ku)
    {
    }

    double get_element(int const i, int j) const override
    {
        assert(i >= 0);
        assert(i < get_size());
        assert(j >= 0);
        assert(i < get_size());
        if (i >= m_nb && j < m_nb) {
            int d = j - i;
            if (d > get_size() / 2)
                d -= get_size();
            if (d < -get_size() / 2)
                d += get_size();

            if (d < -m_kl || d > m_ku)
                return 0.0;
            if (d > 0) {
                if constexpr (Kokkos::SpaceAccessibility<
                                      Kokkos::DefaultHostExecutionSpace,
                                      typename ExecSpace::memory_space>::accessible) {
                    return m_lambda(i - m_nb, j);
                } else {
                    // Inefficient, usage is strongly discouraged
                    double aij;
                    Kokkos::deep_copy(
                            Kokkos::View<double, Kokkos::HostSpace>(&aij),
                            Kokkos::subview(m_lambda, i - m_nb, j));
                    return aij;
                }
            } else {
                if constexpr (Kokkos::SpaceAccessibility<
                                      Kokkos::DefaultHostExecutionSpace,
                                      typename ExecSpace::memory_space>::accessible) {
                    return m_lambda(i - m_nb, j - m_nb + m_k + 1);
                } else {
                    // Inefficient, usage is strongly discouraged
                    double aij;
                    Kokkos::deep_copy(
                            Kokkos::View<double, Kokkos::HostSpace>(&aij),
                            Kokkos::subview(m_lambda, i - m_nb, j - m_nb + m_k + 1));
                    return aij;
                }
            }
        } else {
            return Matrix_Corner_Block<ExecSpace>::get_element(i, j);
        }
    }
    void set_element(int const i, int j, double const aij) override
    {
        assert(i >= 0);
        assert(i < get_size());
        assert(j >= 0);
        assert(i < get_size());
        if (i >= m_nb && j < m_nb) {
            int d = j - i;
            if (d > get_size() / 2)
                d -= get_size();
            if (d < -get_size() / 2)
                d += get_size();

            if (d < -m_kl || d > m_ku) {
                assert(std::fabs(aij) < 1e-20);
                return;
            }

            if (d > 0) {
                if constexpr (Kokkos::SpaceAccessibility<
                                      Kokkos::DefaultHostExecutionSpace,
                                      typename ExecSpace::memory_space>::accessible) {
                    m_lambda(i - m_nb, j) = aij;
                } else {
                    // Inefficient, usage is strongly discouraged
                    Kokkos::deep_copy(
                            Kokkos::subview(m_lambda, i - m_nb, j),
                            Kokkos::View<const double, Kokkos::HostSpace>(&aij));
                }
            } else {
                if constexpr (Kokkos::SpaceAccessibility<
                                      Kokkos::DefaultHostExecutionSpace,
                                      typename ExecSpace::memory_space>::accessible) {
                    m_lambda(i - m_nb, j - m_nb + m_k + 1) = aij;
                } else {
                    // Inefficient, usage is strongly discouraged
                    Kokkos::deep_copy(
                            Kokkos::subview(m_lambda, i - m_nb, j - m_nb + m_k + 1),
                            Kokkos::View<const double, Kokkos::HostSpace>(&aij));
                }
            }
        } else {
            Matrix_Corner_Block<ExecSpace>::set_element(i, j, aij);
        }
    }

public:
    void calculate_delta_to_factorize() override
    {
        auto delta_proxy = *m_delta;
        auto lambda_host = Kokkos::
                create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_lambda);
        auto Abm_1_gamma_host = Kokkos::
                create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_Abm_1_gamma);
        Kokkos::parallel_for(
                "calculate_delta_to_factorize",
                Kokkos::MDRangePolicy<
                        Kokkos::DefaultHostExecutionSpace,
                        Kokkos::Rank<2>>({0, 0}, {m_k, m_k}),
                [&](const int i, const int j) {
                    double val = 0.0;
                    // Upper diagonals in lambda, lower diagonals in Abm_1_gamma
                    for (int l = 0; l < i + 1; ++l) {
                        val += lambda_host(i, l) * Abm_1_gamma_host(l, j);
                    }
                    // Lower diagonals in lambda, upper diagonals in Abm_1_gamma
                    for (int l = i + 1; l < m_k + 1; ++l) {
                        int l_full = m_nb - 1 - m_k + l;
                        val += lambda_host(i, l) * Abm_1_gamma_host(l_full, j);
                    }
                    auto tmp = delta_proxy.get_element(i, j);
                    delta_proxy.set_element(i, j, tmp - val);
                });
    }

    ddc::DSpan2D_stride solve_lambda_section(
            ddc::DSpan2D_stride const v,
            ddc::DView2D_stride const u) const override
    {
        Kokkos::parallel_for(
                "solve_lambda_section",
                Kokkos::TeamPolicy<ExecSpace>(v.extent(1), Kokkos::AUTO),
                KOKKOS_CLASS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, m_k),
                            [&](const int i) {
                                /// Upper diagonals in lambda
                                for (int l = 0; l <= i; ++l) {
                                    Kokkos::atomic_sub(&v(i, j), m_lambda(i, l) * u(l, j));
                                }
                                // Lower diagonals in lambda
                                for (int l = i + 1; l < m_k + 1; ++l) {
                                    Kokkos::atomic_sub(
                                            &v(i, j),
                                            m_lambda(i, l) * u(m_nb - 1 - m_k + l, j));
                                }
                            });
                });
        return v;
    }

    ddc::DSpan2D_stride solve_lambda_section_transpose(
            ddc::DSpan2D_stride const u,
            ddc::DView2D_stride const v) const override
    {
        Kokkos::parallel_for(
                "solve_lambda_section_transpose",
                Kokkos::TeamPolicy<ExecSpace>(v.extent(1), Kokkos::AUTO),
                KOKKOS_CLASS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, m_k),
                            [&](const int i) {
                                /// Upper diagonals in lambda
                                for (int l = 0; l <= i; ++l) {
                                    Kokkos::atomic_sub(&u(l, j), m_lambda(i, l) * v(i, j));
                                }
                                // Lower diagonals in lambda
                                for (int l = i + 1; l < m_k + 1; ++l) {
                                    Kokkos::atomic_sub(
                                            &u(m_nb - 1 - m_k + l, j),
                                            m_lambda(i, l) * v(i, j));
                                }
                            });
                });
        return u;
    }
};

} // namespace ddc::detail
