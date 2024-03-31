// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <utility>

#include <experimental/mdspan>

#include <string.h>

#include "matrix.hpp"
#include "matrix_dense.hpp"
#include "view.hpp"

namespace ddc::detail {

template <class ExecSpace>
class Matrix_Corner_Block : public Matrix
{
protected:
    int const m_k; // small block size
    int const m_nb; // main block matrix size
    //-------------------------------------
    //
    //    q = | q_block | gamma |
    //        |  lambda | delta |
    //
    //-------------------------------------
    std::shared_ptr<Matrix> m_q_block;
    std::shared_ptr<Matrix_Dense<ExecSpace>> m_delta;
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> m_Abm_1_gamma;
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> m_lambda;

public:
    Matrix_Corner_Block(int const n, int const k, std::unique_ptr<Matrix> q)
        : Matrix(n)
        , m_k(k)
        , m_nb(n - k)
        , m_q_block(std::move(q))
        , m_delta(new Matrix_Dense<ExecSpace>(k))
        , m_Abm_1_gamma("Abm_1_gamma", m_nb, m_k)
        , m_lambda("lambda", m_k, m_nb)
    {
        assert(n > 0);
        assert(m_k >= 0);
        assert(m_k <= get_size());
        assert(m_nb == m_q_block->get_size());

        Kokkos::deep_copy(m_Abm_1_gamma, 0.);
        Kokkos::deep_copy(m_lambda, 0.);
    }

    virtual double get_element(int const i, int const j) const override
    {
        assert(i >= 0);
        assert(i < get_size());
        assert(j >= 0);
        assert(i < get_size());
        if (i < m_nb && j < m_nb) {
            return m_q_block->get_element(i, j);
        } else if (i >= m_nb && j >= m_nb) {
            return m_delta->get_element(i - m_nb, j - m_nb);
        } else if (j >= m_nb) {
            return m_Abm_1_gamma(i, j - m_nb);
        } else {
            return m_lambda(i - m_nb, j);
        }
    }
    virtual void set_element(int const i, int const j, double const aij) override
    {
        assert(i >= 0);
        assert(i < get_size());
        assert(j >= 0);
        assert(i < get_size());
        if (i < m_nb && j < m_nb) {
            m_q_block->set_element(i, j, aij);
        } else if (i >= m_nb && j >= m_nb) {
            m_delta->set_element(i - m_nb, j - m_nb, aij);
        } else if (j >= m_nb) {
            m_Abm_1_gamma(i, j - m_nb) = aij;
        } else {
            m_lambda(i - m_nb, j) = aij;
        }
    }
    virtual void factorize() override
    {
        m_q_block->factorize();
        auto Abm_1_gamma_device = create_mirror_view_and_copy(ExecSpace(), m_Abm_1_gamma);
        m_q_block->solve_inplace(
                detail::build_mdspan(Abm_1_gamma_device, std::make_index_sequence<2> {}));
        Kokkos::deep_copy(m_Abm_1_gamma, Abm_1_gamma_device);

        calculate_delta_to_factorize();

        m_delta->factorize();
    }

protected:
    Matrix_Corner_Block(
            int const n,
            int const k,
            std::unique_ptr<Matrix> q,
            int const lambda_size1,
            int const lambda_size2)
        : Matrix(n)
        , m_k(k)
        , m_nb(n - k)
        , m_q_block(std::move(q))
        , m_delta(new Matrix_Dense<ExecSpace>(k))
        , m_Abm_1_gamma("Abm_1_gamma", m_nb, m_k)
        , m_lambda("lambda", lambda_size1, lambda_size2)
    {
        assert(n > 0);
        assert(m_k >= 0);
        assert(m_k <= n);
        assert(m_nb == m_q_block->get_size());
    }

public:
    virtual void calculate_delta_to_factorize()
    {
        auto delta_proxy = *m_delta;
        Kokkos::parallel_for(
                "calculate_delta_to_factorize",
                Kokkos::MDRangePolicy<
                        Kokkos::DefaultHostExecutionSpace,
                        Kokkos::Rank<2>>({0, 0}, {m_k, m_k}),
                [&](const int i, const int j) {
                    double val = 0.0;
                    for (int l = 0; l < m_nb; ++l) {
                        val += m_lambda(i, l) * m_Abm_1_gamma(l, j);
                    }
                    delta_proxy.set_element(i, j, delta_proxy.get_element(i, j) - val);
                });
    }
    virtual ddc::DSpan2D_stride solve_lambda_section(
            ddc::DSpan2D_stride const v,
            ddc::DView2D_stride const u) const
    {
        auto lambda_device = create_mirror_view_and_copy(ExecSpace(), m_lambda);
        auto nb_proxy = m_nb;
        auto k_proxy = m_k;
        Kokkos::parallel_for(
                "solve_lambda_section",
                Kokkos::TeamPolicy<ExecSpace>(v.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, k_proxy),
                            [&](const int i) {
                                // Upper diagonals in lambda
                                for (int l = 0; l < nb_proxy; ++l) {
                                    v(i, j) -= lambda_device(i, l) * u(l, j);
                                }
                            });
                });
        return v;
    }
    virtual ddc::DSpan2D_stride solve_lambda_section_transpose(
            ddc::DSpan2D_stride const u,
            ddc::DView2D_stride const v) const
    {
        auto lambda_device = create_mirror_view_and_copy(ExecSpace(), m_lambda);
        auto nb_proxy = m_nb;
        auto k_proxy = m_k;
        Kokkos::parallel_for(
                "solve_lambda_section_transpose",
                Kokkos::TeamPolicy<ExecSpace>(u.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, nb_proxy),
                            [&](const int i) {
                                // Upper diagonals in lambda
                                for (int l = 0; l < k_proxy; ++l) {
                                    u(i, j) -= lambda_device(l, i) * v(l, j);
                                }
                            });
                });
        return u;
    }
    virtual ddc::DSpan2D_stride solve_gamma_section(
            ddc::DSpan2D_stride const u,
            ddc::DView2D_stride const v) const
    {
        auto Abm_1_gamma_device = create_mirror_view_and_copy(ExecSpace(), m_Abm_1_gamma);
        auto nb_proxy = m_nb;
        auto k_proxy = m_k;
        Kokkos::parallel_for(
                "solve_gamma_section",
                Kokkos::TeamPolicy<ExecSpace>(u.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, nb_proxy),
                            [&](const int i) {
                                // Upper diagonals in lambda
                                for (int l = 0; l < k_proxy; ++l) {
                                    u(i, j) -= Abm_1_gamma_device(i, l) * v(l, j);
                                }
                            });
                });
        return u;
    }
    virtual ddc::DSpan2D_stride solve_gamma_section_transpose(
            ddc::DSpan2D_stride const v,
            ddc::DView2D_stride const u) const
    {
        auto Abm_1_gamma_device = create_mirror_view_and_copy(ExecSpace(), m_Abm_1_gamma);
        auto nb_proxy = m_nb;
        auto k_proxy = m_k;
        Kokkos::parallel_for(
                "solve_gamma_section_transpose",
                Kokkos::TeamPolicy<ExecSpace>(v.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, k_proxy),
                            [&](const int i) {
                                // Upper diagonals in lambda
                                for (int l = 0; l < nb_proxy; ++l) {
                                    v(i, j) -= Abm_1_gamma_device(l, i) * u(l, j);
                                }
                            });
                });
        return v;
    }

protected:
    virtual int factorize_method() override
    {
        return 0;
    }
    virtual int solve_inplace_method(ddc::DSpan2D_stride b, char const transpose) const override
    {
        assert(b.stride(0) == 1);
        int const n_equations = b.extent(1);
        int const stride = b.stride(1);

        std::experimental::layout_stride::mapping<std::experimental::extents<
                size_t,
                std::experimental::dynamic_extent,
                std::experimental::dynamic_extent>>
                layout_mapping_u {
                        std::experimental::dextents<
                                std::size_t,
                                2> {(std::size_t)m_nb, (std::size_t)n_equations},
                        std::array<std::size_t, 2> {1, (std::size_t)stride}};
        std::experimental::layout_stride::mapping<std::experimental::extents<
                size_t,
                std::experimental::dynamic_extent,
                std::experimental::dynamic_extent>>
                layout_mapping_v {
                        std::experimental::dextents<
                                std::size_t,
                                2> {(std::size_t)m_k, (std::size_t)n_equations},
                        std::array<std::size_t, 2> {1, (std::size_t)stride}};

        ddc::DSpan2D_stride const u(b.data_handle(), layout_mapping_u);
        ddc::DSpan2D_stride const v(b.data_handle() + m_nb, layout_mapping_v);

        if (transpose == 'N') {
            m_q_block->solve_inplace(u);

            solve_lambda_section(v, u);

            m_delta->solve_inplace(v);

            solve_gamma_section(u, v);
        } else if (transpose == 'T') {
            solve_gamma_section_transpose(v, u);

            m_delta->solve_transpose_inplace(v);

            solve_lambda_section_transpose(u, v);

            m_q_block->solve_transpose_inplace(u);
        } else {
            return -1;
        }
        return 0;
    }
};

} // namespace ddc::detail
