#ifndef MATRIX_CORNER_BLOCK_H
#define MATRIX_CORNER_BLOCK_H
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
    int const k; // small block size
    int const nb; // main block matrix size
    //-------------------------------------
    //
    //    q = | q_block | gamma |
    //        |  lambda | delta |
    //
    //-------------------------------------
    std::shared_ptr<Matrix> m_q_block;
    std::shared_ptr<Matrix_Dense<ExecSpace>> m_delta;
    Kokkos::View<double**, typename ExecSpace::memory_space> m_Abm_1_gamma;
    Kokkos::View<double**, typename ExecSpace::memory_space> m_lambda;

public:
    Matrix_Corner_Block(int const n, int const k, std::unique_ptr<Matrix> q)
        : Matrix(n)
        , k(k)
        , nb(n - k)
        , m_q_block(std::move(q))
        , m_delta(new Matrix_Dense<ExecSpace>(k))
        , m_Abm_1_gamma("Abm_1_gamma", k, nb)
        , m_lambda("lambda", nb, k)
    {
        assert(n > 0);
        assert(k >= 0);
        assert(k <= get_size());
        assert(nb == m_q_block->get_size());
    }

    virtual void reset() const override
    {
        m_q_block->reset();
        m_delta->reset();
        // TODO: restore
        /*
        Kokkos::parallel_for(
                "fill_abm_lambda",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {k, nb}),
                KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                    m_Abm_1_gamma(i, j) = 0;
                    m_lambda(j, i) = 0;
                });
				*/
    }

    virtual double get_element(int const i, int const j) const override
    {
        assert(i >= 0);
        assert(i < get_size());
        assert(j >= 0);
        assert(i < get_size());
        if (i < nb && j < nb) {
            return m_q_block->get_element(i, j);
        } else if (i >= nb && j >= nb) {
            return m_delta->get_element(i - nb, j - nb);
        } else if (j >= nb) {
            KOKKOS_IF_ON_HOST(
                    if constexpr (Kokkos::SpaceAccessibility<
                                          Kokkos::DefaultHostExecutionSpace,
                                          typename ExecSpace::memory_space>::accessible) {
                        return m_Abm_1_gamma(j - nb, i);
                    } else {
                        // Inefficient, usage is strongly discouraged
                        double aij;
                        Kokkos::deep_copy(
                                Kokkos::View<double*, Kokkos::HostSpace>(&aij),
                                Kokkos::View<double*, typename ExecSpace::memory_space>(
                                        &m_Abm_1_gamma(j - nb, i)));
                        return aij;
                    })
            KOKKOS_IF_ON_DEVICE(return m_Abm_1_gamma(j - nb, i);)
        } else {
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
        }
    }
    virtual void set_element(int const i, int const j, double const aij) const override
    {
        assert(i >= 0);
        assert(i < get_size());
        assert(j >= 0);
        assert(i < get_size());
        if (i < nb && j < nb) {
            m_q_block->set_element(i, j, aij);
        } else if (i >= nb && j >= nb) {
            m_delta->set_element(i - nb, j - nb, aij);
        } else if (j >= nb) {
            KOKKOS_IF_ON_HOST(
                    if constexpr (Kokkos::SpaceAccessibility<
                                          Kokkos::DefaultHostExecutionSpace,
                                          typename ExecSpace::memory_space>::accessible) {
                        m_Abm_1_gamma(j - nb, i) = aij;
                    } else {
                        // Inefficient, usage is strongly discouraged
                        Kokkos::deep_copy(
                                Kokkos::View<double*, typename ExecSpace::memory_space>(
                                        &m_Abm_1_gamma(j - nb, i)),
                                Kokkos::View<const double*, Kokkos::HostSpace>(&aij));
                    })
            KOKKOS_IF_ON_DEVICE(m_Abm_1_gamma(j - nb, i) = aij;)
        } else {
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
        }
    }
    virtual void factorize() override
    {
        m_q_block->factorize();
        m_q_block->solve_multiple_inplace(
                detail::build_mdspan(m_Abm_1_gamma, std::make_index_sequence<2> {}));

        calculate_delta_to_factorize();

        m_delta->factorize();
    }
    virtual ddc::DSpan1D solve_inplace(ddc::DSpan1D const bx) const override
    {
        assert(int(bx.extent(0)) == get_size());

        ddc::DSpan1D const u(bx.data_handle(), nb);
        ddc::DSpan1D const v(bx.data_handle() + nb, k);

        m_q_block->solve_inplace(u);

        solve_lambda_section(v, u);

        m_delta->solve_inplace(v);

        solve_gamma_section(u, v);

        return bx;
    }
    virtual ddc::DSpan1D solve_transpose_inplace(ddc::DSpan1D const bx) const override
    {
        assert(int(bx.extent(0)) == get_size());
        ddc::DSpan1D const u(bx.data_handle(), nb);
        ddc::DSpan1D const v(bx.data_handle() + nb, k);

        solve_gamma_section_transpose(v, u);

        m_delta->solve_transpose_inplace(v);

        solve_lambda_section_transpose(u, v);

        m_q_block->solve_transpose_inplace(u);

        return bx;
    }
    virtual ddc::DSpan2D solve_multiple_inplace(ddc::DSpan2D const bx) const override
    {
        assert(int(bx.extent(0)) == get_size());
        for (std::size_t i(0); i < bx.extent(0); ++i) {
            ddc::DSpan1D const b(bx.data_handle() + get_size() * i, get_size());
            solve_inplace(b);
        }
        return bx;
    }

protected:
    Matrix_Corner_Block(
            int const n,
            int const k,
            std::unique_ptr<Matrix> q,
            int const lambda_size1,
            int const lambda_size2)
        : Matrix(n)
        , k(k)
        , nb(n - k)
        , m_q_block(std::move(q))
        , m_delta(new Matrix_Dense<ExecSpace>(k))
        , m_Abm_1_gamma("Abm_1_gamma", k, nb)
        , m_lambda("lambda", lambda_size1, lambda_size2)
    {
        assert(n > 0);
        assert(k >= 0);
        assert(k <= n);
        assert(nb == m_q_block->get_size());
    }
    virtual void calculate_delta_to_factorize()
    {
        Kokkos::parallel_for(
                "calculate_delta_to_factorize",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {k, k}),
                KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                    double val = 0.0;
                    for (int l = 0; l < nb; ++l) {
                        val += m_lambda(l, i) * m_Abm_1_gamma(j, l);
                    }
                    m_delta->set_element(i, j, m_delta->get_element(i, j) - val);
                });
    }
    virtual ddc::DSpan1D solve_lambda_section(ddc::DSpan1D const v, DView1D const u) const
    {
        Kokkos::parallel_for(
                "solve_lambda_section",
                Kokkos::RangePolicy<ExecSpace>(0, k),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    // Upper diagonals in lambda
                    for (int j = 0; j < nb; ++j) {
                        Kokkos::atomic_sub(&v(i), m_lambda(j, i) * u(j));
                    }
                });
        return v;
    }
    virtual ddc::DSpan1D solve_lambda_section_transpose(ddc::DSpan1D const u, DView1D const v) const
    {
        Kokkos::parallel_for(
                "solve_lambda_section_transpose",
                Kokkos::RangePolicy<ExecSpace>(0, nb),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    // Upper diagonals in lambda
                    for (int j = 0; j < k; ++j) {
                        Kokkos::atomic_sub(&u(i), m_lambda(i, j) * v(j));
                    }
                });
        return u;
    }
    virtual ddc::DSpan1D solve_gamma_section(ddc::DSpan1D const u, DView1D const v) const
    {
        Kokkos::parallel_for(
                "solve_gamma_section",
                Kokkos::RangePolicy<ExecSpace>(0, nb),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    for (int j = 0; j < k; ++j) {
                        Kokkos::atomic_sub(&u(i), m_Abm_1_gamma(j, i) * v(j));
                    }
                });
        return u;
    }
    virtual ddc::DSpan1D solve_gamma_section_transpose(ddc::DSpan1D const v, DView1D const u) const
    {
        Kokkos::parallel_for(
                "solve_gamma_section_transpose",
                Kokkos::RangePolicy<ExecSpace>(0, k),
                KOKKOS_CLASS_LAMBDA(const int j) {
                    for (int i = 0; i < nb; ++i) {
                        Kokkos::atomic_sub(&v(j), m_Abm_1_gamma(j, i) * u(i));
                    }
                });
        return v;
    }

protected:
    virtual int factorize_method() override
    {
        return 0;
    }
    virtual int solve_inplace_method(
            double* const b,
            char const transpose,
            int const n_equations,
            int const stride) const override
    {
        for (std::size_t i(0); i < (std::size_t)n_equations; ++i) {
            ddc::DSpan1D const u(b + i * stride, nb);
            ddc::DSpan1D const v(b + i * stride + nb, k);

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
        }
        return 0;
    }
};

} // namespace ddc::detail
#endif // MATRIX_CORNER_BLOCK_H
