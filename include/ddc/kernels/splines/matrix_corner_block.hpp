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
class Matrix_Corner_Block : public Matrix
{
public:
    Matrix_Corner_Block(int const n, int const k, std::unique_ptr<Matrix> q)
        : Matrix(n)
        , k(k)
        , nb(n - k)
        , Abm_1_gamma_ptr(std::make_unique<double[]>(k * nb))
        , lambda_ptr(std::make_unique<double[]>(k * nb))
        , q_block(std::move(q))
        , delta(k)
        , Abm_1_gamma(Abm_1_gamma_ptr.get(), k, nb)
        , lambda(lambda_ptr.get(), nb, k)
    {
        assert(n > 0);
        assert(k >= 0);
        assert(k <= n);
        assert(nb == q_block->get_size());
        memset(lambda_ptr.get(), 0, sizeof(double) * k * nb);
        memset(Abm_1_gamma_ptr.get(), 0, sizeof(double) * k * nb);
    }
    virtual double get_element(int const i, int const j) const override
    {
        assert(i >= 0);
        assert(i < n);
        assert(j >= 0);
        assert(i < n);
        if (i < nb && j < nb) {
            return q_block->get_element(i, j);
        } else if (i >= nb && j >= nb) {
            return delta.get_element(i - nb, j - nb);
        } else if (j >= nb) {
            return Abm_1_gamma(j - nb, i);
        } else {
            return lambda(j, i - nb);
        }
    }
    virtual void set_element(int const i, int const j, double const a_ij) override
    {
        assert(i >= 0);
        assert(i < n);
        assert(j >= 0);
        assert(i < n);
        if (i < nb && j < nb) {
            q_block->set_element(i, j, a_ij);
        } else if (i >= nb && j >= nb) {
            delta.set_element(i - nb, j - nb, a_ij);
        } else if (j >= nb) {
            Abm_1_gamma(j - nb, i) = a_ij;
        } else {
            lambda(j, i - nb) = a_ij;
        }
    }
    virtual void factorize() override
    {
        q_block->factorize();
        q_block->solve_multiple_inplace(Abm_1_gamma);

        calculate_delta_to_factorize();

        delta.factorize();
    }
    virtual DSpan1D solve_inplace(DSpan1D const bx) const override
    {
        assert(int(bx.extent(0)) == n);

        DSpan1D const u(bx.data_handle(), nb);
        DSpan1D const v(bx.data_handle() + nb, k);

        q_block->solve_inplace(u);

        solve_lambda_section(v, u);

        delta.solve_inplace(v);

        solve_gamma_section(u, v);

        return bx;
    }
    virtual DSpan1D solve_transpose_inplace(DSpan1D const bx) const override
    {
        assert(int(bx.extent(0)) == n);
        DSpan1D const u(bx.data_handle(), nb);
        DSpan1D const v(bx.data_handle() + nb, k);

        solve_gamma_section_transpose(v, u);

        delta.solve_transpose_inplace(v);

        solve_lambda_section_transpose(u, v);

        q_block->solve_transpose_inplace(u);

        return bx;
    }
    virtual DSpan2D solve_multiple_inplace(DSpan2D const bx) const override
    {
        assert(int(bx.extent(0)) == n);
        for (std::size_t i(0); i < bx.extent(0); ++i) {
            DSpan1D const b(bx.data_handle() + n * i, n);
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
        , Abm_1_gamma_ptr(std::make_unique<double[]>(k * nb))
        , lambda_ptr(std::make_unique<double[]>(lambda_size1 * lambda_size2))
        , q_block(std::move(q))
        , delta(k)
        , Abm_1_gamma(Abm_1_gamma_ptr.get(), k, nb)
        , lambda(lambda_ptr.get(), lambda_size1, lambda_size2)
    {
        assert(n > 0);
        assert(k >= 0);
        assert(k <= n);
        assert(nb == q_block->get_size());
        memset(lambda_ptr.get(), 0, sizeof(double) * lambda_size1 * lambda_size2);
        memset(Abm_1_gamma_ptr.get(), 0, sizeof(double) * k * nb);
    }
    virtual void calculate_delta_to_factorize()
    {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                double val = 0.0;
                for (int l = 0; l < nb; ++l) {
                    val += lambda(l, i) * Abm_1_gamma(j, l);
                }
                delta.set_element(i, j, delta.get_element(i, j) - val);
            }
        }
    }
    virtual DSpan1D solve_lambda_section(DSpan1D const v, DView1D const u) const
    {
        for (int i = 0; i < k; ++i) {
            // Upper diagonals in lambda
            for (int j = 0; j < nb; ++j) {
                v(i) -= lambda(j, i) * u(j);
            }
        }
        return v;
    }
    virtual DSpan1D solve_lambda_section_transpose(DSpan1D const u, DView1D const v) const
    {
        for (int i = 0; i < nb; ++i) {
            // Upper diagonals in lambda
            for (int j = 0; j < k; ++j) {
                u(i) -= lambda(i, j) * v(j);
            }
        }
        return u;
    }
    virtual DSpan1D solve_gamma_section(DSpan1D const u, DView1D const v) const
    {
        for (int i = 0; i < nb; ++i) {
            double val = 0.;
            for (int j = 0; j < k; ++j) {
                val += Abm_1_gamma(j, i) * v(j);
            }
            u(i) -= val;
        }
        return u;
    }
    virtual DSpan1D solve_gamma_section_transpose(DSpan1D const v, DView1D const u) const
    {
        for (int j = 0; j < k; ++j) {
            double val = 0.;
            for (int i = 0; i < nb; ++i) {
                val += Abm_1_gamma(j, i) * u(i);
            }
            v(j) -= val;
        }
        return v;
    }
    int const k; // small block size
    int const nb; // main block matrix size
    std::unique_ptr<double[]> Abm_1_gamma_ptr;
    std::unique_ptr<double[]> lambda_ptr;
    //-------------------------------------
    //
    //    q = | q_block | gamma |
    //        |  lambda | delta |
    //
    //-------------------------------------
    std::unique_ptr<Matrix> q_block;
    Matrix_Dense delta;
    DSpan2D Abm_1_gamma;
    DSpan2D lambda;

private:
    virtual int factorize_method() override
    {
        return 0;
    }
    virtual int solve_inplace_method(double*, char, int) const override
    {
        return 0;
    }
};

} // namespace ddc::detail
#endif // MATRIX_CORNER_BLOCK_H