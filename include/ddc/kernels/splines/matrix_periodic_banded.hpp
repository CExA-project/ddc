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
class Matrix;

template <class ExecSpace>
class Matrix_Periodic_Banded : public Matrix_Corner_Block
{
public:
    Matrix_Periodic_Banded(int const n, int const kl, int const ku, std::unique_ptr<Matrix> q)
        : Matrix_Corner_Block(
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
            if (d > 0)
                return m_lambda(j, i - nb);
            else
                return m_lambda(j - nb + k + 1, i - nb);
        } else {
            return Matrix_Corner_Block::get_element(i, j);
        }
    }
    void set_element(int const i, int j, double const a_ij) override
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
                assert(std::fabs(a_ij) < 1e-20);
                return;
            }

            if (d > 0)
                m_lambda(j, i - nb) = a_ij;
            else
                m_lambda(j - nb + k + 1, i - nb) = a_ij;
        } else {
            Matrix_Corner_Block::set_element(i, j, a_ij);
        }
    }

protected:
    void calculate_delta_to_factorize() override
    {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
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
                delta.set_element(i, j, delta.get_element(i, j) - val);
            }
        }
    }
    ddc::DSpan1D solve_lambda_section(ddc::DSpan1D const v, DView1D const u) const override
    {
        for (int i = 0; i < k; ++i) {
            // Upper diagonals in lambda
            for (int j = 0; j <= i; ++j) {
                v(i) -= m_lambda(j, i) * u(j);
            }
            // Lower diagonals in lambda
            for (int j = i + 1; j < k + 1; ++j) {
                v(i) -= m_lambda(j, i) * u(nb - 1 - k + j);
            }
        }
        return v;
    }
    ddc::DSpan1D solve_lambda_section_transpose(ddc::DSpan1D const u, DView1D const v)
            const override
    {
        for (int i = 0; i < k; ++i) {
            // Upper diagonals in lambda
            for (int j = 0; j <= i; ++j) {
                u(j) -= m_lambda(j, i) * v(i);
            }
            // Lower diagonals in lambda
            for (int j = i + 1; j < k + 1; ++j) {
                u(nb - 1 - k + j) -= m_lambda(j, i) * v(i);
            }
        }
        return u;
    }
    int const kl; // no. of subdiagonals
    int const ku; // no. of superdiagonals
};

} // namespace ddc::detail
#endif // MATRIX_PERIODIC_BANDED_H
